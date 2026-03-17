// Copyright contributors to the Bicycle Architecture Compiler project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};
use std::fs;
use std::path::Path;

use bicycle_common::{AutomorphismData, Pauli};
use serde::{Deserialize, Serialize};

use crate::decomposition::{CompleteMeasurementTable, MeasurementTableBuilder};
use crate::draft_core::{native_rows_for_code, transition};
use crate::draft_types::{
    Head, NativeRow, NativeRowMetadata, Tail11, TAIL_MASK, TAIL_QUBITS, TAIL_SPACE_SIZE,
};
use crate::measurement::CodeMeasurement;
use crate::measurement::MeasurementChoices;
use crate::measurement::{GROSS_MEASUREMENT, TWOGROSS_MEASUREMENT};
use crate::native_measurement::NativeMeasurement;
use crate::PauliString;

const HEAD_COUNT: usize = 4;
const TOTAL_QUBITS: usize = 12;
const BLOCK_QUBITS: usize = 6;
const FIXED_PIVOT_INDEX: usize = 0;
const FIXED_PAIR_INDEX: usize = 6;
const NON_I_HEADS: [Head; 3] = [Head::X, Head::Y, Head::Z];
const INF_COST: u32 = u32::MAX / 8;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum SingleShotSourceRule {
    P0I,
    P0Q,
    P2,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum SingleShotSourceWitness {
    P0I {
        native_index: usize,
        tail: Tail11,
        total_cost: u32,
    },
    P0Q {
        q_basis: Head,
        native_index: usize,
        tail: Tail11,
        total_cost: u32,
    },
    P2 {
        q_basis: Head,
        left_tail: Tail11,
        right_tail: Tail11,
        left_native_index: usize,
        right_native_index: usize,
        total_cost: u32,
    },
}

impl SingleShotSourceWitness {
    fn total_cost(self) -> u32 {
        match self {
            Self::P0I { total_cost, .. }
            | Self::P0Q { total_cost, .. }
            | Self::P2 { total_cost, .. } => total_cost,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum SingleShotPathWitness {
    Source(SingleShotSourceWitness),
    P3 {
        prev_head: Head,
        prev_tail: Tail11,
        q_basis: Head,
        axis_tail: Tail11,
        next_head: Head,
        delta_local_measurement_count: u8,
        native_index: usize,
    },
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct SingleShotPathNode {
    pub head: Head,
    pub tail: Tail11,
    pub total_cost: u32,
    pub witness: SingleShotPathWitness,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct OursSingleShotTargetExplain {
    pub target_tail: Tail11,
    pub best_head: Head,
    pub total_cost: u32,
    pub path: Vec<SingleShotPathNode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactHistogramReport {
    pub histogram: BTreeMap<u32, u64>,
    pub total_targets: u64,
    pub reachable_targets: u64,
    pub unreachable_targets: u64,
    pub support: Vec<u32>,
    pub median: Option<u32>,
    pub mean: Option<f64>,
    pub sample_unreachable_tails: Vec<Tail11>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafePivotTransformationReport {
    pub name: String,
    pub shift_x: u8,
    pub shift_y: u8,
    pub pivot_image: Option<u8>,
    pub pair_image: Option<[u8; 2]>,
    pub pivot_image_preserved: bool,
    pub lpu_pair_preserved: bool,
    pub p0_grammar_preserved: bool,
    pub p2_grammar_preserved: bool,
    pub p3_transition_preserved: bool,
    pub accepted: bool,
    pub failure_reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafePivotCertificationReport {
    pub candidate_qubits: Vec<u8>,
    pub accepted_qubits: Vec<u8>,
    pub rejected_qubits: Vec<u8>,
    pub rejection_reasons: BTreeMap<u8, Vec<String>>,
    pub admissible_transformations_used: Vec<SafePivotTransformationReport>,
    pub examined_transformations: Vec<SafePivotTransformationReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafePivotSummaryReport {
    pub j_safe: Vec<u8>,
    pub mean_fixed_pivot_cost: Option<f64>,
    pub mean_safe_pivot_cost: Option<f64>,
    pub total_count: u64,
    pub fixed_pivot_support: Vec<u32>,
    pub safe_pivot_support: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafePivotExperimentReport {
    pub fixed_pivot: ExactHistogramReport,
    pub safe_pivot: ExactHistogramReport,
    pub certification: SafePivotCertificationReport,
    pub summary: SafePivotSummaryReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PivotScanPivotSummary {
    pub pivot_index: u8,
    pub validation_passed: bool,
    pub rejection_reasons: Vec<String>,
    pub total_targets: u64,
    pub reachable_targets: u64,
    pub mean: Option<f64>,
    pub support: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PivotScanSummaryReport {
    pub j_emp: Vec<u8>,
    pub j_valid: Vec<u8>,
    pub pivots: Vec<PivotScanPivotSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BestSinglePivotSummaryReport {
    pub j_valid: Vec<u8>,
    pub best_pivot_by_mean: Option<u8>,
    pub fixed_pivot_mean: Option<f64>,
    pub best_single_pivot_mean: Option<f64>,
    pub improvement_over_fixed: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PivotScanExperimentReport {
    pub per_pivot_exact_hist: BTreeMap<u8, ExactHistogramReport>,
    pub pivot_scan_summary: PivotScanSummaryReport,
    pub fixed_pivot_exact_hist: ExactHistogramReport,
    pub best_single_pivot_exact_hist: ExactHistogramReport,
    pub best_single_pivot_summary: BestSinglePivotSummaryReport,
}

#[derive(Debug, Clone, Copy)]
struct AxisInfo {
    q_basis: Head,
    q_basis_idx: usize,
    tail: Tail11,
    tail_bits: u32,
    x_bits: u32,
    z_bits: u32,
    native_index: usize,
}

#[derive(Debug, Clone)]
struct SourceRecord {
    total_cost: u32,
    witness: SingleShotSourceWitness,
}

#[derive(Debug, Clone)]
struct SingleShotModel {
    axes: Vec<AxisInfo>,
    source_states: HashMap<usize, SourceRecord>,
    transition_table: [[(usize, u8); 3]; 4],
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct HeapItem {
    total_cost: u32,
    state_idx: usize,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .total_cost
            .cmp(&self.total_cost)
            .then_with(|| self.state_idx.cmp(&other.state_idx))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Copy)]
enum PathPred {
    Source(SingleShotSourceWitness),
    P3 {
        prev_state_idx: usize,
        q_basis: Head,
        axis_tail: Tail11,
        delta_local_measurement_count: u8,
        native_index: usize,
    },
}

#[derive(Debug, Clone, Copy)]
struct SymmetryAction {
    shift_x: u8,
    shift_y: u8,
    x_rows: [u8; BLOCK_QUBITS],
    z_rows: [u8; BLOCK_QUBITS],
}

#[derive(Debug, Clone)]
struct CandidateCheckResult {
    report: SafePivotTransformationReport,
    pivot_image_index: Option<usize>,
}

#[derive(Debug, Clone)]
struct CertifiedSafePivots {
    report: SafePivotCertificationReport,
    canonical_actions: BTreeMap<usize, SymmetryAction>,
}

fn head_index(head: Head) -> usize {
    match head {
        Head::I => 0,
        Head::X => 1,
        Head::Y => 2,
        Head::Z => 3,
    }
}

fn head_from_index(index: usize) -> Head {
    match index {
        0 => Head::I,
        1 => Head::X,
        2 => Head::Y,
        3 => Head::Z,
        _ => panic!("Invalid head index: {index}"),
    }
}

fn q_basis_index(q_basis: Head) -> usize {
    match q_basis {
        Head::X => 0,
        Head::Y => 1,
        Head::Z => 2,
        Head::I => panic!("q_basis_index requires non-I head"),
    }
}

fn state_index(head: Head, tail_bits: u32) -> usize {
    head_index(head) * TAIL_SPACE_SIZE as usize + tail_bits as usize
}

fn decode_state(state_idx: usize) -> (Head, u32) {
    let head_idx = state_idx / TAIL_SPACE_SIZE as usize;
    let tail_bits = (state_idx % TAIL_SPACE_SIZE as usize) as u32;
    (head_from_index(head_idx), tail_bits)
}

fn tails_commute(a: u32, b: u32) -> bool {
    let a_x = a & TAIL_MASK;
    let a_z = a >> TAIL_QUBITS;
    let b_x = b & TAIL_MASK;
    let b_z = b >> TAIL_QUBITS;
    (((a_x & b_z).count_ones() + (a_z & b_x).count_ones()) & 1) == 0
}

fn tails_anticommute(a_x: u32, a_z: u32, b_x: u32, b_z: u32) -> bool {
    (((a_x & b_z).count_ones() + (a_z & b_x).count_ones()) & 1) == 1
}

fn head_to_pauli(head: Head) -> Pauli {
    match head {
        Head::I => Pauli::I,
        Head::X => Pauli::X,
        Head::Y => Pauli::Y,
        Head::Z => Pauli::Z,
    }
}

fn pauli_to_head(pauli: Pauli) -> Head {
    match pauli {
        Pauli::I => Head::I,
        Pauli::X => Head::X,
        Pauli::Y => Head::Y,
        Pauli::Z => Head::Z,
    }
}

fn pauli_from_symplectic_bits(has_x: bool, has_z: bool) -> Pauli {
    match (has_x, has_z) {
        (false, false) => Pauli::I,
        (true, false) => Pauli::X,
        (false, true) => Pauli::Z,
        (true, true) => Pauli::Y,
    }
}

fn single_qubit_pauli(qubit: usize, pauli: Pauli) -> [Pauli; TOTAL_QUBITS] {
    let mut out = [Pauli::I; TOTAL_QUBITS];
    out[qubit] = pauli;
    out
}

fn full_paulis_from_native_row(row: NativeRow, pivot_index: usize) -> [Pauli; TOTAL_QUBITS] {
    assert!(pivot_index < TOTAL_QUBITS);
    let mut out = [Pauli::I; TOTAL_QUBITS];
    out[pivot_index] = head_to_pauli(row.head);
    let tail_paulis = row.tail.to_paulis();
    let mut tail_i = 0usize;
    for (qubit, slot) in out.iter_mut().enumerate() {
        if qubit == pivot_index {
            continue;
        }
        *slot = tail_paulis[tail_i];
        tail_i += 1;
    }
    out
}

fn native_row_from_full_paulis(
    index: usize,
    paulis: [Pauli; TOTAL_QUBITS],
    pivot_index: usize,
) -> NativeRow {
    assert!(pivot_index < TOTAL_QUBITS);
    let head = pauli_to_head(paulis[pivot_index]);
    let mut tail_paulis = [Pauli::I; TAIL_QUBITS as usize];
    let mut tail_i = 0usize;
    for (qubit, pauli) in paulis.into_iter().enumerate() {
        if qubit == pivot_index {
            continue;
        }
        tail_paulis[tail_i] = pauli;
        tail_i += 1;
    }

    NativeRow {
        index,
        head,
        tail: Tail11::from_paulis(tail_paulis),
        metadata: None,
    }
}

fn full_paulis_from_tail_with_pivot_head(
    tail: Tail11,
    pivot_index: usize,
    pivot_head: Head,
) -> [Pauli; TOTAL_QUBITS] {
    let row = NativeRow {
        index: 0,
        head: pivot_head,
        tail,
        metadata: None,
    };
    full_paulis_from_native_row(row, pivot_index)
}

fn apply_rows_to_bits(rows: &[u8; BLOCK_QUBITS], bits: u8) -> u8 {
    let mut out = 0u8;
    for (row_idx, row_mask) in rows.iter().copied().enumerate() {
        let parity = (row_mask & bits).count_ones() & 1;
        out |= (parity as u8) << row_idx;
    }
    out
}

fn extract_block_bits(paulis: &[Pauli; TOTAL_QUBITS], offset: usize) -> (u8, u8) {
    let mut x = 0u8;
    let mut z = 0u8;
    for i in 0..BLOCK_QUBITS {
        let bit = 1u8 << i;
        match paulis[offset + i] {
            Pauli::I => {}
            Pauli::X => x |= bit,
            Pauli::Z => z |= bit,
            Pauli::Y => {
                x |= bit;
                z |= bit;
            }
        }
    }
    (x, z)
}

fn write_block_bits(paulis: &mut [Pauli; TOTAL_QUBITS], offset: usize, x: u8, z: u8) {
    for i in 0..BLOCK_QUBITS {
        let bit = 1u8 << i;
        let has_x = x & bit != 0;
        let has_z = z & bit != 0;
        paulis[offset + i] = pauli_from_symplectic_bits(has_x, has_z);
    }
}

impl SymmetryAction {
    fn from_shift(code: CodeMeasurement, shift_x: u8, shift_y: u8) -> Self {
        let shift = AutomorphismData::new(shift_x, shift_y);
        let action = |a: AutomorphismData| {
            (code.mx.pow(u32::from(a.get_x())) * code.my.pow(u32::from(a.get_y()))).map(|v| v % 2)
        };

        let x_action = action(shift.inv());
        let z_action = action(shift).transpose();

        let mut x_rows = [0u8; BLOCK_QUBITS];
        let mut z_rows = [0u8; BLOCK_QUBITS];
        for row in 0..BLOCK_QUBITS {
            let mut x_mask = 0u8;
            let mut z_mask = 0u8;
            for col in 0..BLOCK_QUBITS {
                if x_action[(row, col)] & 1 == 1 {
                    x_mask |= 1u8 << col;
                }
                if z_action[(row, col)] & 1 == 1 {
                    z_mask |= 1u8 << col;
                }
            }
            x_rows[row] = x_mask;
            z_rows[row] = z_mask;
        }

        Self {
            shift_x,
            shift_y,
            x_rows,
            z_rows,
        }
    }

    fn name(self) -> String {
        format!("shift({}, {})", self.shift_x, self.shift_y)
    }

    fn apply_full_paulis(self, paulis: [Pauli; TOTAL_QUBITS]) -> [Pauli; TOTAL_QUBITS] {
        let mut out = [Pauli::I; TOTAL_QUBITS];

        let (x_first, z_first) = extract_block_bits(&paulis, 0);
        let (x_second, z_second) = extract_block_bits(&paulis, BLOCK_QUBITS);

        let mapped_x_first = apply_rows_to_bits(&self.x_rows, x_first);
        let mapped_z_first = apply_rows_to_bits(&self.z_rows, z_first);
        let mapped_x_second = apply_rows_to_bits(&self.x_rows, x_second);
        let mapped_z_second = apply_rows_to_bits(&self.z_rows, z_second);

        write_block_bits(&mut out, 0, mapped_x_first, mapped_z_first);
        write_block_bits(&mut out, BLOCK_QUBITS, mapped_x_second, mapped_z_second);

        out
    }
}

fn single_axis_location(paulis: &[Pauli; TOTAL_QUBITS], axis: Pauli) -> Option<usize> {
    if !matches!(axis, Pauli::X | Pauli::Z) {
        return None;
    }
    let mut location = None;
    for (idx, pauli) in paulis.iter().copied().enumerate() {
        match pauli {
            Pauli::I => {}
            p if p == axis => {
                if location.is_some() {
                    return None;
                }
                location = Some(idx);
            }
            _ => return None,
        }
    }
    location
}

fn transport_row_from_fixed_pivot(
    action: SymmetryAction,
    row: NativeRow,
    target_pivot_index: usize,
) -> NativeRow {
    let full = full_paulis_from_native_row(row, FIXED_PIVOT_INDEX);
    let mapped = action.apply_full_paulis(full);
    native_row_from_full_paulis(row.index, mapped, target_pivot_index)
}

fn transport_head_from_fixed_pivot(
    action: SymmetryAction,
    target_pivot_index: usize,
    head: Head,
) -> Option<Head> {
    let mapped =
        action.apply_full_paulis(single_qubit_pauli(FIXED_PIVOT_INDEX, head_to_pauli(head)));
    let mapped_row = native_row_from_full_paulis(0, mapped, target_pivot_index);
    if mapped_row.tail != Tail11::new(0) {
        return None;
    }
    Some(mapped_row.head)
}

fn certify_pivot_image(action: SymmetryAction) -> Result<usize, String> {
    let mapped_x = action.apply_full_paulis(single_qubit_pauli(FIXED_PIVOT_INDEX, Pauli::X));
    let mapped_z = action.apply_full_paulis(single_qubit_pauli(FIXED_PIVOT_INDEX, Pauli::Z));

    let x_loc = single_axis_location(&mapped_x, Pauli::X).ok_or_else(|| {
        "criterion 3.1 failed: X_1 is not mapped to a single-qubit X image".to_string()
    })?;
    let z_loc = single_axis_location(&mapped_z, Pauli::Z).ok_or_else(|| {
        "criterion 3.1 failed: Z_1 is not mapped to a single-qubit Z image".to_string()
    })?;

    if x_loc != z_loc {
        return Err(format!(
            "criterion 3.1 failed: X_1 and Z_1 map to different qubits ({} vs {})",
            x_loc + 1,
            z_loc + 1
        ));
    }

    Ok(x_loc)
}

fn certify_lpu_pair_structure(
    action: SymmetryAction,
    pivot_image_index: usize,
) -> Result<usize, String> {
    let mapped_x = action.apply_full_paulis(single_qubit_pauli(FIXED_PAIR_INDEX, Pauli::X));
    let mapped_z = action.apply_full_paulis(single_qubit_pauli(FIXED_PAIR_INDEX, Pauli::Z));

    let x_loc = single_axis_location(&mapped_x, Pauli::X).ok_or_else(|| {
        "criterion 3.2 failed: X_7 is not mapped to a single-qubit X image".to_string()
    })?;
    let z_loc = single_axis_location(&mapped_z, Pauli::Z).ok_or_else(|| {
        "criterion 3.2 failed: Z_7 is not mapped to a single-qubit Z image".to_string()
    })?;

    if x_loc != z_loc {
        return Err(format!(
            "criterion 3.2 failed: X_7 and Z_7 map to different qubits ({} vs {})",
            x_loc + 1,
            z_loc + 1
        ));
    }

    if pivot_image_index >= BLOCK_QUBITS {
        return Err(format!(
            "criterion 3.2 failed: pivot image qubit {} is outside the primal block",
            pivot_image_index + 1
        ));
    }

    let expected_pair = pivot_image_index + BLOCK_QUBITS;
    if x_loc != expected_pair {
        return Err(format!(
            "criterion 3.2 failed: pair image is {} but expected {} for pivot image {}",
            x_loc + 1,
            expected_pair + 1,
            pivot_image_index + 1
        ));
    }

    Ok(x_loc)
}

fn certify_p0_grammar(
    base_rows: &[NativeRow],
    target_rows: &[NativeRow],
    target_pivot_index: usize,
    action: SymmetryAction,
) -> Result<Vec<NativeRow>, String> {
    let target_sig = signature(target_rows);
    let mut transported = Vec::with_capacity(base_rows.len());

    for row in base_rows.iter().copied() {
        let mapped = transport_row_from_fixed_pivot(action, row, target_pivot_index);
        if row.head == Head::I && mapped.head != Head::I {
            return Err(format!(
                "criterion 3.3 failed: P0_I row {} changed head I -> {}",
                row.index, mapped.head
            ));
        }
        if row.head != Head::I && mapped.head != row.head {
            return Err(format!(
                "criterion 3.3 failed: P0_Q row {} changed head {} -> {}",
                row.index, row.head, mapped.head
            ));
        }
        if !target_sig.contains(&(mapped.head, mapped.tail)) {
            return Err(format!(
                "criterion 3.3 failed: transported row {} is not legal in pivot {} native family",
                row.index,
                target_pivot_index + 1
            ));
        }
        transported.push(mapped);
    }

    Ok(transported)
}

fn certify_p2_same_basis_grammar(
    base_rows: &[NativeRow],
    transported_rows: &[NativeRow],
) -> Result<(), String> {
    for i in 0..base_rows.len() {
        let left = base_rows[i];
        if left.head == Head::I {
            continue;
        }
        for j in (i + 1)..base_rows.len() {
            let right = base_rows[j];
            if right.head != left.head {
                continue;
            }
            let left_t = transported_rows[i];
            let right_t = transported_rows[j];
            if left_t.head != left.head || right_t.head != right.head {
                return Err(format!(
                    "criterion 3.3 failed: same-basis P2 bucket changed basis for row pair ({}, {})",
                    left.index, right.index
                ));
            }
            let base_comm = left.tail.commutes_with(right.tail);
            let mapped_comm = left_t.tail.commutes_with(right_t.tail);
            if base_comm != mapped_comm {
                return Err(format!(
                    "criterion 3.3 failed: same-basis P2 commutation changed for row pair ({}, {})",
                    left.index, right.index
                ));
            }
        }
    }

    Ok(())
}

fn certify_p3_transitions_with_mapper<F>(mut head_mapper: F) -> Result<(), String>
where
    F: FnMut(Head) -> Option<Head>,
{
    for head in [Head::I, Head::X, Head::Y, Head::Z] {
        let mapped = head_mapper(head).ok_or_else(|| {
            format!(
                "criterion 3.4 failed: head {} does not map to a legal head",
                head
            )
        })?;
        if mapped != head {
            return Err(format!(
                "criterion 3.4 failed: head {} transports to {} (must preserve prev_head semantics)",
                head, mapped
            ));
        }
    }

    for prev_head in [Head::I, Head::X, Head::Y, Head::Z] {
        for q_basis in NON_I_HEADS {
            let mapped_prev = head_mapper(prev_head).ok_or_else(|| {
                format!(
                    "criterion 3.4 failed: prev_head {} does not transport as a legal head",
                    prev_head
                )
            })?;
            let mapped_q = head_mapper(q_basis).ok_or_else(|| {
                format!(
                    "criterion 3.4 failed: q_basis {} does not transport as a legal head",
                    q_basis
                )
            })?;

            let original = transition(prev_head, q_basis);
            let mapped_next_from_original = head_mapper(original.next_head).ok_or_else(|| {
                format!(
                    "criterion 3.4 failed: next_head {} does not transport as a legal head",
                    original.next_head
                )
            })?;
            let transported = transition(mapped_prev, mapped_q);

            if mapped_prev != prev_head
                || mapped_q != q_basis
                || mapped_next_from_original != original.next_head
            {
                return Err(format!(
                    "criterion 3.4 failed: transition row ({prev_head}, {q_basis}) changed head semantics under transport"
                ));
            }

            if transported.next_head != mapped_next_from_original
                || transported.delta_stage_cost != original.delta_stage_cost
                || transported.delta_local_measurement_count
                    != original.delta_local_measurement_count
            {
                return Err(format!(
                    "criterion 3.4 failed: weighted transition row ({prev_head}, {q_basis}) changed under transport"
                ));
            }
        }
    }

    Ok(())
}

fn certify_p3_transitions(action: SymmetryAction, target_pivot_index: usize) -> Result<(), String> {
    certify_p3_transitions_with_mapper(|head| {
        transport_head_from_fixed_pivot(action, target_pivot_index, head)
    })
}

fn evaluate_shift_candidate(
    action: SymmetryAction,
    rows_by_pivot: &[Vec<NativeRow>],
) -> CandidateCheckResult {
    let name = action.name();
    let base_rows = &rows_by_pivot[FIXED_PIVOT_INDEX];

    let mut failures = Vec::new();
    let mut pivot_image_index = None;
    let mut pair_image_index = None;
    let mut pivot_image_preserved = false;
    let mut lpu_pair_preserved = false;
    let mut p0_grammar_preserved = false;
    let mut p2_grammar_preserved = false;
    let mut p3_transition_preserved = false;

    match certify_pivot_image(action) {
        Ok(idx) => {
            pivot_image_preserved = true;
            pivot_image_index = Some(idx);
        }
        Err(msg) => failures.push(msg),
    }

    if let Some(idx) = pivot_image_index {
        match certify_lpu_pair_structure(action, idx) {
            Ok(pair_idx) => {
                lpu_pair_preserved = true;
                pair_image_index = Some(pair_idx);
            }
            Err(msg) => failures.push(msg),
        }

        if let Some(target_rows) = rows_by_pivot.get(idx) {
            match certify_p0_grammar(base_rows, target_rows, idx, action) {
                Ok(transported_rows) => {
                    p0_grammar_preserved = true;
                    match certify_p2_same_basis_grammar(base_rows, &transported_rows) {
                        Ok(()) => {
                            p2_grammar_preserved = true;
                        }
                        Err(msg) => failures.push(msg),
                    }
                }
                Err(msg) => failures.push(msg),
            }

            match certify_p3_transitions(action, idx) {
                Ok(()) => p3_transition_preserved = true,
                Err(msg) => failures.push(msg),
            }
        } else {
            failures.push(format!(
                "internal error: pivot image {} is outside available pivot range",
                idx + 1
            ));
        }
    }

    let accepted = pivot_image_preserved
        && lpu_pair_preserved
        && p0_grammar_preserved
        && p2_grammar_preserved
        && p3_transition_preserved;

    let report = SafePivotTransformationReport {
        name,
        shift_x: action.shift_x,
        shift_y: action.shift_y,
        pivot_image: pivot_image_index.map(|idx| (idx + 1) as u8),
        pair_image: match (pivot_image_index, pair_image_index) {
            (Some(pivot), Some(pair)) => Some([(pivot + 1) as u8, (pair + 1) as u8]),
            _ => None,
        },
        pivot_image_preserved,
        lpu_pair_preserved,
        p0_grammar_preserved,
        p2_grammar_preserved,
        p3_transition_preserved,
        accepted,
        failure_reasons: failures,
    };

    CandidateCheckResult {
        report,
        pivot_image_index,
    }
}

fn certify_safe_pivots(
    code: CodeMeasurement,
    rows_by_pivot: &[Vec<NativeRow>],
) -> CertifiedSafePivots {
    let mut accepted_qubits = BTreeSet::new();
    let mut per_qubit_reasons: BTreeMap<u8, BTreeSet<String>> = (1..=TOTAL_QUBITS as u8)
        .map(|q| (q, BTreeSet::new()))
        .collect();
    let mut admissible_transformations = Vec::new();
    let mut examined_transformations = Vec::new();
    let mut canonical_actions = BTreeMap::new();

    for shift_x in 0..6u8 {
        for shift_y in 0..6u8 {
            let action = SymmetryAction::from_shift(code, shift_x, shift_y);
            let evaluated = evaluate_shift_candidate(action, rows_by_pivot);
            let report = evaluated.report.clone();
            if report.accepted {
                let pivot_idx = evaluated
                    .pivot_image_index
                    .expect("accepted transformation must have a pivot image");
                accepted_qubits.insert((pivot_idx + 1) as u8);
                canonical_actions.entry(pivot_idx).or_insert(action);
                admissible_transformations.push(report.clone());
            } else if let Some(pivot_idx) = evaluated.pivot_image_index {
                let qubit = (pivot_idx + 1) as u8;
                if let Some(reason_set) = per_qubit_reasons.get_mut(&qubit) {
                    for reason in &report.failure_reasons {
                        reason_set.insert(reason.clone());
                    }
                }
            }
            examined_transformations.push(report);
        }
    }

    let candidate_qubits: Vec<u8> = (1..=TOTAL_QUBITS as u8).collect();
    let accepted_qubits_vec: Vec<u8> = accepted_qubits.iter().copied().collect();

    let mut rejected_qubits = Vec::new();
    let mut rejection_reasons = BTreeMap::new();
    for &qubit in &candidate_qubits {
        if accepted_qubits.contains(&qubit) {
            continue;
        }
        rejected_qubits.push(qubit);
        let reasons: Vec<String> = per_qubit_reasons
            .get(&qubit)
            .into_iter()
            .flat_map(|set| set.iter().cloned())
            .collect();
        if reasons.is_empty() {
            rejection_reasons.insert(
                qubit,
                vec![
                    "criterion 3.1 failed for all trusted shifts: no single-qubit pivot image to this qubit."
                        .to_string(),
                ],
            );
        } else {
            rejection_reasons.insert(qubit, reasons);
        }
    }

    CertifiedSafePivots {
        report: SafePivotCertificationReport {
            candidate_qubits,
            accepted_qubits: accepted_qubits_vec,
            rejected_qubits,
            rejection_reasons,
            admissible_transformations_used: admissible_transformations,
            examined_transformations,
        },
        canonical_actions,
    }
}

pub fn native_rows_for_code_with_pivot(
    code: CodeMeasurement,
    pivot_index: usize,
) -> Vec<NativeRow> {
    assert!(pivot_index < TOTAL_QUBITS);
    NativeMeasurement::all()
        .into_iter()
        .enumerate()
        .map(|(index, native)| {
            let measured = code.measures(&native);
            let measured_paulis: [Pauli; TOTAL_QUBITS] = measured.into();
            let mut row = native_row_from_full_paulis(index, measured_paulis, pivot_index);
            row.metadata = Some(NativeRowMetadata {
                measured_pauli_bits: measured.0,
                logical: native.logical,
                automorphism: native.automorphism,
            });
            row
        })
        .collect()
}

pub fn native_rows_for_code_all_pivots(code: CodeMeasurement) -> Vec<Vec<NativeRow>> {
    (0..TOTAL_QUBITS)
        .map(|pivot_index| native_rows_for_code_with_pivot(code, pivot_index))
        .collect()
}

pub fn infer_measurement_choice_from_csv_path(path: &Path) -> Option<MeasurementChoices> {
    let name = path.file_name()?.to_string_lossy().to_ascii_lowercase();
    if name.contains("two_gross") || name.contains("two-gross") || name.contains("twogross") {
        Some(MeasurementChoices::TwoGross)
    } else if name.contains("gross") {
        Some(MeasurementChoices::Gross)
    } else {
        None
    }
}

fn signature(rows: &[NativeRow]) -> BTreeSet<(Head, Tail11)> {
    rows.iter().map(|row| (row.head, row.tail)).collect()
}

pub fn infer_measurement_choice_from_native_rows(rows: &[NativeRow]) -> Option<MeasurementChoices> {
    let sig = signature(rows);
    let gross_sig = signature(&native_rows_for_code(GROSS_MEASUREMENT));
    if sig == gross_sig {
        return Some(MeasurementChoices::Gross);
    }

    let two_gross_sig = signature(&native_rows_for_code(TWOGROSS_MEASUREMENT));
    if sig == two_gross_sig {
        return Some(MeasurementChoices::TwoGross);
    }

    None
}

pub fn parse_native_rows_from_csv(
    path: &Path,
    pivot_index: usize,
) -> Result<Vec<NativeRow>, String> {
    if pivot_index >= 12 {
        return Err(format!("pivot_index must be in [0, 11], got {pivot_index}"));
    }

    let contents = fs::read_to_string(path)
        .map_err(|err| format!("failed to read CSV '{}': {err}", path.display()))?;

    let mut rows = Vec::new();
    for (line_no, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let cols: Vec<&str> = line.split(',').map(|field| field.trim()).collect();
        let full_raw = match cols.len() {
            6 => {
                let mut full = String::with_capacity(12);
                full.push_str(cols[4]);
                full.push_str(cols[5]);
                full
            }
            1 => cols[0].to_string(),
            _ => {
                if rows.is_empty() {
                    continue;
                }
                return Err(format!(
                    "unsupported CSV row format at line {} in '{}': {} columns",
                    line_no + 1,
                    path.display(),
                    cols.len()
                ));
            }
        };

        let full: String = full_raw
            .chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| c.to_ascii_uppercase())
            .collect();

        let valid = full.len() == 12 && full.chars().all(|c| matches!(c, 'I' | 'X' | 'Y' | 'Z'));
        if !valid {
            if rows.is_empty() {
                continue;
            }
            return Err(format!(
                "invalid 12-qubit Pauli row at line {} in '{}': '{}'",
                line_no + 1,
                path.display(),
                full_raw
            ));
        }

        let mut chars: Vec<char> = full.chars().collect();
        let head = Head::from_char(chars[pivot_index]).map_err(|err| {
            format!(
                "failed to parse pivot head at line {} in '{}': {}",
                line_no + 1,
                path.display(),
                err
            )
        })?;
        chars.remove(pivot_index);
        let tail_label: String = chars.into_iter().collect();
        let tail = Tail11::from_label(&tail_label).map_err(|err| {
            format!(
                "failed to parse tail at line {} in '{}': {}",
                line_no + 1,
                path.display(),
                err
            )
        })?;

        rows.push(NativeRow {
            index: rows.len(),
            head,
            tail,
            metadata: None,
        });
    }

    if rows.is_empty() {
        return Err(format!("no native rows parsed from '{}'", path.display()));
    }

    Ok(rows)
}

fn paper_total_cost_from_rotation_count(rotation_count: u32) -> u32 {
    // Paper-side 11q comparison metric counts one native measurement plus pre/post
    // native rotations. With native rotations counted as 3 bicycle instructions each:
    // beta(P) = 1 + 2 * k * 3 = 1 + 6k.
    1 + 6 * rotation_count
}

pub fn transition_local(prev_head: Head, q_basis: Head) -> (Head, u8) {
    let outcome = transition(prev_head, q_basis);
    assert_ne!(
        outcome.delta_local_measurement_count, 5,
        "Corrected local transition table must not emit 5"
    );
    assert!(
        outcome.delta_local_measurement_count < 7,
        "Corrected local transition table must not emit values >= 7"
    );
    (outcome.next_head, outcome.delta_local_measurement_count)
}

impl SingleShotModel {
    fn from_native_rows(native_rows: &[NativeRow]) -> Self {
        let mut axis_map: BTreeMap<(Head, Tail11), usize> = BTreeMap::new();
        for row in native_rows.iter().copied() {
            if row.head == Head::I {
                continue;
            }
            axis_map.entry((row.head, row.tail)).or_insert(row.index);
        }

        let mut axes: Vec<AxisInfo> = axis_map
            .into_iter()
            .map(|((q_basis, tail), native_index)| AxisInfo {
                q_basis,
                q_basis_idx: q_basis_index(q_basis),
                tail,
                tail_bits: tail.0,
                x_bits: tail.x_bits(),
                z_bits: tail.z_bits(),
                native_index,
            })
            .collect();
        axes.sort_by_key(|axis| (axis.q_basis_idx, axis.tail_bits));

        let mut source_states: HashMap<usize, SourceRecord> = HashMap::new();

        // P0_I and P0_Q direct templates.
        for row in native_rows.iter().copied() {
            match row.head {
                Head::I => {
                    let idx = state_index(Head::I, row.tail.0);
                    let witness = SingleShotSourceWitness::P0I {
                        native_index: row.index,
                        tail: row.tail,
                        total_cost: 1,
                    };
                    insert_source(&mut source_states, idx, witness);
                }
                q_basis => {
                    let idx = state_index(q_basis, row.tail.0);
                    let witness = SingleShotSourceWitness::P0Q {
                        q_basis,
                        native_index: row.index,
                        tail: row.tail,
                        total_cost: 1,
                    };
                    insert_source(&mut source_states, idx, witness);
                }
            }
        }

        // P2 direct templates: same-basis pairs only.
        for q_basis in NON_I_HEADS {
            let bucket: Vec<_> = axes
                .iter()
                .copied()
                .filter(|axis| axis.q_basis == q_basis)
                .collect();
            for i in 0..bucket.len() {
                for j in (i + 1)..bucket.len() {
                    let left = bucket[i];
                    let right = bucket[j];
                    if !tails_commute(left.tail_bits, right.tail_bits) {
                        continue;
                    }
                    let r = left.tail_bits ^ right.tail_bits;
                    let idx = state_index(Head::I, r);
                    let witness = SingleShotSourceWitness::P2 {
                        q_basis,
                        left_tail: left.tail,
                        right_tail: right.tail,
                        left_native_index: left.native_index,
                        right_native_index: right.native_index,
                        total_cost: 4,
                    };
                    insert_source(&mut source_states, idx, witness);
                }
            }
        }

        let transition_table = build_transition_table();

        Self {
            axes,
            source_states,
            transition_table,
        }
    }
}

fn build_transition_table() -> [[(usize, u8); 3]; 4] {
    let mut table = [[(0usize, 0u8); 3]; 4];
    for prev_head in [Head::I, Head::X, Head::Y, Head::Z] {
        let prev_idx = head_index(prev_head);
        for q_basis in NON_I_HEADS {
            let q_idx = q_basis_index(q_basis);
            let (next_head, delta_local) = transition_local(prev_head, q_basis);
            table[prev_idx][q_idx] = (head_index(next_head), delta_local);
        }
    }
    table
}

fn insert_source(
    source_states: &mut HashMap<usize, SourceRecord>,
    state_idx: usize,
    witness: SingleShotSourceWitness,
) {
    let candidate_cost = witness.total_cost();
    match source_states.get(&state_idx) {
        None => {
            source_states.insert(
                state_idx,
                SourceRecord {
                    total_cost: candidate_cost,
                    witness,
                },
            );
        }
        Some(existing) => {
            if candidate_cost < existing.total_cost {
                source_states.insert(
                    state_idx,
                    SourceRecord {
                        total_cost: candidate_cost,
                        witness,
                    },
                );
            }
        }
    }
}

fn solve_all_state_costs(model: &SingleShotModel) -> Vec<u32> {
    let state_count = HEAD_COUNT * TAIL_SPACE_SIZE as usize;
    let mut dist = vec![INF_COST; state_count];
    let mut heap = BinaryHeap::new();

    for (state_idx, source) in &model.source_states {
        if source.total_cost < dist[*state_idx] {
            dist[*state_idx] = source.total_cost;
            heap.push(HeapItem {
                total_cost: source.total_cost,
                state_idx: *state_idx,
            });
        }
    }

    while let Some(item) = heap.pop() {
        if item.total_cost != dist[item.state_idx] {
            continue;
        }

        let (head, tail_bits) = decode_state(item.state_idx);
        let head_idx = head_index(head);
        let tail_x = tail_bits & TAIL_MASK;
        let tail_z = tail_bits >> TAIL_QUBITS;

        for axis in &model.axes {
            if !tails_anticommute(tail_x, tail_z, axis.x_bits, axis.z_bits) {
                continue;
            }
            let (next_head_idx, delta_local) = model.transition_table[head_idx][axis.q_basis_idx];
            let next_tail_bits = tail_bits ^ axis.tail_bits;
            let next_state_idx = next_head_idx * TAIL_SPACE_SIZE as usize + next_tail_bits as usize;
            let next_cost = item.total_cost + delta_local as u32;
            if next_cost < dist[next_state_idx] {
                dist[next_state_idx] = next_cost;
                heap.push(HeapItem {
                    total_cost: next_cost,
                    state_idx: next_state_idx,
                });
            }
        }
    }

    dist
}

fn beta_for_tail(dist: &[u32], tail: Tail11) -> Option<u32> {
    let mut best = INF_COST;
    for head in NON_I_HEADS {
        let idx = state_index(head, tail.0);
        best = best.min(dist[idx]);
    }
    (best < INF_COST).then_some(best)
}

fn beta_costs_from_distances(dist: &[u32]) -> Vec<Option<u32>> {
    let mut out = Vec::with_capacity(TAIL_SPACE_SIZE as usize);
    for tail_bits in 0..TAIL_SPACE_SIZE {
        out.push(beta_for_tail(dist, Tail11::new(tail_bits)));
    }
    out
}

fn histogram_from_beta_costs(beta_costs: &[Option<u32>]) -> ExactHistogramReport {
    let total_targets = TAIL_SPACE_SIZE as u64;
    let mut histogram = BTreeMap::new();
    let mut reachable_targets = 0u64;
    let mut sample_unreachable_tails = Vec::new();

    for (tail_bits, cost) in beta_costs.iter().copied().enumerate() {
        let tail = Tail11::new(tail_bits as u32);
        match cost {
            Some(cost) => {
                reachable_targets += 1;
                *histogram.entry(cost).or_insert(0) += 1;
            }
            None => {
                if sample_unreachable_tails.len() < 16 {
                    sample_unreachable_tails.push(tail);
                }
            }
        }
    }

    let unreachable_targets = total_targets - reachable_targets;
    let support: Vec<u32> = histogram.keys().copied().collect();
    let mean = if reachable_targets == 0 {
        None
    } else {
        let weighted_sum: u128 = histogram
            .iter()
            .map(|(cost, count)| *cost as u128 * *count as u128)
            .sum();
        Some(weighted_sum as f64 / reachable_targets as f64)
    };
    let median = if reachable_targets == 0 {
        None
    } else {
        let rank = (reachable_targets + 1) / 2;
        let mut running = 0u64;
        let mut med = None;
        for (cost, count) in &histogram {
            running += *count;
            if running >= rank {
                med = Some(*cost);
                break;
            }
        }
        med
    };

    ExactHistogramReport {
        histogram,
        total_targets,
        reachable_targets,
        unreachable_targets,
        support,
        median,
        mean,
        sample_unreachable_tails,
    }
}

fn histogram_from_distances(dist: &[u32]) -> ExactHistogramReport {
    let beta_costs = beta_costs_from_distances(dist);
    histogram_from_beta_costs(&beta_costs)
}

fn explain_target_with_model(
    model: &SingleShotModel,
    target_tail: Tail11,
) -> Option<OursSingleShotTargetExplain> {
    let mut dist: HashMap<usize, u32> = HashMap::new();
    let mut pred: HashMap<usize, PathPred> = HashMap::new();
    let mut heap = BinaryHeap::new();

    for (state_idx, source) in &model.source_states {
        let should_insert = match dist.get(state_idx) {
            None => true,
            Some(existing) => source.total_cost < *existing,
        };
        if should_insert {
            dist.insert(*state_idx, source.total_cost);
            pred.insert(*state_idx, PathPred::Source(source.witness));
            heap.push(HeapItem {
                total_cost: source.total_cost,
                state_idx: *state_idx,
            });
        }
    }

    let mut best_target: Option<(usize, u32)> = None;

    while let Some(item) = heap.pop() {
        let Some(current_cost) = dist.get(&item.state_idx).copied() else {
            continue;
        };
        if current_cost != item.total_cost {
            continue;
        }

        if let Some((_, best_cost)) = best_target {
            if item.total_cost > best_cost {
                break;
            }
        }

        let (head, tail_bits) = decode_state(item.state_idx);
        if tail_bits == target_tail.0 && head != Head::I {
            match best_target {
                None => best_target = Some((item.state_idx, item.total_cost)),
                Some((_, best_cost)) if item.total_cost < best_cost => {
                    best_target = Some((item.state_idx, item.total_cost))
                }
                _ => {}
            }
        }

        let head_idx = head_index(head);
        let tail_x = tail_bits & TAIL_MASK;
        let tail_z = tail_bits >> TAIL_QUBITS;

        for axis in &model.axes {
            if !tails_anticommute(tail_x, tail_z, axis.x_bits, axis.z_bits) {
                continue;
            }

            let (next_head_idx, delta_local) = model.transition_table[head_idx][axis.q_basis_idx];
            let next_tail_bits = tail_bits ^ axis.tail_bits;
            let next_state_idx = next_head_idx * TAIL_SPACE_SIZE as usize + next_tail_bits as usize;
            let next_cost = item.total_cost + delta_local as u32;

            let should_relax = match dist.get(&next_state_idx) {
                None => true,
                Some(existing) => next_cost < *existing,
            };
            if !should_relax {
                continue;
            }

            dist.insert(next_state_idx, next_cost);
            pred.insert(
                next_state_idx,
                PathPred::P3 {
                    prev_state_idx: item.state_idx,
                    q_basis: axis.q_basis,
                    axis_tail: axis.tail,
                    delta_local_measurement_count: delta_local,
                    native_index: axis.native_index,
                },
            );
            heap.push(HeapItem {
                total_cost: next_cost,
                state_idx: next_state_idx,
            });
        }
    }

    let (best_state_idx, best_cost) = best_target?;
    let (best_head, _) = decode_state(best_state_idx);

    let mut reverse_path = Vec::new();
    let mut cursor = best_state_idx;
    loop {
        let total_cost = *dist.get(&cursor)?;
        let (head, tail_bits) = decode_state(cursor);
        let tail = Tail11::new(tail_bits);

        match pred.get(&cursor).copied()? {
            PathPred::Source(source) => {
                reverse_path.push(SingleShotPathNode {
                    head,
                    tail,
                    total_cost,
                    witness: SingleShotPathWitness::Source(source),
                });
                break;
            }
            PathPred::P3 {
                prev_state_idx,
                q_basis,
                axis_tail,
                delta_local_measurement_count,
                native_index,
            } => {
                let (prev_head, prev_tail_bits) = decode_state(prev_state_idx);
                reverse_path.push(SingleShotPathNode {
                    head,
                    tail,
                    total_cost,
                    witness: SingleShotPathWitness::P3 {
                        prev_head,
                        prev_tail: Tail11::new(prev_tail_bits),
                        q_basis,
                        axis_tail,
                        next_head: head,
                        delta_local_measurement_count,
                        native_index,
                    },
                });
                cursor = prev_state_idx;
            }
        }
    }

    reverse_path.reverse();

    Some(OursSingleShotTargetExplain {
        target_tail,
        best_head,
        total_cost: best_cost,
        path: reverse_path,
    })
}

pub fn compute_ours_single_shot_exact_hist_for_code(code: CodeMeasurement) -> ExactHistogramReport {
    let native_rows = native_rows_for_code(code);
    compute_ours_single_shot_exact_hist_from_native_rows(&native_rows)
}

pub fn compute_ours_single_shot_exact_hist_from_native_rows(
    native_rows: &[NativeRow],
) -> ExactHistogramReport {
    let model = SingleShotModel::from_native_rows(native_rows);
    let dist = solve_all_state_costs(&model);
    histogram_from_distances(&dist)
}

fn compute_ours_single_shot_tail_costs_from_native_rows(
    native_rows: &[NativeRow],
) -> Vec<Option<u32>> {
    let model = SingleShotModel::from_native_rows(native_rows);
    let dist = solve_all_state_costs(&model);
    beta_costs_from_distances(&dist)
}

fn option_min_cost(lhs: Option<u32>, rhs: Option<u32>) -> Option<u32> {
    match (lhs, rhs) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn transport_tail_from_fixed_pivot(
    action: SymmetryAction,
    target_pivot_index: usize,
    tail: Tail11,
) -> Option<Tail11> {
    let full = full_paulis_from_tail_with_pivot_head(tail, FIXED_PIVOT_INDEX, Head::I);
    let mapped = action.apply_full_paulis(full);
    let mapped_row = native_row_from_full_paulis(0, mapped, target_pivot_index);
    if mapped_row.head != Head::I {
        return None;
    }
    Some(mapped_row.tail)
}

fn monotonicity_violations(fixed: &[Option<u32>], safe: &[Option<u32>]) -> Vec<Tail11> {
    let mut out = Vec::new();
    for (idx, (fixed_cost, safe_cost)) in fixed.iter().zip(safe.iter()).enumerate() {
        let violates = match (fixed_cost, safe_cost) {
            (Some(fixed_cost), Some(safe_cost)) => safe_cost > fixed_cost,
            (Some(_), None) => true,
            _ => false,
        };
        if violates {
            out.push(Tail11::new(idx as u32));
        }
    }
    out
}

pub fn compute_safe_pivot_experiment_for_code(
    code: CodeMeasurement,
) -> Result<SafePivotExperimentReport, String> {
    let rows_by_pivot = native_rows_for_code_all_pivots(code);
    compute_safe_pivot_experiment_from_native_rows_by_pivot(code, &rows_by_pivot)
}

pub fn compute_safe_pivot_experiment_from_native_rows_by_pivot(
    code: CodeMeasurement,
    rows_by_pivot: &[Vec<NativeRow>],
) -> Result<SafePivotExperimentReport, String> {
    if rows_by_pivot.len() != TOTAL_QUBITS {
        return Err(format!(
            "expected native rows for {TOTAL_QUBITS} pivots, got {}",
            rows_by_pivot.len()
        ));
    }

    let fixed_rows = rows_by_pivot
        .get(FIXED_PIVOT_INDEX)
        .ok_or("missing fixed pivot rows")?;
    let fixed_costs = compute_ours_single_shot_tail_costs_from_native_rows(fixed_rows);
    let fixed_report = histogram_from_beta_costs(&fixed_costs);

    let certified = certify_safe_pivots(code, rows_by_pivot);
    let accepted_nontrivial = certified
        .report
        .accepted_qubits
        .iter()
        .any(|qubit| *qubit != 1);
    let mut j_safe = if accepted_nontrivial {
        let mut out = certified.report.accepted_qubits.clone();
        if !out.contains(&1) {
            out.push(1);
        }
        out.sort_unstable();
        out.dedup();
        out
    } else {
        vec![1]
    };
    j_safe.sort_unstable();

    let mut per_pivot_costs: BTreeMap<usize, Vec<Option<u32>>> = BTreeMap::new();
    for &pivot_qubit in &j_safe {
        let pivot_index = usize::from(pivot_qubit - 1);
        if pivot_index == FIXED_PIVOT_INDEX {
            continue;
        }
        let rows = rows_by_pivot.get(pivot_index).ok_or_else(|| {
            format!("missing native rows for certified pivot qubit {pivot_qubit}")
        })?;
        per_pivot_costs.insert(
            pivot_index,
            compute_ours_single_shot_tail_costs_from_native_rows(rows),
        );
    }

    let mut safe_costs = fixed_costs.clone();
    for &pivot_qubit in &j_safe {
        let pivot_index = usize::from(pivot_qubit - 1);
        if pivot_index == FIXED_PIVOT_INDEX {
            continue;
        }
        let action = certified
            .canonical_actions
            .get(&pivot_index)
            .ok_or_else(|| {
                format!(
                    "internal error: no canonical admissible transformation for certified pivot {}",
                    pivot_qubit
                )
            })?;
        let pivot_costs = per_pivot_costs
            .get(&pivot_index)
            .ok_or("internal error: missing cached pivot costs")?;

        for tail_bits in 0..TAIL_SPACE_SIZE {
            let tail = Tail11::new(tail_bits);
            let mapped_tail = transport_tail_from_fixed_pivot(*action, pivot_index, tail)
                .ok_or_else(|| {
                    format!(
                        "certified transformation to pivot {} failed to map fixed tail {}",
                        pivot_qubit, tail
                    )
                })?;
            let mapped_cost = pivot_costs[mapped_tail.0 as usize];
            let slot = &mut safe_costs[tail_bits as usize];
            *slot = option_min_cost(*slot, mapped_cost);
        }
    }

    let safe_report = histogram_from_beta_costs(&safe_costs);
    let violations = monotonicity_violations(&fixed_costs, &safe_costs);
    if !violations.is_empty() {
        let sample: Vec<String> = violations
            .iter()
            .take(8)
            .map(|tail| tail.to_label())
            .collect();
        return Err(format!(
            "safe-pivot monotonicity violated for {} tails; sample={sample:?}",
            violations.len()
        ));
    }

    let summary = SafePivotSummaryReport {
        j_safe: j_safe.clone(),
        mean_fixed_pivot_cost: fixed_report.mean,
        mean_safe_pivot_cost: safe_report.mean,
        total_count: fixed_report.total_targets,
        fixed_pivot_support: fixed_report.support.clone(),
        safe_pivot_support: safe_report.support.clone(),
    };

    Ok(SafePivotExperimentReport {
        fixed_pivot: fixed_report,
        safe_pivot: safe_report,
        certification: certified.report,
        summary,
    })
}

fn validate_pivot_candidate(
    native_rows: &[NativeRow],
    model: &SingleShotModel,
    report: &ExactHistogramReport,
) -> Vec<String> {
    let mut reasons = Vec::new();

    if report.reachable_targets != report.total_targets {
        reasons.push(format!(
            "coverage incomplete: reachable={} total={}",
            report.reachable_targets, report.total_targets
        ));
    }

    for prev_head in [Head::I, Head::X, Head::Y, Head::Z] {
        for q_basis in NON_I_HEADS {
            let (_, delta_local) = transition_local(prev_head, q_basis);
            if delta_local == 5 {
                reasons.push(format!(
                    "local delta hygiene failure: transition ({prev_head}, {q_basis}) produced 5"
                ));
            }
            if delta_local >= 7 {
                reasons.push(format!(
                    "local delta hygiene failure: transition ({prev_head}, {q_basis}) produced >= 7"
                ));
            }
        }
    }

    let native_sig = signature(native_rows);
    for (state_idx, source) in &model.source_states {
        match source.witness {
            SingleShotSourceWitness::P0I { tail, .. } => {
                if !native_sig.contains(&(Head::I, tail)) {
                    reasons.push(format!(
                        "P0_I template mismatch at tail {}: source not present in native rows",
                        tail
                    ));
                }
            }
            SingleShotSourceWitness::P0Q { q_basis, tail, .. } => {
                if !native_sig.contains(&(q_basis, tail)) {
                    reasons.push(format!(
                        "P0_Q template mismatch at (head={}, tail={}): source not present in native rows",
                        q_basis, tail
                    ));
                }
            }
            SingleShotSourceWitness::P2 {
                q_basis,
                left_tail,
                right_tail,
                ..
            } => {
                if !matches!(q_basis, Head::X | Head::Y | Head::Z) {
                    reasons.push("P2 same-basis rule failure: q_basis must be non-I".to_string());
                }
                if !left_tail.commutes_with(right_tail) {
                    reasons.push(format!(
                        "P2 commutation failure for pair ({}, {})",
                        left_tail, right_tail
                    ));
                }
                if !native_sig.contains(&(q_basis, left_tail))
                    || !native_sig.contains(&(q_basis, right_tail))
                {
                    reasons.push(format!(
                        "P2 template failure: pair ({}, {}) is not in same-basis native bucket {}",
                        left_tail, right_tail, q_basis
                    ));
                }
                let expected_state_idx = state_index(Head::I, (left_tail ^ right_tail).0);
                if *state_idx != expected_state_idx {
                    reasons.push(format!(
                        "P2 output state mismatch: expected state {} but got {}",
                        expected_state_idx, state_idx
                    ));
                }
            }
        }
    }

    reasons
}

pub fn compute_pivot_scan_experiment_for_code(
    code: CodeMeasurement,
    candidate_pivots: &[u8],
) -> Result<PivotScanExperimentReport, String> {
    let rows_by_pivot = native_rows_for_code_all_pivots(code);
    compute_pivot_scan_experiment_from_native_rows_by_pivot(code, &rows_by_pivot, candidate_pivots)
}

pub fn compute_pivot_scan_experiment_from_native_rows_by_pivot(
    _code: CodeMeasurement,
    rows_by_pivot: &[Vec<NativeRow>],
    candidate_pivots: &[u8],
) -> Result<PivotScanExperimentReport, String> {
    if rows_by_pivot.len() != TOTAL_QUBITS {
        return Err(format!(
            "expected native rows for {TOTAL_QUBITS} pivots, got {}",
            rows_by_pivot.len()
        ));
    }

    let mut j_emp: Vec<u8> = if candidate_pivots.is_empty() {
        (1..=TOTAL_QUBITS as u8).collect()
    } else {
        candidate_pivots.to_vec()
    };
    if !j_emp.contains(&1) {
        j_emp.push(1);
    }
    j_emp.sort_unstable();
    j_emp.dedup();

    let mut j_valid = Vec::new();
    let mut pivot_summaries = Vec::new();
    let mut per_pivot_hist = BTreeMap::new();
    let mut per_pivot_costs: BTreeMap<u8, Vec<Option<u32>>> = BTreeMap::new();

    for &pivot_qubit in &j_emp {
        if pivot_qubit == 0 || usize::from(pivot_qubit) > TOTAL_QUBITS {
            pivot_summaries.push(PivotScanPivotSummary {
                pivot_index: pivot_qubit,
                validation_passed: false,
                rejection_reasons: vec!["pivot index out of supported range [1, 12]".to_string()],
                total_targets: TAIL_SPACE_SIZE as u64,
                reachable_targets: 0,
                mean: None,
                support: Vec::new(),
            });
            continue;
        }

        let pivot_index = usize::from(pivot_qubit - 1);
        let native_rows = &rows_by_pivot[pivot_index];
        let model = SingleShotModel::from_native_rows(native_rows);
        let dist = solve_all_state_costs(&model);
        let costs = beta_costs_from_distances(&dist);
        let hist = histogram_from_beta_costs(&costs);
        let reasons = validate_pivot_candidate(native_rows, &model, &hist);
        let validation_passed = reasons.is_empty();
        if validation_passed {
            j_valid.push(pivot_qubit);
            per_pivot_hist.insert(pivot_qubit, hist.clone());
            per_pivot_costs.insert(pivot_qubit, costs);
        }

        pivot_summaries.push(PivotScanPivotSummary {
            pivot_index: pivot_qubit,
            validation_passed,
            rejection_reasons: reasons,
            total_targets: hist.total_targets,
            reachable_targets: hist.reachable_targets,
            mean: hist.mean,
            support: hist.support.clone(),
        });
    }

    if !j_valid.contains(&1) {
        return Err("pivot 1 failed validation; fixed-pivot baseline is unavailable".to_string());
    }

    let fixed_pivot_exact_hist = per_pivot_hist
        .get(&1)
        .cloned()
        .ok_or("internal error: fixed-pivot histogram missing")?;
    let fixed_costs = per_pivot_costs
        .get(&1)
        .cloned()
        .ok_or("internal error: fixed-pivot costs missing")?;

    let mut best_single_costs = vec![None; TAIL_SPACE_SIZE as usize];
    for valid_pivot in &j_valid {
        let costs = per_pivot_costs
            .get(valid_pivot)
            .ok_or("internal error: missing valid pivot costs")?;
        for (idx, cost) in costs.iter().copied().enumerate() {
            best_single_costs[idx] = option_min_cost(best_single_costs[idx], cost);
        }
    }

    let violations = monotonicity_violations(&fixed_costs, &best_single_costs);
    if !violations.is_empty() {
        let sample: Vec<String> = violations
            .iter()
            .take(8)
            .map(|tail| tail.to_label())
            .collect();
        return Err(format!(
            "best-single-pivot monotonicity violated for {} tails; sample={sample:?}",
            violations.len()
        ));
    }

    let best_single_pivot_exact_hist = histogram_from_beta_costs(&best_single_costs);
    let best_pivot_by_mean = pivot_summaries
        .iter()
        .filter(|entry| entry.validation_passed)
        .filter_map(|entry| entry.mean.map(|mean| (entry.pivot_index, mean)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|pair| pair.0);

    let best_single_pivot_summary = BestSinglePivotSummaryReport {
        j_valid: j_valid.clone(),
        best_pivot_by_mean,
        fixed_pivot_mean: fixed_pivot_exact_hist.mean,
        best_single_pivot_mean: best_single_pivot_exact_hist.mean,
        improvement_over_fixed: match (
            fixed_pivot_exact_hist.mean,
            best_single_pivot_exact_hist.mean,
        ) {
            (Some(fixed), Some(best)) => Some(fixed - best),
            _ => None,
        },
    };

    Ok(PivotScanExperimentReport {
        per_pivot_exact_hist: per_pivot_hist,
        pivot_scan_summary: PivotScanSummaryReport {
            j_emp,
            j_valid,
            pivots: pivot_summaries,
        },
        fixed_pivot_exact_hist,
        best_single_pivot_exact_hist,
        best_single_pivot_summary,
    })
}

pub fn explain_ours_single_shot_target_for_code(
    code: CodeMeasurement,
    target_tail: Tail11,
) -> Option<OursSingleShotTargetExplain> {
    let native_rows = native_rows_for_code(code);
    explain_ours_single_shot_target_from_native_rows(&native_rows, target_tail)
}

pub fn explain_ours_single_shot_target_from_native_rows(
    native_rows: &[NativeRow],
    target_tail: Tail11,
) -> Option<OursSingleShotTargetExplain> {
    let model = SingleShotModel::from_native_rows(native_rows);
    explain_target_with_model(&model, target_tail)
}

pub fn compute_paper_baseline_exact_hist_for_code(code: CodeMeasurement) -> ExactHistogramReport {
    let mut builder = MeasurementTableBuilder::new(NativeMeasurement::all(), code);
    builder.build();
    let complete = builder
        .complete()
        .expect("Paper baseline table build should succeed");
    compute_paper_baseline_exact_hist_with_table(&complete)
}

pub fn compute_paper_baseline_exact_hist_with_table(
    complete: &CompleteMeasurementTable,
) -> ExactHistogramReport {
    let mut histogram: BTreeMap<u32, u64> = BTreeMap::new();

    for tail_bits in 0..TAIL_SPACE_SIZE {
        let p = PauliString::rotation(tail_bits);
        let rotations = complete.min_data(p).rotations().len() as u32;
        let total_cost = paper_total_cost_from_rotation_count(rotations);
        *histogram.entry(total_cost).or_insert(0) += 1;
    }

    let reachable_targets = TAIL_SPACE_SIZE as u64;
    let total_targets = reachable_targets;
    let support: Vec<u32> = histogram.keys().copied().collect();
    let weighted_sum: u128 = histogram
        .iter()
        .map(|(cost, count)| *cost as u128 * *count as u128)
        .sum();
    let mean = Some(weighted_sum as f64 / reachable_targets as f64);
    let rank = (reachable_targets + 1) / 2;
    let mut running = 0u64;
    let mut median = None;
    for (cost, count) in &histogram {
        running += *count;
        if running >= rank {
            median = Some(*cost);
            break;
        }
    }

    ExactHistogramReport {
        histogram,
        total_targets,
        reachable_targets,
        unreachable_targets: 0,
        support,
        median,
        mean,
        sample_unreachable_tails: Vec::new(),
    }
}

pub fn compute_ours_single_shot_exact_hist_for_choice(
    choice: crate::measurement::MeasurementChoices,
) -> ExactHistogramReport {
    compute_ours_single_shot_exact_hist_for_code(choice.measurement())
}

pub fn compute_paper_baseline_exact_hist_for_choice(
    choice: crate::measurement::MeasurementChoices,
) -> ExactHistogramReport {
    compute_paper_baseline_exact_hist_for_code(choice.measurement())
}

pub fn compute_safe_pivot_experiment_for_choice(
    choice: crate::measurement::MeasurementChoices,
) -> Result<SafePivotExperimentReport, String> {
    compute_safe_pivot_experiment_for_code(choice.measurement())
}

pub fn compute_pivot_scan_experiment_for_choice(
    choice: crate::measurement::MeasurementChoices,
    candidate_pivots: &[u8],
) -> Result<PivotScanExperimentReport, String> {
    compute_pivot_scan_experiment_for_code(choice.measurement(), candidate_pivots)
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use bicycle_common::Pauli;

    use super::*;
    use crate::draft_core::build_draft_library;

    fn row(index: usize, head: Head, tail: Tail11) -> NativeRow {
        NativeRow {
            index,
            head,
            tail,
            metadata: None,
        }
    }

    fn state_cost_for(
        native_rows: &[NativeRow],
        target_head: Head,
        target_tail: Tail11,
    ) -> Option<u32> {
        let model = SingleShotModel::from_native_rows(native_rows);
        let dist = solve_all_state_costs(&model);
        let idx = state_index(target_head, target_tail.0);
        (dist[idx] < INF_COST).then_some(dist[idx])
    }

    fn beta_cost_for(native_rows: &[NativeRow], target_tail: Tail11) -> Option<u32> {
        let model = SingleShotModel::from_native_rows(native_rows);
        let dist = solve_all_state_costs(&model);
        beta_for_tail(&dist, target_tail)
    }

    fn identity_rows() -> [u8; BLOCK_QUBITS] {
        [1, 2, 4, 8, 16, 32]
    }

    fn toy_rows_by_pivot() -> Vec<Vec<NativeRow>> {
        let t0 = Tail11::from_label("XIIIIIIIIII").unwrap();
        let t1 = Tail11::from_label("IXIIIIIIIII").unwrap();
        let fixed_rows = vec![
            row(0, Head::I, t0),
            row(1, Head::X, t0),
            row(2, Head::X, t1),
        ];
        (0..TOTAL_QUBITS).map(|_| fixed_rows.clone()).collect()
    }

    #[test]
    fn direct_native_q_gives_unit_cost_state() {
        let target = Tail11::from_label("XIIIIIIIIII").unwrap();
        let rows = vec![row(0, Head::X, target)];
        let cost = state_cost_for(&rows, Head::X, target).unwrap();
        assert_eq!(cost, 1);
    }

    #[test]
    fn p2_adds_direct_cost4_template() {
        let left = Tail11::from_label("XIIIIIIIIII").unwrap();
        let right = Tail11::from_label("IXIIIIIIIII").unwrap();
        let target = left ^ right;
        let rows = vec![row(0, Head::X, left), row(1, Head::X, right)];
        let cost_i = state_cost_for(&rows, Head::I, target).unwrap();
        assert_eq!(cost_i, 4);
    }

    #[test]
    fn p3_local_transition_has_no_five() {
        for prev in [Head::I, Head::X, Head::Y, Head::Z] {
            for q in [Head::X, Head::Y, Head::Z] {
                let (_, delta) = transition_local(prev, q);
                assert_ne!(delta, 5);
                assert!(delta < 7);
            }
        }

        let (next_x, delta_x) = transition_local(Head::I, Head::X);
        let (next_y, delta_y) = transition_local(Head::I, Head::Y);
        let (next_z, delta_z) = transition_local(Head::I, Head::Z);

        assert_eq!(next_x, Head::X);
        assert_eq!(next_y, Head::Y);
        assert_eq!(next_z, Head::Z);
        assert_eq!(delta_x, 4);
        assert_eq!(delta_y, 4);
        assert_eq!(delta_z, 4);
    }

    #[test]
    fn no_cross_target_sharing_leakage_from_library_semantics() {
        let base = Tail11::from_label("XIIIIIIIIII").unwrap();
        let axis = Tail11::from_label("ZIIIIIIIIII").unwrap();
        let target = base ^ axis;
        let rows = vec![row(0, Head::I, base), row(1, Head::X, axis)];

        let library = build_draft_library(&rows);
        assert!(library.library.contains_key(&target));

        // In single-shot mode there is no direct-use shortcut for target from previously
        // synthesized tails; the best route must pay its own synthesis path.
        let beta = beta_cost_for(&rows, target).unwrap();
        assert_eq!(beta, 5);
    }

    #[test]
    fn beta_projection_is_min_over_xyz_heads() {
        let base = Tail11::from_label("XIIIIIIIIII").unwrap();
        let axis = Tail11::from_label("ZIIIIIIIIII").unwrap();
        let target = base ^ axis;
        let rows = vec![
            row(0, Head::I, base),
            row(1, Head::X, axis),
            row(2, Head::Y, target),
        ];

        // d(X,target) can be obtained through P3 from I,base and axis, with +4 delta.
        let d_x = state_cost_for(&rows, Head::X, target).unwrap();
        assert_eq!(d_x, 5);

        // direct native Y target gives d(Y,target)=1, so beta must choose 1.
        let beta = beta_cost_for(&rows, target).unwrap();
        assert_eq!(beta, 1);
    }

    #[test]
    fn histogram_sanity_on_toy_subset() {
        let t0 = Tail11::from_label("XIIIIIIIIII").unwrap();
        let t1 = Tail11::from_label("YIIIIIIIIII").unwrap();
        let rows = vec![row(0, Head::X, t0), row(1, Head::Y, t1)];

        let model = SingleShotModel::from_native_rows(&rows);
        let dist = solve_all_state_costs(&model);

        let mut hist = BTreeMap::new();
        for tail in [t0, t1] {
            let beta = beta_for_tail(&dist, tail).unwrap();
            *hist.entry(beta).or_insert(0) += 1;
        }

        assert_eq!(hist.values().sum::<u64>(), 2);
        assert_eq!(hist.get(&1), Some(&2));
    }

    #[test]
    fn paper_cost_scaling_is_one_plus_six_k() {
        assert_eq!(paper_total_cost_from_rotation_count(0), 1);
        assert_eq!(paper_total_cost_from_rotation_count(1), 7);
        assert_eq!(paper_total_cost_from_rotation_count(2), 13);
        assert_eq!(paper_total_cost_from_rotation_count(3), 19);
    }

    #[test]
    fn infer_measurement_choice_from_rows_matches_known_codes() {
        let gross_rows = native_rows_for_code(GROSS_MEASUREMENT);
        let two_gross_rows = native_rows_for_code(TWOGROSS_MEASUREMENT);

        assert!(matches!(
            infer_measurement_choice_from_native_rows(&gross_rows),
            Some(MeasurementChoices::Gross)
        ));
        assert!(matches!(
            infer_measurement_choice_from_native_rows(&two_gross_rows),
            Some(MeasurementChoices::TwoGross)
        ));
    }

    #[test]
    fn certification_rejects_non_single_qubit_pivot_image() {
        let rows_by_pivot = native_rows_for_code_all_pivots(GROSS_MEASUREMENT);
        let action = SymmetryAction::from_shift(GROSS_MEASUREMENT, 0, 1);
        let checked = evaluate_shift_candidate(action, &rows_by_pivot);
        assert!(!checked.report.accepted);
        assert!(
            checked
                .report
                .failure_reasons
                .iter()
                .any(|reason| reason.contains("criterion 3.1")),
            "expected criterion 3.1 failure, got {:?}",
            checked.report.failure_reasons
        );
    }

    #[test]
    fn certification_rejects_p2_same_basis_breakage() {
        let t_x3 = Tail11::from_label("IIXIIIIIIII").unwrap();
        let t_z2 = Tail11::from_label("IZIIIIIIIII").unwrap();
        let t_z3 = Tail11::from_label("IIZIIIIIIII").unwrap();
        let fixed_rows = vec![
            row(0, Head::X, t_x3),
            row(1, Head::X, t_z2),
            row(2, Head::X, t_z3),
        ];

        let mut rows_by_pivot: Vec<Vec<NativeRow>> =
            (0..TOTAL_QUBITS).map(|_| Vec::new()).collect();
        rows_by_pivot[FIXED_PIVOT_INDEX] = fixed_rows;

        let x_rows = identity_rows();
        let z_rows = [0b000001, 0b000010, 0, 0b001100, 0b010000, 0b100000];
        let action = SymmetryAction {
            shift_x: 0,
            shift_y: 0,
            x_rows,
            z_rows,
        };

        let checked = evaluate_shift_candidate(action, &rows_by_pivot);
        assert!(!checked.report.accepted);
        assert!(checked.report.pivot_image_preserved);
        assert!(checked.report.lpu_pair_preserved);
        assert!(checked.report.p0_grammar_preserved);
        assert!(!checked.report.p2_grammar_preserved);
        assert!(
            checked
                .report
                .failure_reasons
                .iter()
                .any(|reason| reason.contains("same-basis P2")),
            "expected P2 rejection reason, got {:?}",
            checked.report.failure_reasons
        );
    }

    #[test]
    fn certification_rejects_changed_p3_weighted_transition_rows() {
        let err = certify_p3_transitions_with_mapper(|head| {
            Some(match head {
                Head::Z => Head::X,
                _ => head,
            })
        })
        .unwrap_err();
        assert!(err.contains("criterion 3.4"));
    }

    #[test]
    fn safe_pivot_monotonicity_holds_on_toy_cases() {
        let rows_by_pivot = toy_rows_by_pivot();
        for code in [GROSS_MEASUREMENT, TWOGROSS_MEASUREMENT] {
            let report =
                compute_safe_pivot_experiment_from_native_rows_by_pivot(code, &rows_by_pivot)
                    .unwrap();
            if let (Some(fixed), Some(safe)) = (
                report.summary.mean_fixed_pivot_cost,
                report.summary.mean_safe_pivot_cost,
            ) {
                assert!(safe <= fixed);
            }
        }
    }

    #[test]
    fn safe_pivot_trivial_fallback_matches_fixed() {
        let rows_by_pivot = toy_rows_by_pivot();
        let report = compute_safe_pivot_experiment_from_native_rows_by_pivot(
            GROSS_MEASUREMENT,
            &rows_by_pivot,
        )
        .unwrap();
        assert_eq!(report.summary.j_safe, vec![1]);
        assert_eq!(report.safe_pivot.histogram, report.fixed_pivot.histogram);
    }

    #[test]
    fn safe_pivot_report_integrity_includes_rejection_explanations() {
        let rows_by_pivot = toy_rows_by_pivot();
        let report = compute_safe_pivot_experiment_from_native_rows_by_pivot(
            GROSS_MEASUREMENT,
            &rows_by_pivot,
        )
        .unwrap();
        let cert_json = serde_json::to_value(&report.certification).unwrap();

        assert!(cert_json.get("candidate_qubits").is_some());
        assert!(cert_json.get("accepted_qubits").is_some());
        assert!(cert_json.get("rejected_qubits").is_some());
        assert!(cert_json.get("rejection_reasons").is_some());

        for rejected in &report.certification.rejected_qubits {
            let reasons = report
                .certification
                .rejection_reasons
                .get(rejected)
                .unwrap();
            assert!(!reasons.is_empty());
        }
    }

    #[test]
    fn pivot_rebuild_isolation_parsing_changes_with_pivot_choice() {
        let rows_pivot_1 = native_rows_for_code_with_pivot(GROSS_MEASUREMENT, 0);
        let rows_pivot_2 = native_rows_for_code_with_pivot(GROSS_MEASUREMENT, 1);
        assert_ne!(signature(&rows_pivot_1), signature(&rows_pivot_2));

        let model_pivot_1 = SingleShotModel::from_native_rows(&rows_pivot_1);
        let model_pivot_2 = SingleShotModel::from_native_rows(&rows_pivot_2);
        let src_1: BTreeSet<_> = model_pivot_1.source_states.keys().copied().collect();
        let src_2: BTreeSet<_> = model_pivot_2.source_states.keys().copied().collect();
        assert_ne!(src_1, src_2);
    }

    #[test]
    fn pivot_scan_rejects_invalid_candidate_and_keeps_pivot1_valid() {
        let mut rows_by_pivot: Vec<Vec<NativeRow>> =
            (0..TOTAL_QUBITS).map(|_| Vec::new()).collect();
        rows_by_pivot[0] = native_rows_for_code_with_pivot(GROSS_MEASUREMENT, 0);
        rows_by_pivot[1] = Vec::new();

        let report = compute_pivot_scan_experiment_from_native_rows_by_pivot(
            GROSS_MEASUREMENT,
            &rows_by_pivot,
            &[1, 2],
        )
        .unwrap();
        assert!(report.pivot_scan_summary.j_valid.contains(&1));
        let pivot_2 = report
            .pivot_scan_summary
            .pivots
            .iter()
            .find(|entry| entry.pivot_index == 2)
            .unwrap();
        assert!(!pivot_2.validation_passed);
        assert!(!pivot_2.rejection_reasons.is_empty());
    }

    #[test]
    fn pivot_j_single_shot_still_forbids_global_sharing() {
        let mut full_base = [Pauli::I; TOTAL_QUBITS];
        full_base[2] = Pauli::X;
        let mut full_axis = [Pauli::I; TOTAL_QUBITS];
        full_axis[1] = Pauli::X;
        full_axis[2] = Pauli::Z;

        let base_row = native_row_from_full_paulis(0, full_base, 1);
        let axis_row = native_row_from_full_paulis(1, full_axis, 1);
        let rows = vec![base_row, axis_row];
        let target = base_row.tail ^ axis_row.tail;

        let beta = beta_cost_for(&rows, target).unwrap();
        assert_eq!(beta, 5);
    }

    #[test]
    fn best_single_pivot_monotonicity_on_toy_targets() {
        let fixed = vec![Some(9), Some(7), Some(11), Some(5)];
        let pivot_2 = vec![Some(8), Some(8), Some(10), Some(7)];
        let pivot_3 = vec![Some(9), Some(6), Some(12), Some(5)];
        let mut best = vec![None; fixed.len()];
        for idx in 0..fixed.len() {
            best[idx] = option_min_cost(option_min_cost(fixed[idx], pivot_2[idx]), pivot_3[idx]);
        }
        assert!(monotonicity_violations(&fixed, &best).is_empty());
    }

    #[test]
    fn local_delta_hygiene_holds_for_all_valid_pivots_in_scan() {
        let mut rows_by_pivot: Vec<Vec<NativeRow>> =
            (0..TOTAL_QUBITS).map(|_| Vec::new()).collect();
        rows_by_pivot[0] = native_rows_for_code_with_pivot(GROSS_MEASUREMENT, 0);
        let report = compute_pivot_scan_experiment_from_native_rows_by_pivot(
            GROSS_MEASUREMENT,
            &rows_by_pivot,
            &[1],
        )
        .unwrap();

        for pivot in &report.pivot_scan_summary.pivots {
            if !pivot.validation_passed {
                continue;
            }
            for prev_head in [Head::I, Head::X, Head::Y, Head::Z] {
                for q_basis in NON_I_HEADS {
                    let (_, delta_local) = transition_local(prev_head, q_basis);
                    assert_ne!(
                        delta_local, 5,
                        "pivot {} emitted local delta 5",
                        pivot.pivot_index
                    );
                    assert!(
                        delta_local < 7,
                        "pivot {} emitted local delta >= 7",
                        pivot.pivot_index
                    );
                }
            }
        }
    }

    #[test]
    fn parse_native_rows_from_csv_six_column_layout() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("draft_single_shot_parse_{unique}.csv"));
        let csv = "0,0,I,X,XIIIII,IIIIII\n0,0,I,Y,YIIIII,IIIIII\n";
        std::fs::write(&path, csv).unwrap();

        let parsed = parse_native_rows_from_csv(&path, 0).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].head, Head::X);
        assert_eq!(parsed[0].tail, Tail11::from_label("IIIIIIIIIII").unwrap());
        assert_eq!(parsed[1].head, Head::Y);
        assert_eq!(parsed[1].tail, Tail11::from_label("IIIIIIIIIII").unwrap());
    }
}
