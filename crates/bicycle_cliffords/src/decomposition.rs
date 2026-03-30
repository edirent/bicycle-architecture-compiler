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

use std::collections::{BTreeMap, HashMap};

use crate::measurement::CodeMeasurement;
use crate::pauli_string::PauliString;
use crate::{native_measurement::NativeMeasurement, pauli_string};

use bicycle_common::{AutomorphismData, BicycleISA, TwoBases};
use log::{debug, error, info, trace, warn};
use serde::{Deserialize, Serialize};

// Defines a rotation that is implemented by a rotation conjugated with a base rotation.
// Need appropriate measurements conjugating the rotation on the pivot.
// Assume that conjugated_with anti-commutes with rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct MeasurementTableEntry {
    measurement: PauliString,
    conjugated_with: Option<PauliString>,
    cost: u32,
}

impl MeasurementTableEntry {
    pub fn cost(&self) -> u32 {
        self.cost
    }

    pub fn implements(&self) -> PauliString {
        if let Some(conj) = self.conjugated_with {
            // Whenever we conjugate by a rotation, the pivot gets reset.
            self.measurement.conjugate_with(conj.zero_pivot())
        } else {
            self.measurement
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize, Default)]
pub struct ResourceSet {
    pub pivot: bool,
    pub bridge: bool,
    pub bridge_checks: bool,
    pub half_lpu_left: bool,
    pub half_lpu_right: bool,
    pub code_factory: bool,
}

impl ResourceSet {
    fn union(self, rhs: Self) -> Self {
        Self {
            pivot: self.pivot || rhs.pivot,
            bridge: self.bridge || rhs.bridge,
            bridge_checks: self.bridge_checks || rhs.bridge_checks,
            half_lpu_left: self.half_lpu_left || rhs.half_lpu_left,
            half_lpu_right: self.half_lpu_right || rhs.half_lpu_right,
            code_factory: self.code_factory || rhs.code_factory,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum NativePrimitiveKind {
    InModule,
    CodeCode,
    CodeFactory,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativePrimitive {
    kind: NativePrimitiveKind,
    observable: PauliString,
    instructions: Vec<BicycleISA>,
    #[serde(default)]
    native_measurement: Option<NativeMeasurement>,
    resources: ResourceSet,
    cost: u32,
}

impl NativePrimitive {
    pub fn new(
        kind: NativePrimitiveKind,
        observable: PauliString,
        instructions: Vec<BicycleISA>,
        native_measurement: Option<NativeMeasurement>,
        resources: ResourceSet,
        cost: u32,
    ) -> Self {
        Self {
            kind,
            observable,
            instructions,
            native_measurement,
            resources,
            cost,
        }
    }

    pub fn from_native_in_module(observable: PauliString, native: NativeMeasurement) -> Self {
        Self::new(
            NativePrimitiveKind::InModule,
            observable,
            native.implementation().to_vec(),
            Some(native),
            ResourceSet {
                pivot: true,
                ..ResourceSet::default()
            },
            1,
        )
    }

    pub fn kind(&self) -> NativePrimitiveKind {
        self.kind
    }

    pub fn observable(&self) -> PauliString {
        self.observable
    }

    pub fn instructions(&self) -> &[BicycleISA] {
        &self.instructions
    }

    pub fn native_measurement(&self) -> Option<NativeMeasurement> {
        self.native_measurement
    }

    pub fn resources(&self) -> ResourceSet {
        self.resources
    }

    pub fn cost(&self) -> u32 {
        self.cost
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize, Default)]
pub struct FrameState {
    pub sign_parity: u8,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize, Default)]
pub enum OutcomeReducer {
    #[default]
    Identity,
    XorFirstTwo,
    XorAll,
}

impl OutcomeReducer {
    fn bit(raw_bits: &[u8], i: usize) -> u8 {
        raw_bits.get(i).copied().unwrap_or(0) & 1
    }

    pub fn reduce(&self, raw_bits: &[u8], frame: FrameState) -> u8 {
        let frame_parity = frame.sign_parity & 1;
        match self {
            OutcomeReducer::Identity => Self::bit(raw_bits, 0) ^ frame_parity,
            OutcomeReducer::XorFirstTwo => {
                Self::bit(raw_bits, 0) ^ Self::bit(raw_bits, 1) ^ frame_parity
            }
            OutcomeReducer::XorAll => raw_bits
                .iter()
                .fold(frame_parity, |acc, bit| acc ^ (bit & 1)),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum ByproductEffect {
    None,
    GaugeOnly,
    ApplyPauliIfBitSet { bit_index: u8, pauli: PauliString },
}

impl ByproductEffect {
    pub fn effect(&self, raw_bits: &[u8]) -> ByproductEffect {
        match self {
            ByproductEffect::None => ByproductEffect::None,
            ByproductEffect::GaugeOnly => ByproductEffect::GaugeOnly,
            ByproductEffect::ApplyPauliIfBitSet { bit_index, pauli } => {
                let bit_is_set = raw_bits.get(*bit_index as usize).copied().unwrap_or(0) & 1 == 1;
                if bit_is_set {
                    ByproductEffect::ApplyPauliIfBitSet {
                        bit_index: *bit_index,
                        pauli: *pauli,
                    }
                } else {
                    ByproductEffect::None
                }
            }
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct MeasurementWitness {
    target: PauliString,
    primitives: Vec<NativePrimitive>,
    reducer: OutcomeReducer,
    byproduct: ByproductEffect,
    resources: ResourceSet,
    cost: u32,
    #[serde(default)]
    metadata: BTreeMap<String, String>,
}

impl MeasurementWitness {
    pub fn new(
        target: PauliString,
        primitives: Vec<NativePrimitive>,
        reducer: OutcomeReducer,
        byproduct: ByproductEffect,
        metadata: BTreeMap<String, String>,
    ) -> Self {
        let (resources, cost) = primitives.iter().fold(
            (ResourceSet::default(), 0_u32),
            |(resources, cost), primitive| {
                (
                    resources.union(primitive.resources()),
                    cost.saturating_add(primitive.cost()),
                )
            },
        );
        Self {
            target,
            primitives,
            reducer,
            byproduct,
            resources,
            cost,
            metadata,
        }
    }

    pub fn singleton_native(target: PauliString, native: NativeMeasurement) -> Self {
        Self::new(
            target,
            vec![NativePrimitive::from_native_in_module(target, native)],
            OutcomeReducer::Identity,
            ByproductEffect::None,
            BTreeMap::new(),
        )
    }

    pub fn target(&self) -> PauliString {
        self.target
    }

    pub fn primitives(&self) -> &[NativePrimitive] {
        &self.primitives
    }

    pub fn reducer(&self) -> OutcomeReducer {
        self.reducer
    }

    pub fn byproduct(&self) -> &ByproductEffect {
        &self.byproduct
    }

    pub fn resources(&self) -> ResourceSet {
        self.resources
    }

    pub fn cost(&self) -> u32 {
        self.cost
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn evaluate(&self, raw_bits: &[u8], frame: FrameState) -> (u8, ByproductEffect) {
        let logical = self.reducer.reduce(raw_bits, frame);
        let effect = self.byproduct.effect(raw_bits);
        (logical, effect)
    }

    pub fn emit_primitive_instructions(&self) -> impl Iterator<Item = BicycleISA> + '_ {
        self.primitives
            .iter()
            .flat_map(|primitive| primitive.instructions().iter().copied())
    }

    pub fn single_native_impl(&self) -> Option<NativeMeasurementImpl> {
        if self.primitives.len() != 1 {
            return None;
        }
        let primitive = &self.primitives[0];
        primitive
            .native_measurement()
            .map(|native| NativeMeasurementImpl::new(native, primitive.observable()))
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MeasurementImpl {
    base_witness: MeasurementWitness,
    base_native: Option<NativeMeasurementImpl>,
    rotations: Vec<NativeMeasurementImpl>,
    measures: PauliString,
}

impl MeasurementImpl {
    pub fn base_witness(&self) -> &MeasurementWitness {
        &self.base_witness
    }

    pub fn base_measurement(&self) -> &NativeMeasurementImpl {
        self.base_native
            .as_ref()
            .expect("Base witness is not a singleton native primitive")
    }

    pub fn rotations(&self) -> &Vec<NativeMeasurementImpl> {
        &self.rotations
    }

    pub fn measures(&self) -> PauliString {
        self.measures
    }

    pub fn cost(&self) -> u32 {
        self.base_witness
            .cost()
            .saturating_add(2_u32.saturating_mul(self.rotations.len() as u32))
    }
}

/// A wrapper for &NativeMeasurement that caches what it measures
/// Basically a nice wrapper for (PauliString, &NativeMeasurement)
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct NativeMeasurementImpl {
    native: NativeMeasurement,
    measures: PauliString,
}

impl NativeMeasurementImpl {
    pub fn new(native: NativeMeasurement, measures: PauliString) -> Self {
        Self { native, measures }
    }

    pub fn logical(&self) -> TwoBases {
        self.native.logical
    }

    pub fn automorphism(&self) -> AutomorphismData {
        self.native.automorphism
    }

    pub fn implementation(&self) -> [BicycleISA; 3] {
        self.native.implementation()
    }

    pub fn measures(&self) -> PauliString {
        self.measures
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteMeasurementTable {
    measurements: Vec<MeasurementTableEntry>,
    native_measurements: HashMap<PauliString, NativeMeasurement>,
    #[serde(default)]
    witnesses: HashMap<PauliString, MeasurementWitness>,
}

impl CompleteMeasurementTable {
    /// Look up the implementation for measuring a PauliString
    fn get(&self, p: PauliString) -> Option<&MeasurementTableEntry> {
        self.measurements.get(MeasurementTableBuilder::index(p))
    }

    pub fn lookup_witness(&self, p: PauliString) -> Option<MeasurementWitness> {
        self.witnesses.get(&p).cloned().or_else(|| {
            self.native_measurements
                .get(&p)
                .copied()
                .map(|native| MeasurementWitness::singleton_native(p, native))
        })
    }

    /// Returns the a native measurement and its conjugating native measurements that implement rotations
    /// The ordering of rotations is such that the first element conjugates the measurement first.
    /// The given PauliString must be a valid measurement defined on 12 qubits.
    pub fn implementation(&self, p: PauliString) -> MeasurementImpl {
        assert!(p.0 <= 4_u32.pow(12), "{}", p);
        assert!(p.0 != 0); // Cannot measure identity

        let mut implementation = self.get(p).unwrap();

        let mut rots = vec![];
        while let Some(conjugate) = implementation.conjugated_with {
            rots.push(conjugate);
            implementation = self.get(implementation.measurement).unwrap();
        }

        let base_witness = self
            .lookup_witness(implementation.measurement)
            .unwrap_or_else(|| {
                panic!(
                    "Missing witness for base measurement {} while implementing {}",
                    implementation.measurement, p
                )
            });
        let base_native = base_witness.single_native_impl();

        let native_rots = rots
            .into_iter()
            .map(|p| {
                self.native_measurements
                    .get(&p)
                    .map(|native| NativeMeasurementImpl::new(*native, p))
                    .unwrap()
            })
            .rev()
            .collect();
        MeasurementImpl {
            measures: p,
            base_witness,
            base_native,
            rotations: native_rots,
        }
    }

    /// Minimize over the Pauli on the pivot to measure 11 qubits in the basis p.
    /// This can be useful if you do not care about the basis of the pivot.
    /// TODO: If this becomes the only method needed, then we can shrink table by factor 4.
    pub fn min_data(&self, p: PauliString) -> MeasurementImpl {
        assert!(p.0 <= 4_u32.pow(12), "{}", p);
        assert!(
            p.pivot_bits() == pauli_string::ID,
            "Expected identity on pivot for {p}"
        );

        // Find minimum-length implementation out of three options for the pivot.

        [pauli_string::X1, pauli_string::Z1, pauli_string::Y1]
            .into_iter()
            .map(|pivot_pauli| p * pivot_pauli) // insert pivot basis
            .map(|q| self.implementation(q)) // look up implementation
            .min_by_key(|meas_impl| meas_impl.cost())
            .unwrap()
    }
}

impl TryFrom<MeasurementTableBuilder> for CompleteMeasurementTable {
    type Error = String;

    fn try_from(value: MeasurementTableBuilder) -> Result<Self, Self::Error> {
        let measurements: Option<Vec<_>> = value.measurements.into_iter().collect();
        Ok(CompleteMeasurementTable {
            measurements: measurements.ok_or("All measurements should have an implementation")?,
            native_measurements: value.native_measurements,
            witnesses: value.witnesses,
        })
    }
}

#[derive(Debug)]
pub struct MeasurementTableBuilder {
    measurements: Vec<Option<MeasurementTableEntry>>,
    native_measurements: HashMap<PauliString, NativeMeasurement>,
    witnesses: HashMap<PauliString, MeasurementWitness>,
    len: usize, // Count how many Some entries there are in measurements
    code: CodeMeasurement,
}

impl MeasurementTableBuilder {
    pub fn new(native_measurements: Vec<NativeMeasurement>, code: CodeMeasurement) -> Self {
        let len = 0;
        let measurements = vec![None; 4usize.pow(12)];

        let native_lookup: HashMap<PauliString, NativeMeasurement> = native_measurements
            .into_iter()
            .map(|meas| (code.measures(&meas), meas))
            .collect();
        let witness_lookup: HashMap<PauliString, MeasurementWitness> = native_lookup
            .iter()
            .map(|(observable, native)| {
                (
                    *observable,
                    MeasurementWitness::singleton_native(*observable, *native),
                )
            })
            .collect();

        let mut table = MeasurementTableBuilder {
            measurements,
            native_measurements: HashMap::new(), // Placeholder; set later.
            witnesses: HashMap::new(),           // Placeholder; set later.
            len,
            code,
        };

        for witness in witness_lookup.values() {
            table.insert(MeasurementTableEntry {
                measurement: witness.target(),
                conjugated_with: None,
                cost: witness.cost(), // TODO: Adjust me depending on noise simulations!
            });
        }
        table.native_measurements = native_lookup;
        table.witnesses = witness_lookup;

        // Insert identity
        let identity = MeasurementTableEntry {
            measurement: PauliString(0),
            conjugated_with: None,
            cost: 0,
        };
        table.insert(identity);

        table
    }

    fn validate_table_level_witness(&self, witness: &MeasurementWitness) -> Result<(), String> {
        if witness.primitives().len() != 1 {
            return Err(format!(
                "Witness for {} must be a singleton native terminal at table level",
                witness.target()
            ));
        }
        let primitive = &witness.primitives()[0];
        if primitive.kind() != NativePrimitiveKind::InModule {
            return Err(format!(
                "Witness for {} must use only InModule primitives at table level",
                witness.target()
            ));
        }
        let Some(native) = primitive.native_measurement() else {
            return Err(format!(
                "Witness for {} must wrap a true native measurement at table level",
                witness.target()
            ));
        };
        let true_target = self.code.measures(&native);
        if witness.target() != true_target {
            return Err(format!(
                "Witness target {} does not match code.measures(native) {}",
                witness.target(),
                true_target
            ));
        }
        if primitive.observable() != witness.target() {
            return Err(format!(
                "Witness target {} must match primitive observable {} at table level",
                witness.target(),
                primitive.observable()
            ));
        }
        if primitive.instructions() != native.implementation() {
            return Err(format!(
                "Witness for {} must use the exact native implementation at table level",
                witness.target()
            ));
        }
        if witness.reducer() != OutcomeReducer::Identity {
            return Err(format!(
                "Witness for {} must use Identity reducer at table level",
                witness.target()
            ));
        }
        match witness.byproduct() {
            ByproductEffect::None | ByproductEffect::GaugeOnly => Ok(()),
            ByproductEffect::ApplyPauliIfBitSet { .. } => Err(format!(
                "Witness for {} uses unsupported byproduct effect at table level",
                witness.target()
            )),
        }
    }

    pub fn register_witness(&mut self, witness: MeasurementWitness) -> Result<(), String> {
        if witness.target().0 == 0 {
            return Err("Cannot register witness for the identity".to_string());
        }
        if witness.primitives().is_empty() {
            return Err(format!(
                "Witness for {} must contain at least one primitive",
                witness.target()
            ));
        }
        if witness
            .primitives()
            .iter()
            .any(|primitive| primitive.instructions().is_empty())
        {
            return Err(format!(
                "Witness for {} contains an empty primitive implementation",
                witness.target()
            ));
        }
        self.validate_table_level_witness(&witness)?;

        let target = witness.target();
        if let Some(existing) = self.witnesses.get(&target) {
            if existing == &witness {
                return Ok(());
            }
            return Err(format!(
                "Witness for {target} is already registered with a different definition"
            ));
        }

        self.witnesses.insert(target, witness.clone());
        let should_insert = match self.get(target) {
            None => true,
            Some(existing) => existing.cost() > witness.cost(),
        };
        if should_insert {
            self.insert(MeasurementTableEntry {
                measurement: target,
                conjugated_with: None,
                cost: witness.cost(),
            });
        }
        Ok(())
    }

    pub fn register_witnesses<I>(&mut self, witnesses: I) -> Result<(), String>
    where
        I: IntoIterator<Item = MeasurementWitness>,
    {
        for witness in witnesses {
            self.register_witness(witness)?;
        }
        Ok(())
    }

    pub fn build(&mut self) {
        info!("Synthesizing all measurements from base measurements");

        // 4^12 possible Pauli measurements on 12 qubits
        let nr_paulis: usize = 4_usize.pow(12);

        let mut next_paulis: Vec<_> = self.witnesses.keys().copied().collect();

        // Create a set of base rotations
        // We pick the cheapest rotation for each paulistring, if there is duplication
        let mut base_rots: HashMap<PauliString, MeasurementTableEntry> = HashMap::new();
        for native_impl in self.native_impls() {
            let p = native_impl.implements();
            // Must have pivot support so we can prepare an ancilla there
            if !p.has_pivot_support() {
                continue;
            }

            // Insert cheapest measurement implementation
            base_rots
                .entry(p)
                .and_modify(|cur| {
                    if cur.cost() > native_impl.cost() {
                        *cur = *native_impl;
                    }
                })
                .or_insert(*native_impl);
        }

        debug!(
            "Starting search with {} witness seeds and {} base rotations",
            self.len(),
            base_rots.len()
        );
        for meas in self.native_impls() {
            trace!("Native measurement: {:?}", meas.implements());
        }

        let mut cur = 1; // Count loop iterations by the cost of the current rotation
        while self.len() < nr_paulis {
            let prev_paulis = next_paulis;
            next_paulis = Vec::new();

            cur += 1;
            debug!("Iteration {cur}");

            // Conjugate all rotations of the cur cost by all base measurements to find new rotations
            for prev_pauli in prev_paulis {
                // Tight inner loop of fixed size, maybe optimize somehow by giving compiler hint?
                for (rot_pauli, rot_impl) in base_rots.iter() {
                    let prev_meas = self.get(prev_pauli)
                        .expect("MeasurementTable should contain a previously found Pauli measurement implementation.");
                    let new_rotation_impl = MeasurementTableEntry {
                        measurement: prev_pauli,
                        conjugated_with: Some(*rot_pauli),
                        cost: prev_meas.cost() + 2 * rot_impl.cost(),
                    };

                    let new_pauli = new_rotation_impl.implements();
                    let existing = self.get(new_pauli);
                    match existing {
                        None => {
                            self.insert(new_rotation_impl);
                            next_paulis.push(new_pauli);
                        }
                        Some(existing_impl) => {
                            if existing_impl.cost() > new_rotation_impl.cost() {
                                self.insert(new_rotation_impl);
                                next_paulis.push(new_pauli);
                            }
                        }
                    }
                }
            }

            debug!("Found {} new operations of {} cost", next_paulis.len(), cur);
            debug!("Total operations found: {} / {}", self.len(), nr_paulis);

            if next_paulis.is_empty() {
                error!(
                    "Did not find new operations, aborting. Found {} / {} operations",
                    self.len(),
                    nr_paulis
                );
                for (index, meas_impl) in self.measurements.iter().enumerate() {
                    if meas_impl.is_none() {
                        warn!("Did not find {}", PauliString(index as u32));
                    }
                }
                break;
            }
        }
    }

    /// Try to convert to a complete measurement table
    pub fn complete(self) -> Result<CompleteMeasurementTable, String> {
        self.try_into()
    }

    fn index(p: PauliString) -> usize {
        let i = p.0 as usize;

        assert!(
            i <= 4_usize.pow(12),
            "PauliString {p:?} has index too large"
        );
        i
    }

    /// Look up the implementation for measuring a PauliString
    fn get(&self, p: PauliString) -> Option<&MeasurementTableEntry> {
        self.measurements[MeasurementTableBuilder::index(p)].as_ref()
    }

    /// Insert a MeasurementImpl into the table
    fn insert(&mut self, meas_impl: MeasurementTableEntry) {
        let i = MeasurementTableBuilder::index(meas_impl.implements());
        if self.measurements[i].is_none() {
            self.len += 1;
        }
        self.measurements[i] = Some(meas_impl);
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len > 0
    }

    fn native_impls(&self) -> impl Iterator<Item = &MeasurementTableEntry> {
        self.native_measurements
            .keys()
            .map(|k| self.get(*k).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bicycle_common::Pauli::{I, X, Y, Z};
    use bicycle_common::{AutomorphismData, BicycleISA, TwoBases};

    use crate::{GROSS_MEASUREMENT, TWOGROSS_MEASUREMENT};

    use super::*;

    #[test]
    fn table_constructor() {
        let native = vec![NativeMeasurement {
            automorphism: AutomorphismData::new(0, 0),
            logical: TwoBases::new(X, Y).unwrap(),
        }];

        let mut table = MeasurementTableBuilder::new(native, GROSS_MEASUREMENT);
        assert_eq!(2, table.len());

        let p: PauliString = (&[Y, Y, I, I, I, Y, I, I, I, I, I, Z]).into();
        table.insert(MeasurementTableEntry {
            measurement: p,
            conjugated_with: None,
            cost: 0,
        });

        assert_eq!(3, table.len());
    }

    #[test]
    fn table_insert() {
        let mut table = MeasurementTableBuilder::new(vec![], GROSS_MEASUREMENT);

        let nrs = [
            0b111111111111111111111111,
            0b111111111111111111111110,
            0b000000000000000000000001,
        ];
        for nr in nrs {
            let p = PauliString(nr);
            let p_impl = MeasurementTableEntry {
                measurement: p,
                conjugated_with: None,
                cost: 0,
            };
            table.insert(p_impl);
        }

        assert_eq!(4, table.len());
    }

    #[test]
    fn table_get() {
        let mut table = MeasurementTableBuilder::new(vec![], GROSS_MEASUREMENT);
        let p: PauliString = (&[Y, Y, I, I, I, Y, I, I, I, I, I, Z]).into();
        let p_impl = MeasurementTableEntry {
            measurement: p,
            conjugated_with: None,
            cost: 1,
        };

        table.insert(p_impl);
        assert_eq!(Some(&p_impl), table.get(p));
    }

    #[test]
    fn test_gross_table() -> Result<(), String> {
        table_tests(GROSS_MEASUREMENT)
    }

    #[test]
    fn test_twogross_table() -> Result<(), String> {
        table_tests(TWOGROSS_MEASUREMENT)
    }

    fn table_tests(m: CodeMeasurement) -> Result<(), String> {
        let table: CompleteMeasurementTable = build_complete_table(m)?;
        check_correct_implementation(&table);
        check_native_measurements(&table, m);
        Ok(())
    }

    fn build_complete_table(m: CodeMeasurement) -> Result<CompleteMeasurementTable, String> {
        let mut table = MeasurementTableBuilder::new(NativeMeasurement::all(), m);
        table.build();

        let measurements = table.measurements.clone();
        let res: Vec<_> = measurements.into_iter().flatten().collect();
        assert_eq!(res.len(), table.len);

        table.complete()
    }

    fn check_correct_implementation(complete: &CompleteMeasurementTable) {
        // Check that the completed table gives correct implementations for each pauli string
        for i in 1..4_u32.pow(12) {
            let p = PauliString(i);
            let meas_impl = complete.implementation(p);
            let mut q = meas_impl.base_witness().target();

            for rot in meas_impl.rotations() {
                q = q.conjugate_with(rot.measures().zero_pivot());
            }

            assert_eq!(p, q);
        }
    }

    fn check_native_measurements(table: &CompleteMeasurementTable, code: CodeMeasurement) {
        let native_ps: Vec<_> = NativeMeasurement::all()
            .iter()
            .map(|native| code.measures(native))
            .collect();

        for native_p in native_ps {
            let implementation = table.implementation(native_p);
            assert_eq!(native_p, implementation.measures());
            assert_eq!(0, implementation.rotations().len());
        }
    }

    #[test]
    fn lookup_native_witness_returns_singleton() {
        let native = NativeMeasurement {
            automorphism: AutomorphismData::new(1, 2),
            logical: TwoBases::new(X, Z).unwrap(),
        };
        let target = GROSS_MEASUREMENT.measures(&native);
        let table = CompleteMeasurementTable {
            measurements: vec![],
            native_measurements: HashMap::from([(target, native)]),
            witnesses: HashMap::from([(
                target,
                MeasurementWitness::singleton_native(target, native),
            )]),
        };
        let witness = table
            .lookup_witness(target)
            .expect("native target should have a witness");

        assert_eq!(target, witness.target());
        assert_eq!(1, witness.primitives().len());
        assert_eq!(1, witness.cost());
    }

    #[test]
    fn register_witness_adds_terminal_entry() -> Result<(), String> {
        let mut builder = MeasurementTableBuilder::new(vec![], GROSS_MEASUREMENT);
        let native = NativeMeasurement {
            automorphism: AutomorphismData::new(0, 0),
            logical: TwoBases::new(X, Y).unwrap(),
        };
        let target = GROSS_MEASUREMENT.measures(&native);
        let witness = MeasurementWitness::singleton_native(target, native);
        builder.register_witness(witness.clone())?;

        assert_eq!(Some(&witness), builder.witnesses.get(&target));
        assert_eq!(
            Some(witness.cost()),
            builder.get(target).map(|entry| entry.cost())
        );
        Ok(())
    }

    fn singleton_native_witness(
        target: PauliString,
        native: NativeMeasurement,
        reducer: OutcomeReducer,
        byproduct: ByproductEffect,
        cost: u32,
    ) -> MeasurementWitness {
        MeasurementWitness::new(
            target,
            vec![NativePrimitive::new(
                NativePrimitiveKind::InModule,
                target,
                native.implementation().to_vec(),
                Some(native),
                ResourceSet {
                    pivot: true,
                    ..ResourceSet::default()
                },
                cost,
            )],
            reducer,
            byproduct,
            BTreeMap::new(),
        )
    }

    #[test]
    fn register_witness_rejects_duplicate_target_without_replacing_active_witness() {
        let native = NativeMeasurement {
            automorphism: AutomorphismData::new(0, 0),
            logical: TwoBases::new(X, Y).unwrap(),
        };
        let target = GROSS_MEASUREMENT.measures(&native);
        let mut builder = MeasurementTableBuilder::new(vec![native], GROSS_MEASUREMENT);

        let old_witness = builder
            .witnesses
            .get(&target)
            .expect("expected initial witness")
            .clone();
        let old_cost = builder
            .get(target)
            .expect("expected initial terminal entry")
            .cost();

        let replacement = singleton_native_witness(
            target,
            native,
            OutcomeReducer::Identity,
            ByproductEffect::None,
            old_cost + 5,
        );
        let err = builder
            .register_witness(replacement)
            .expect_err("duplicate target with different witness should fail");
        assert!(err.contains("already registered"));

        assert_eq!(Some(&old_witness), builder.witnesses.get(&target));
        assert_eq!(
            Some(old_cost),
            builder.get(target).map(|entry| entry.cost())
        );
    }

    #[test]
    fn register_witness_rejects_non_inmodule_primitive() {
        let mut builder = MeasurementTableBuilder::new(vec![], GROSS_MEASUREMENT);
        let target = PauliString(0b1011);
        let witness = MeasurementWitness::new(
            target,
            vec![NativePrimitive::new(
                NativePrimitiveKind::CodeCode,
                target,
                vec![BicycleISA::JointMeasure(TwoBases::new(X, Z).unwrap())],
                None,
                ResourceSet {
                    bridge: true,
                    ..ResourceSet::default()
                },
                2,
            )],
            OutcomeReducer::Identity,
            ByproductEffect::None,
            BTreeMap::new(),
        );
        let err = builder
            .register_witness(witness)
            .expect_err("non-InModule primitive should be rejected");
        assert!(err.contains("InModule"));
    }

    #[test]
    fn register_witness_rejects_non_identity_reducer() {
        let native = NativeMeasurement {
            automorphism: AutomorphismData::new(0, 0),
            logical: TwoBases::new(X, Y).unwrap(),
        };
        let mut builder = MeasurementTableBuilder::new(vec![], GROSS_MEASUREMENT);
        let target = GROSS_MEASUREMENT.measures(&native);
        let witness = singleton_native_witness(
            target,
            native,
            OutcomeReducer::XorAll,
            ByproductEffect::None,
            1,
        );
        let err = builder
            .register_witness(witness)
            .expect_err("non-Identity reducer should be rejected");
        assert!(err.contains("Identity reducer"));
    }

    #[test]
    fn register_witness_rejects_apply_pauli_byproduct() {
        let native = NativeMeasurement {
            automorphism: AutomorphismData::new(0, 0),
            logical: TwoBases::new(X, Y).unwrap(),
        };
        let mut builder = MeasurementTableBuilder::new(vec![], GROSS_MEASUREMENT);
        let target = GROSS_MEASUREMENT.measures(&native);
        let witness = singleton_native_witness(
            target,
            native,
            OutcomeReducer::Identity,
            ByproductEffect::ApplyPauliIfBitSet {
                bit_index: 0,
                pauli: target,
            },
            1,
        );
        let err = builder
            .register_witness(witness)
            .expect_err("ApplyPauliIfBitSet should be rejected at table level");
        assert!(err.contains("unsupported byproduct effect"));
    }

    #[test]
    fn register_witness_allows_gauge_only_byproduct() -> Result<(), String> {
        let mut builder = MeasurementTableBuilder::new(vec![], GROSS_MEASUREMENT);
        let native = NativeMeasurement {
            automorphism: AutomorphismData::new(0, 0),
            logical: TwoBases::new(X, Y).unwrap(),
        };
        let target = GROSS_MEASUREMENT.measures(&native);
        let witness = singleton_native_witness(
            target,
            native,
            OutcomeReducer::Identity,
            ByproductEffect::GaugeOnly,
            3,
        );
        builder.register_witness(witness.clone())?;

        assert_eq!(Some(&witness), builder.witnesses.get(&target));
        assert_eq!(
            Some(witness.cost()),
            builder.get(target).map(|entry| entry.cost())
        );
        Ok(())
    }

    #[test]
    fn register_witness_rejects_mismatched_target_and_primitive_observable() {
        let native = NativeMeasurement {
            automorphism: AutomorphismData::new(0, 0),
            logical: TwoBases::new(X, Y).unwrap(),
        };
        let other_native = NativeMeasurement {
            automorphism: AutomorphismData::new(0, 0),
            logical: TwoBases::new(X, Z).unwrap(),
        };
        let target = GROSS_MEASUREMENT.measures(&native);
        let other_target = GROSS_MEASUREMENT.measures(&other_native);
        assert_ne!(target, other_target);

        let mut builder = MeasurementTableBuilder::new(vec![native], GROSS_MEASUREMENT);
        let old_witness = builder
            .witnesses
            .get(&target)
            .expect("expected initial witness")
            .clone();
        let old_cost = builder
            .get(target)
            .expect("expected initial terminal entry")
            .cost();

        let mismatched = MeasurementWitness::new(
            target,
            vec![NativePrimitive::new(
                NativePrimitiveKind::InModule,
                other_target,
                native.implementation().to_vec(),
                Some(native),
                ResourceSet {
                    pivot: true,
                    ..ResourceSet::default()
                },
                old_cost + 1,
            )],
            OutcomeReducer::Identity,
            ByproductEffect::None,
            BTreeMap::new(),
        );
        let err = builder
            .register_witness(mismatched)
            .expect_err("target/primitive observable mismatch should be rejected");
        assert!(err.contains("must match primitive observable"));
        assert_eq!(Some(&old_witness), builder.witnesses.get(&target));
        assert_eq!(
            Some(old_cost),
            builder.get(target).map(|entry| entry.cost())
        );
    }

    #[test]
    fn register_witness_rejects_target_not_equal_code_measures_native() {
        let native = NativeMeasurement {
            automorphism: AutomorphismData::new(0, 0),
            logical: TwoBases::new(X, Y).unwrap(),
        };
        let other_native = NativeMeasurement {
            automorphism: AutomorphismData::new(0, 0),
            logical: TwoBases::new(X, Z).unwrap(),
        };
        let true_target = GROSS_MEASUREMENT.measures(&native);
        let fake_target = GROSS_MEASUREMENT.measures(&other_native);
        assert_ne!(true_target, fake_target);

        let mut builder = MeasurementTableBuilder::new(vec![native], GROSS_MEASUREMENT);
        let old_witness = builder
            .witnesses
            .get(&true_target)
            .expect("expected initial witness")
            .clone();
        let old_cost = builder
            .get(true_target)
            .expect("expected initial terminal entry")
            .cost();

        let mismatched = MeasurementWitness::new(
            fake_target,
            vec![NativePrimitive::new(
                NativePrimitiveKind::InModule,
                fake_target,
                native.implementation().to_vec(),
                Some(native),
                ResourceSet {
                    pivot: true,
                    ..ResourceSet::default()
                },
                1,
            )],
            OutcomeReducer::Identity,
            ByproductEffect::None,
            BTreeMap::new(),
        );
        let err = builder
            .register_witness(mismatched)
            .expect_err("target != code.measures(native) should be rejected");
        assert!(err.contains("does not match code.measures(native)"));
        assert!(builder.witnesses.get(&fake_target).is_none());
        assert_eq!(None, builder.get(fake_target).map(|entry| entry.cost()));
        assert_eq!(Some(&old_witness), builder.witnesses.get(&true_target));
        assert_eq!(
            Some(old_cost),
            builder.get(true_target).map(|entry| entry.cost())
        );
    }

    #[test]
    fn register_witness_rejects_inmodule_primitive_without_native_measurement() {
        let native = NativeMeasurement {
            automorphism: AutomorphismData::new(0, 0),
            logical: TwoBases::new(X, Y).unwrap(),
        };
        let target = GROSS_MEASUREMENT.measures(&native);
        let mut builder = MeasurementTableBuilder::new(vec![native], GROSS_MEASUREMENT);
        let old_witness = builder
            .witnesses
            .get(&target)
            .expect("expected initial witness")
            .clone();
        let old_cost = builder
            .get(target)
            .expect("expected initial terminal entry")
            .cost();

        let hand_rolled = MeasurementWitness::new(
            target,
            vec![NativePrimitive::new(
                NativePrimitiveKind::InModule,
                target,
                native.implementation().to_vec(),
                None,
                ResourceSet {
                    pivot: true,
                    ..ResourceSet::default()
                },
                old_cost + 2,
            )],
            OutcomeReducer::Identity,
            ByproductEffect::None,
            BTreeMap::new(),
        );
        let err = builder
            .register_witness(hand_rolled)
            .expect_err("hand-rolled InModule primitive should be rejected");
        assert!(err.contains("true native measurement"));
        assert_eq!(Some(&old_witness), builder.witnesses.get(&target));
        assert_eq!(
            Some(old_cost),
            builder.get(target).map(|entry| entry.cost())
        );
    }

    #[test]
    fn register_witness_rejects_multi_primitive_table_level_witness() {
        let native = NativeMeasurement {
            automorphism: AutomorphismData::new(0, 0),
            logical: TwoBases::new(X, Y).unwrap(),
        };
        let target = GROSS_MEASUREMENT.measures(&native);
        let mut builder = MeasurementTableBuilder::new(vec![native], GROSS_MEASUREMENT);
        let old_witness = builder
            .witnesses
            .get(&target)
            .expect("expected initial witness")
            .clone();
        let old_cost = builder
            .get(target)
            .expect("expected initial terminal entry")
            .cost();

        let invalid = MeasurementWitness::new(
            target,
            vec![
                NativePrimitive::new(
                    NativePrimitiveKind::InModule,
                    target,
                    native.implementation().to_vec(),
                    Some(native),
                    ResourceSet {
                        pivot: true,
                        ..ResourceSet::default()
                    },
                    1,
                ),
                NativePrimitive::new(
                    NativePrimitiveKind::InModule,
                    target,
                    native.implementation().to_vec(),
                    Some(native),
                    ResourceSet {
                        pivot: true,
                        ..ResourceSet::default()
                    },
                    1,
                ),
            ],
            OutcomeReducer::Identity,
            ByproductEffect::None,
            BTreeMap::new(),
        );
        let err = builder
            .register_witness(invalid)
            .expect_err("multi-primitive table-level witness should be rejected");
        assert!(err.contains("singleton native terminal"));
        assert_eq!(Some(&old_witness), builder.witnesses.get(&target));
        assert_eq!(
            Some(old_cost),
            builder.get(target).map(|entry| entry.cost())
        );
    }

    #[test]
    fn register_witness_rejects_mismatched_target_before_semantic_harness_can_be_misled() {
        let native = NativeMeasurement {
            automorphism: AutomorphismData::new(0, 0),
            logical: TwoBases::new(X, Y).unwrap(),
        };
        let other_native = NativeMeasurement {
            automorphism: AutomorphismData::new(0, 0),
            logical: TwoBases::new(X, Z).unwrap(),
        };
        let target = GROSS_MEASUREMENT.measures(&native);
        let other_target = GROSS_MEASUREMENT.measures(&other_native);
        assert_ne!(target, other_target);

        let mut builder = MeasurementTableBuilder::new(vec![], GROSS_MEASUREMENT);
        let mismatched = MeasurementWitness::new(
            target,
            vec![NativePrimitive::new(
                NativePrimitiveKind::InModule,
                other_target,
                native.implementation().to_vec(),
                Some(native),
                ResourceSet {
                    pivot: true,
                    ..ResourceSet::default()
                },
                1,
            )],
            OutcomeReducer::Identity,
            ByproductEffect::None,
            BTreeMap::new(),
        );

        let err = builder
            .register_witness(mismatched)
            .expect_err("mismatched target should be rejected during registration");
        assert!(err.contains("must match primitive observable"));
    }

    #[test]
    fn register_witness_accepts_true_singleton_native_wrapper() -> Result<(), String> {
        let native = NativeMeasurement {
            automorphism: AutomorphismData::new(1, 2),
            logical: TwoBases::new(X, Z).unwrap(),
        };
        let target = GROSS_MEASUREMENT.measures(&native);
        let witness = singleton_native_witness(
            target,
            native,
            OutcomeReducer::Identity,
            ByproductEffect::None,
            7,
        );
        let mut builder = MeasurementTableBuilder::new(vec![], GROSS_MEASUREMENT);
        builder.register_witness(witness.clone())?;

        assert_eq!(Some(&witness), builder.witnesses.get(&target));
        assert_eq!(
            Some(witness.cost()),
            builder.get(target).map(|entry| entry.cost())
        );
        Ok(())
    }

    #[test]
    fn outcome_reducer_variants() {
        let frame = FrameState { sign_parity: 1 };
        let bits = [1, 1, 0, 1];

        assert_eq!(0, OutcomeReducer::Identity.reduce(&bits, frame));
        assert_eq!(1, OutcomeReducer::XorFirstTwo.reduce(&bits, frame));
        assert_eq!(0, OutcomeReducer::XorAll.reduce(&bits, frame));
    }

    #[test]
    fn byproduct_effect_apply_pauli_if_bit_set() {
        let pauli = PauliString(0b1010);
        let effect = ByproductEffect::ApplyPauliIfBitSet {
            bit_index: 1,
            pauli,
        };
        assert_eq!(
            ByproductEffect::ApplyPauliIfBitSet {
                bit_index: 1,
                pauli
            },
            effect.effect(&[0, 1])
        );
        assert_eq!(ByproductEffect::None, effect.effect(&[0, 0]));
    }

    #[test]
    fn witness_evaluate_returns_logical_and_effect() {
        let pauli = PauliString(0b11);
        let witness = MeasurementWitness::new(
            pauli,
            vec![NativePrimitive::new(
                NativePrimitiveKind::CodeCode,
                pauli,
                vec![BicycleISA::JointMeasure(TwoBases::new(X, Z).unwrap())],
                None,
                ResourceSet {
                    bridge: true,
                    ..ResourceSet::default()
                },
                2,
            )],
            OutcomeReducer::XorAll,
            ByproductEffect::ApplyPauliIfBitSet {
                bit_index: 0,
                pauli,
            },
            BTreeMap::new(),
        );

        let (logical, effect) = witness.evaluate(&[1, 1, 0], FrameState::default());
        assert_eq!(0, logical);
        assert_eq!(
            ByproductEffect::ApplyPauliIfBitSet {
                bit_index: 0,
                pauli
            },
            effect
        );
    }
}
