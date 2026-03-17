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

/*
Implementation notes from advisor handwritten draft
===================================================

- M := ∅, where elements are 11-qubit Paulis P ∈ {I,X,Y,Z}^{⊗11}
- N is the set of 540 native measurements in {I,X,Y,Z}^{⊗12}

- P0:
  - if (I, P) ∈ N, add P into M, cost = 1
  - if (Q, P) ∈ N, Q ∈ {X,Y,Z}, add P into M, cost = 2

- P2:
  - if (Q, P) ∈ N, (Q, P') ∈ N, and P, P' commute,
    add P xor P' into M, cost = 4

- P3:
  - if M_elem ∈ M, (Q, P) ∈ N, and P, M_elem anti-commute,
    add M_elem xor P into M
  - the handwritten draft indicates this step is stateful and not a naive tail-only closure

Also explicitly:
- P2 is same-Q bucket only
- P3 depends on head-state transitions
- local measurement count and stage cost must be separated
*/

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap};

use crate::draft_types::{
    DraftBuildResult, DraftHistograms, Head, InsertionWitness, LibraryEntry, NativeRow,
    NativeRowMetadata, P0IWitness, P0QWitness, P2Witness, P3Witness, RuleClass, StateEntry,
    StateKey, Tail11, TransitionOutcome, TAIL_SPACE_SIZE,
};
use crate::measurement::CodeMeasurement;
use crate::native_measurement::NativeMeasurement;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct QueueItem {
    stage_cost: u32,
    key: StateKey,
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering to use BinaryHeap as a min-heap by stage_cost.
        other
            .stage_cost
            .cmp(&self.stage_cost)
            .then_with(|| self.key.tail.cmp(&other.key.tail))
            .then_with(|| self.key.head.cmp(&other.key.head))
    }
}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn state_rule_class(state: &StateEntry) -> RuleClass {
    match state.witness {
        InsertionWitness::P0I(_) => RuleClass::P0I,
        InsertionWitness::P0Q(_) => RuleClass::P0Q,
        InsertionWitness::P2(_) => RuleClass::P2,
        InsertionWitness::P3(witness) => {
            if witness.delta_local_measurement_count == 4 {
                RuleClass::P3Delta4
            } else {
                RuleClass::P3Delta6
            }
        }
    }
}

fn library_entry_from_state(state: &StateEntry) -> LibraryEntry {
    LibraryEntry {
        tail: state.key.tail,
        resulting_head: state.key.head,
        first_rule: state_rule_class(state),
        insertion_stage_cost: state.insertion_stage_cost,
        local_measurement_count: state.local_measurement_count,
        witness: state.witness,
    }
}

fn better_state_candidate(candidate: &StateEntry, existing: &StateEntry) -> bool {
    candidate.insertion_stage_cost < existing.insertion_stage_cost
        || (candidate.insertion_stage_cost == existing.insertion_stage_cost
            && candidate.local_measurement_count < existing.local_measurement_count)
}

fn better_library_candidate(candidate: &LibraryEntry, existing: &LibraryEntry) -> bool {
    candidate.insertion_stage_cost < existing.insertion_stage_cost
        || (candidate.insertion_stage_cost == existing.insertion_stage_cost
            && candidate.local_measurement_count < existing.local_measurement_count)
        || (candidate.insertion_stage_cost == existing.insertion_stage_cost
            && candidate.local_measurement_count == existing.local_measurement_count
            && candidate.first_rule < existing.first_rule)
}

fn try_insert_state(
    states: &mut HashMap<StateKey, StateEntry>,
    frontier: &mut BinaryHeap<QueueItem>,
    candidate: StateEntry,
) {
    let should_insert = match states.get(&candidate.key) {
        None => true,
        Some(existing) => better_state_candidate(&candidate, existing),
    };

    if should_insert {
        frontier.push(QueueItem {
            stage_cost: candidate.insertion_stage_cost,
            key: candidate.key,
        });
        states.insert(candidate.key, candidate);
    }
}

fn build_histograms(library: &HashMap<Tail11, LibraryEntry>) -> DraftHistograms {
    let mut insertion_stage_cost_hist = BTreeMap::new();
    let mut local_measurement_count_hist = BTreeMap::new();
    let mut rule_hist = BTreeMap::new();

    for entry in library.values() {
        *insertion_stage_cost_hist
            .entry(entry.insertion_stage_cost)
            .or_insert(0) += 1;
        *local_measurement_count_hist
            .entry(entry.local_measurement_count)
            .or_insert(0) += 1;
        *rule_hist.entry(entry.first_rule).or_insert(0) += 1;
    }

    DraftHistograms {
        direct_use_cost_hist: BTreeMap::new(),
        insertion_stage_cost_hist,
        local_measurement_count_hist,
        rule_hist,
    }
}

fn direct_use_cost_histogram(library_size: usize, full_coverage: bool) -> BTreeMap<u32, u64> {
    let mut hist = BTreeMap::new();
    let direct_cost_count = if full_coverage {
        TAIL_SPACE_SIZE as u64
    } else {
        library_size as u64
    };
    hist.insert(1, direct_cost_count);
    hist
}

pub fn transition(prev_head: Head, q_basis: Head) -> TransitionOutcome {
    match (prev_head, q_basis) {
        (Head::I, Head::X) => TransitionOutcome {
            next_head: Head::X,
            delta_stage_cost: 5,
            delta_local_measurement_count: 4,
        },
        (Head::I, Head::Y) => TransitionOutcome {
            next_head: Head::Y,
            delta_stage_cost: 5,
            delta_local_measurement_count: 4,
        },
        (Head::I, Head::Z) => TransitionOutcome {
            next_head: Head::Z,
            delta_stage_cost: 5,
            delta_local_measurement_count: 4,
        },
        (Head::X, Head::X) => TransitionOutcome {
            next_head: Head::I,
            delta_stage_cost: 6,
            delta_local_measurement_count: 6,
        },
        (Head::X, Head::Y) => TransitionOutcome {
            next_head: Head::Z,
            delta_stage_cost: 6,
            delta_local_measurement_count: 6,
        },
        (Head::X, Head::Z) => TransitionOutcome {
            next_head: Head::Y,
            delta_stage_cost: 6,
            delta_local_measurement_count: 6,
        },
        (Head::Y, Head::X) => TransitionOutcome {
            next_head: Head::Z,
            delta_stage_cost: 6,
            delta_local_measurement_count: 6,
        },
        (Head::Y, Head::Y) => TransitionOutcome {
            next_head: Head::I,
            delta_stage_cost: 6,
            delta_local_measurement_count: 6,
        },
        (Head::Y, Head::Z) => TransitionOutcome {
            next_head: Head::X,
            delta_stage_cost: 6,
            delta_local_measurement_count: 6,
        },
        (Head::Z, Head::X) => TransitionOutcome {
            next_head: Head::Y,
            delta_stage_cost: 6,
            delta_local_measurement_count: 6,
        },
        (Head::Z, Head::Y) => TransitionOutcome {
            next_head: Head::X,
            delta_stage_cost: 6,
            delta_local_measurement_count: 6,
        },
        (Head::Z, Head::Z) => TransitionOutcome {
            next_head: Head::I,
            delta_stage_cost: 6,
            delta_local_measurement_count: 6,
        },
        (_, Head::I) => panic!("P3 transition requires q_basis in {{X,Y,Z}}"),
    }
}

pub fn native_rows_for_code(code: CodeMeasurement) -> Vec<NativeRow> {
    NativeMeasurement::all()
        .into_iter()
        .enumerate()
        .map(|(index, native)| {
            let measured = code.measures(&native);
            NativeRow {
                index,
                head: Head::from_pauli(measured.get_pauli(0)),
                tail: Tail11::new(measured.logical_bits().0),
                metadata: Some(NativeRowMetadata {
                    measured_pauli_bits: measured.0,
                    logical: native.logical,
                    automorphism: native.automorphism,
                }),
            }
        })
        .collect()
}

pub fn build_draft_library_for_code(code: CodeMeasurement) -> DraftBuildResult {
    let native_rows = native_rows_for_code(code);
    build_draft_library(&native_rows)
}

pub fn build_draft_library(native_rows: &[NativeRow]) -> DraftBuildResult {
    let mut states: HashMap<StateKey, StateEntry> = HashMap::new();
    let mut library: HashMap<Tail11, LibraryEntry> = HashMap::new();
    let mut frontier = BinaryHeap::new();

    let mut non_i_native_rows = Vec::new();
    let mut same_q_buckets: BTreeMap<Head, Vec<NativeRow>> = BTreeMap::new();

    for row in native_rows.iter().copied() {
        match row.head {
            Head::I => {
                let state = StateEntry {
                    key: StateKey {
                        tail: row.tail,
                        head: Head::I,
                    },
                    insertion_stage_cost: 1,
                    local_measurement_count: 1,
                    witness: InsertionWitness::P0I(P0IWitness {
                        native_index: row.index,
                    }),
                };
                try_insert_state(&mut states, &mut frontier, state);
            }
            head @ (Head::X | Head::Y | Head::Z) => {
                non_i_native_rows.push(row);
                same_q_buckets.entry(head).or_default().push(row);
                let state = StateEntry {
                    key: StateKey {
                        tail: row.tail,
                        head,
                    },
                    insertion_stage_cost: 2,
                    local_measurement_count: 2,
                    witness: InsertionWitness::P0Q(P0QWitness {
                        q_basis: head,
                        native_index: row.index,
                    }),
                };
                try_insert_state(&mut states, &mut frontier, state);
            }
        }
    }

    // P2: combine only within the same non-I head bucket.
    for q_basis in Head::NON_I {
        let bucket = same_q_buckets.get(&q_basis).cloned().unwrap_or_default();
        for i in 0..bucket.len() {
            for j in (i + 1)..bucket.len() {
                let left = bucket[i];
                let right = bucket[j];
                if !left.tail.commutes_with(right.tail) {
                    continue;
                }
                let state = StateEntry {
                    key: StateKey {
                        tail: left.tail ^ right.tail,
                        head: Head::I,
                    },
                    insertion_stage_cost: 4,
                    local_measurement_count: 4,
                    witness: InsertionWitness::P2(P2Witness {
                        q_basis,
                        left_tail: left.tail,
                        right_tail: right.tail,
                        left_native_index: left.index,
                        right_native_index: right.index,
                    }),
                };
                try_insert_state(&mut states, &mut frontier, state);
            }
        }
    }

    // Dijkstra over stateful (tail, head) nodes. P2/P3 products are inserted back into the same
    // reusable state space, so they are available as future primitives.
    while let Some(item) = frontier.pop() {
        let state = match states.get(&item.key).copied() {
            None => continue,
            Some(state) => {
                if state.insertion_stage_cost != item.stage_cost {
                    continue;
                }
                state
            }
        };

        let library_candidate = library_entry_from_state(&state);
        let should_insert_library = match library.get(&state.key.tail) {
            None => true,
            Some(existing) => better_library_candidate(&library_candidate, existing),
        };
        if should_insert_library {
            library.insert(state.key.tail, library_candidate);
        }

        if library.len() == TAIL_SPACE_SIZE as usize {
            break;
        }

        // P3 closure with explicit transition table.
        for row in non_i_native_rows.iter().copied() {
            if !state.key.tail.anticommutes_with(row.tail) {
                continue;
            }
            let outcome = transition(state.key.head, row.head);
            let next_state = StateEntry {
                key: StateKey {
                    tail: state.key.tail ^ row.tail,
                    head: outcome.next_head,
                },
                insertion_stage_cost: state.insertion_stage_cost + outcome.delta_stage_cost,
                local_measurement_count: outcome.delta_local_measurement_count,
                witness: InsertionWitness::P3(P3Witness {
                    prev_head: state.key.head,
                    prev_tail: state.key.tail,
                    q_basis: row.head,
                    axis_tail: row.tail,
                    next_head: outcome.next_head,
                    delta_stage_cost: outcome.delta_stage_cost,
                    delta_local_measurement_count: outcome.delta_local_measurement_count,
                    prev_state: state.key,
                    native_index: row.index,
                }),
            };
            try_insert_state(&mut states, &mut frontier, next_state);
        }
    }

    let full_coverage = library.len() == TAIL_SPACE_SIZE as usize;
    let mut histograms = build_histograms(&library);
    histograms.direct_use_cost_hist = direct_use_cost_histogram(library.len(), full_coverage);

    DraftBuildResult {
        native_rows: native_rows.to_vec(),
        states,
        library,
        histograms,
        full_coverage,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use bicycle_common::Pauli;

    use crate::draft_trace::{
        explain_tail, grouped_by_local_measurement_count, grouped_by_rule,
        grouped_by_transition_row,
    };

    use super::*;

    fn row(index: usize, head: Head, tail: Tail11) -> NativeRow {
        NativeRow {
            index,
            head,
            tail,
            metadata: None,
        }
    }

    #[test]
    fn symplectic_roundtrip_and_algebra() {
        let tail = Tail11::from_label("IXYZIIXYZII").unwrap();
        assert_eq!(tail.to_label(), "IXYZIIXYZII");

        let x = Tail11::from_paulis([Pauli::X; 11]);
        let z = Tail11::from_paulis([Pauli::Z; 11]);
        let y = x ^ z;

        assert!(x.anticommutes_with(z));
        assert_eq!(y, Tail11::from_paulis([Pauli::Y; 11]));
    }

    #[test]
    fn p0_rule_classes_and_local_counts() {
        let p = Tail11::from_label("XIIIIIIIIII").unwrap();
        let rows = vec![row(0, Head::I, p), row(1, Head::X, p)];

        let result = build_draft_library(&rows);
        let entry = result.library.get(&p).unwrap();

        assert!(matches!(entry.first_rule, RuleClass::P0I | RuleClass::P0Q));
        assert!(entry.local_measurement_count == 1 || entry.local_measurement_count == 2);
        assert!(result
            .histograms
            .local_measurement_count_hist
            .keys()
            .all(|k| *k == 1 || *k == 2));
    }

    #[test]
    fn p0_i_inserts_with_local_count_1() {
        let p = Tail11::from_label("IXIIIIIIIII").unwrap();
        let rows = vec![row(0, Head::I, p)];
        let result = build_draft_library(&rows);
        let entry = result.library.get(&p).unwrap();
        assert_eq!(entry.first_rule, RuleClass::P0I);
        assert_eq!(entry.local_measurement_count, 1);
        assert_eq!(entry.insertion_stage_cost, 1);
    }

    #[test]
    fn p0_q_inserts_with_local_count_2() {
        let p = Tail11::from_label("IIXIIIIIIII").unwrap();
        let rows = vec![row(0, Head::Y, p)];
        let result = build_draft_library(&rows);
        let entry = result.library.get(&p).unwrap();
        assert_eq!(entry.first_rule, RuleClass::P0Q);
        assert_eq!(entry.local_measurement_count, 2);
        assert_eq!(entry.insertion_stage_cost, 2);
    }

    #[test]
    fn p2_is_same_basis_only_and_inserts_into_library() {
        let p = Tail11::from_label("XIIIIIIIIII").unwrap();
        let p_prime = Tail11::from_label("IXIIIIIIIII").unwrap();
        let cross = Tail11::from_label("IIXIIIIIIII").unwrap();

        let rows = vec![
            row(0, Head::X, p),
            row(1, Head::X, p_prime),
            row(2, Head::Y, cross),
        ];

        let result = build_draft_library(&rows);
        let expected = p ^ p_prime;

        let entry = result.library.get(&expected).unwrap();
        assert_eq!(entry.first_rule, RuleClass::P2);
        assert_eq!(entry.local_measurement_count, 4);
        if let InsertionWitness::P2(witness) = entry.witness {
            assert_eq!(witness.q_basis, Head::X);
        } else {
            panic!("Expected P2 witness");
        }
    }

    #[test]
    fn p2_cross_basis_pairs_are_forbidden() {
        let left = Tail11::from_label("XIIIIIIIIII").unwrap();
        let right = Tail11::from_label("IXIIIIIIIII").unwrap();
        let expected_cross_product = left ^ right;
        let rows = vec![row(0, Head::X, left), row(1, Head::Y, right)];
        let result = build_draft_library(&rows);

        assert!(
            result.library.get(&expected_cross_product).is_none(),
            "Cross-basis pair must not produce a P2 insertion"
        );
    }

    #[test]
    fn p3_transition_table_uses_delta4_for_identity_rows() {
        let base = Tail11::from_label("XIIIIIIIIII").unwrap();
        let anti_x = Tail11::from_label("ZIIIIIIIIII").unwrap();
        let rows = vec![row(0, Head::I, base), row(1, Head::X, anti_x)];

        let result = build_draft_library(&rows);

        let mut saw_delta4 = false;
        for entry in result.library.values() {
            if let InsertionWitness::P3(witness) = entry.witness {
                if witness.prev_head == Head::I && witness.q_basis == Head::X {
                    assert_eq!(witness.next_head, Head::X);
                    assert_eq!(witness.delta_stage_cost, 5);
                    assert_eq!(witness.delta_local_measurement_count, 4);
                    saw_delta4 = true;
                }
                assert_ne!(witness.delta_local_measurement_count, 5);
                assert!(witness.delta_local_measurement_count < 7);
            }
        }

        assert!(saw_delta4, "Expected at least one (I, X)->X P3 transition");
    }

    #[test]
    fn transition_identity_rows_are_delta4_and_stage5() {
        let rows = [(Head::X, Head::X), (Head::Y, Head::Y), (Head::Z, Head::Z)];
        for (q_basis, expected_next_head) in rows {
            let outcome = transition(Head::I, q_basis);
            assert_eq!(outcome.next_head, expected_next_head);
            assert_eq!(outcome.delta_stage_cost, 5);
            assert_eq!(outcome.delta_local_measurement_count, 4);
        }
    }

    #[test]
    fn p3_requires_anticommutation() {
        let tail = Tail11::from_label("XIIIIIIIIII").unwrap();
        let rows = vec![row(0, Head::I, tail), row(1, Head::X, tail)];
        let result = build_draft_library(&rows);

        assert!(result
            .library
            .values()
            .all(|entry| !matches!(entry.witness, InsertionWitness::P3(_))));
    }

    #[test]
    fn histogram_families_are_separated_and_local_keys_are_bounded() {
        let p = Tail11::from_label("XIIIIIIIIII").unwrap();
        let p2 = Tail11::from_label("IXIIIIIIIII").unwrap();
        let p3 = Tail11::from_label("ZIIIIIIIIII").unwrap();

        let rows = vec![
            row(0, Head::I, p),
            row(1, Head::X, p),
            row(2, Head::X, p2),
            row(3, Head::Y, p3),
        ];

        let result = build_draft_library(&rows);

        let local_keys: BTreeSet<_> = result
            .histograms
            .local_measurement_count_hist
            .keys()
            .copied()
            .collect();
        assert!(local_keys.is_subset(&BTreeSet::from([1, 2, 4, 6])));
        assert!(!local_keys.contains(&5));

        assert_eq!(
            result.histograms.direct_use_cost_hist.get(&1),
            Some(&(result.library.len() as u64))
        );
        assert!(!result.histograms.insertion_stage_cost_hist.is_empty());
        assert!(!result.histograms.rule_hist.is_empty());
    }

    #[test]
    fn explain_and_grouped_forensics_include_p3_transition_fields() {
        let base = Tail11::from_label("XIIIIIIIIII").unwrap();
        let anti = Tail11::from_label("ZIIIIIIIIII").unwrap();

        let rows = vec![row(0, Head::I, base), row(1, Head::X, anti)];
        let result = build_draft_library(&rows);

        let generated_tail = base ^ anti;
        let explain = explain_tail(&result, generated_tail).unwrap();
        assert!(matches!(
            explain.first_rule,
            RuleClass::P3Delta4 | RuleClass::P3Delta6
        ));

        if let InsertionWitness::P3(witness) = explain.witness {
            assert_eq!(witness.prev_head, Head::I);
            assert_eq!(witness.q_basis, Head::X);
            assert_eq!(witness.next_head, Head::X);
            assert_eq!(explain.insertion_stage_cost, 6);
            assert_eq!(explain.local_measurement_count, 4);
        } else {
            panic!("Expected P3 witness");
        }

        let by_rule = grouped_by_rule(&result);
        assert!(!by_rule.is_empty());

        let by_transition = grouped_by_transition_row(&result);
        assert!(!by_transition.is_empty());

        let by_local = grouped_by_local_measurement_count(&result);
        assert!(!by_local.is_empty());
    }

    #[test]
    fn explain_distinguishes_p0_p2_p3() {
        let p0_tail = Tail11::from_label("XIIIIIIIIII").unwrap();
        let p2_left = Tail11::from_label("IXIIIIIIIII").unwrap();
        let p2_right = Tail11::from_label("IIXIIIIIIII").unwrap();
        let p3_axis = Tail11::from_label("ZIIIIIIIIII").unwrap();
        let rows = vec![
            row(0, Head::I, p0_tail),
            row(1, Head::X, p2_left),
            row(2, Head::X, p2_right),
            row(3, Head::X, p3_axis),
        ];

        let result = build_draft_library(&rows);

        let p0_explain = explain_tail(&result, p0_tail).unwrap();
        assert_eq!(p0_explain.first_rule, RuleClass::P0I);

        let p2_tail = p2_left ^ p2_right;
        let p2_explain = explain_tail(&result, p2_tail).unwrap();
        assert_eq!(p2_explain.first_rule, RuleClass::P2);

        let p3_tail = p0_tail ^ p3_axis;
        let p3_explain = explain_tail(&result, p3_tail).unwrap();
        assert!(matches!(
            p3_explain.first_rule,
            RuleClass::P3Delta4 | RuleClass::P3Delta6
        ));
    }

    #[test]
    fn direct_use_hist_matches_full_coverage_definition() {
        let hist = direct_use_cost_histogram(17, true);
        assert_eq!(hist, BTreeMap::from([(1, TAIL_SPACE_SIZE as u64)]));
    }
}
