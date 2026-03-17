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

use std::collections::BTreeMap;

use crate::draft_types::{
    DraftBuildResult, InsertionWitness, LibraryEntry, RuleClass, Tail11, TransitionRow,
};

pub fn explain_tail(build: &DraftBuildResult, tail: Tail11) -> Option<LibraryEntry> {
    build.library.get(&tail).copied()
}

pub fn grouped_by_rule(build: &DraftBuildResult) -> BTreeMap<RuleClass, Vec<LibraryEntry>> {
    let mut grouped = BTreeMap::new();
    for entry in build.library.values().copied() {
        grouped
            .entry(entry.first_rule)
            .or_insert_with(Vec::new)
            .push(entry);
    }
    sort_group_entries(&mut grouped);
    grouped
}

pub fn grouped_by_transition_row(
    build: &DraftBuildResult,
) -> BTreeMap<TransitionRow, Vec<LibraryEntry>> {
    let mut grouped = BTreeMap::new();
    for entry in build.library.values().copied() {
        let row = match entry.witness {
            InsertionWitness::P3(witness) => TransitionRow {
                prev_head: witness.prev_head,
                q_basis: witness.q_basis,
                next_head: witness.next_head,
                delta_stage_cost: witness.delta_stage_cost as u8,
                delta_local_measurement_count: witness.delta_local_measurement_count,
            },
            _ => continue,
        };
        grouped.entry(row).or_insert_with(Vec::new).push(entry);
    }
    sort_group_entries(&mut grouped);
    grouped
}

pub fn grouped_by_local_measurement_count(
    build: &DraftBuildResult,
) -> BTreeMap<u8, Vec<LibraryEntry>> {
    let mut grouped = BTreeMap::new();
    for entry in build.library.values().copied() {
        grouped
            .entry(entry.local_measurement_count)
            .or_insert_with(Vec::new)
            .push(entry);
    }
    sort_group_entries(&mut grouped);
    grouped
}

pub fn format_explain_entry(entry: &LibraryEntry) -> String {
    format!(
        "tail={} rule={:?} stage_cost={} local_count={} head={}",
        entry.tail,
        entry.first_rule,
        entry.insertion_stage_cost,
        entry.local_measurement_count,
        entry.resulting_head,
    )
}

fn sort_group_entries<K: Ord>(groups: &mut BTreeMap<K, Vec<LibraryEntry>>) {
    for entries in groups.values_mut() {
        entries.sort_by_key(|entry| entry.tail);
    }
}
