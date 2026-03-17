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

use serde::{Deserialize, Serialize};

use crate::draft_trace::{
    grouped_by_local_measurement_count, grouped_by_rule, grouped_by_transition_row,
};
use crate::draft_types::{
    DraftBuildResult, DraftHistograms, LibraryEntry, RuleClass, TransitionRow,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleGroupReport {
    pub rule: RuleClass,
    pub entries: Vec<LibraryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionGroupReport {
    pub transition: TransitionRow,
    pub entries: Vec<LibraryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalCountGroupReport {
    pub local_measurement_count: u8,
    pub entries: Vec<LibraryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftJsonReport {
    pub full_coverage: bool,
    pub library_size: usize,
    pub histograms: DraftHistograms,
    pub by_rule: Vec<RuleGroupReport>,
    pub by_transition_row: Vec<TransitionGroupReport>,
    pub by_local_measurement_count: Vec<LocalCountGroupReport>,
}

pub fn build_json_report(build: &DraftBuildResult) -> DraftJsonReport {
    let by_rule = grouped_by_rule(build)
        .into_iter()
        .map(|(rule, entries)| RuleGroupReport { rule, entries })
        .collect();

    let by_transition_row = grouped_by_transition_row(build)
        .into_iter()
        .map(|(transition, entries)| TransitionGroupReport {
            transition,
            entries,
        })
        .collect();

    let by_local_measurement_count = grouped_by_local_measurement_count(build)
        .into_iter()
        .map(|(local_measurement_count, entries)| LocalCountGroupReport {
            local_measurement_count,
            entries,
        })
        .collect();

    DraftJsonReport {
        full_coverage: build.full_coverage,
        library_size: build.library_size(),
        histograms: build.histograms.clone(),
        by_rule,
        by_transition_row,
        by_local_measurement_count,
    }
}

pub fn report_to_json_pretty(build: &DraftBuildResult) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&build_json_report(build))
}
