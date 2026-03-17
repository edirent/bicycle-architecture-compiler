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

pub mod measurement;
pub use measurement::{
    CodeMeasurement, MeasurementChoices, GROSS_MEASUREMENT, TWOGROSS_MEASUREMENT,
};

pub mod native_measurement;
mod pauli_string;

pub use pauli_string::PauliString;

pub mod draft_types;
pub use draft_types::{
    DraftBuildResult, DraftHistograms, Head, InsertionWitness, LibraryEntry, NativeRow, RuleClass,
    StateEntry, StateKey, Tail11, TransitionOutcome, TransitionRow, TAIL_QUBITS, TAIL_SPACE_SIZE,
};

pub mod draft_core;
pub use draft_core::{
    build_draft_library, build_draft_library_for_code, native_rows_for_code, transition,
};

pub mod draft_trace;
pub use draft_trace::{
    explain_tail, format_explain_entry, grouped_by_local_measurement_count, grouped_by_rule,
    grouped_by_transition_row,
};

pub mod draft_reports;
pub use draft_reports::{build_json_report, report_to_json_pretty, DraftJsonReport};

pub mod draft_single_shot;
pub use draft_single_shot::{
    compute_ours_single_shot_exact_hist_for_choice, compute_ours_single_shot_exact_hist_for_code,
    compute_ours_single_shot_exact_hist_from_native_rows,
    compute_paper_baseline_exact_hist_for_choice, compute_paper_baseline_exact_hist_for_code,
    compute_paper_baseline_exact_hist_with_table, compute_pivot_scan_experiment_for_choice,
    compute_pivot_scan_experiment_for_code,
    compute_pivot_scan_experiment_from_native_rows_by_pivot,
    compute_safe_pivot_experiment_for_choice, compute_safe_pivot_experiment_for_code,
    compute_safe_pivot_experiment_from_native_rows_by_pivot,
    explain_ours_single_shot_target_for_code, explain_ours_single_shot_target_from_native_rows,
    infer_measurement_choice_from_csv_path, infer_measurement_choice_from_native_rows,
    native_rows_for_code_all_pivots, native_rows_for_code_with_pivot, parse_native_rows_from_csv,
    transition_local, BestSinglePivotSummaryReport, ExactHistogramReport,
    OursSingleShotTargetExplain, PivotScanExperimentReport, PivotScanPivotSummary,
    PivotScanSummaryReport, SafePivotCertificationReport, SafePivotExperimentReport,
    SafePivotSummaryReport, SafePivotTransformationReport, SingleShotPathNode,
    SingleShotPathWitness, SingleShotSourceRule, SingleShotSourceWitness,
};

pub mod decomposition;
pub use decomposition::{CompleteMeasurementTable, MeasurementTableBuilder};

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use super::*;
    use native_measurement::NativeMeasurement;

    static MEASUREMENT_IMPLS: LazyLock<CompleteMeasurementTable> = LazyLock::new(|| {
        let mut builder =
            MeasurementTableBuilder::new(NativeMeasurement::all(), TWOGROSS_MEASUREMENT);
        builder.build();
        builder
            .complete()
            .expect("Generating a complete measurement table should succeed")
    });

    #[test]
    fn qubit_measurements_are_native() {
        for i in 0..24 {
            let p: PauliString = PauliString(1 << i);
            let meas_impl = MEASUREMENT_IMPLS.implementation(p);
            assert_eq!(0, meas_impl.rotations().len());
        }
    }
}
