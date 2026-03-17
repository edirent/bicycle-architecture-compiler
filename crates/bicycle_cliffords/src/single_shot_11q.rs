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
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{
    ExactHistogramReport, Head, MeasurementChoices, compute_ours_single_shot_exact_hist_for_choice,
    compute_ours_single_shot_exact_hist_from_native_rows,
    compute_paper_baseline_exact_hist_for_choice, infer_measurement_choice_from_csv_path,
    infer_measurement_choice_from_native_rows, parse_native_rows_from_csv, transition_local,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SingleShotSummary {
    pub total_targets: u64,
    pub reachable_targets: u64,
    pub unreachable_targets: u64,
    pub support: Vec<u32>,
    pub median: Option<u32>,
    pub mean: Option<f64>,
    pub local_measurement_count_hist: Option<BTreeMap<u8, u64>>,
}

impl SingleShotSummary {
    fn from_exact_report(
        report: &ExactHistogramReport,
        local_hist: Option<BTreeMap<u8, u64>>,
    ) -> Self {
        Self {
            total_targets: report.total_targets,
            reachable_targets: report.reachable_targets,
            unreachable_targets: report.unreachable_targets,
            support: report.support.clone(),
            median: report.median,
            mean: report.mean,
            local_measurement_count_hist: local_hist,
        }
    }
}

pub trait SingleShot11QAlgorithm {
    fn name(&self) -> &'static str;
    fn compute_exact_hist(&self, csv_path: &str, pivot_index: usize) -> Result<BTreeMap<u32, u64>>;
    fn compute_summary(&self, csv_path: &str, pivot_index: usize) -> Result<SingleShotSummary>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PaperBaseline11Q;

#[derive(Debug, Clone, Copy, Default)]
pub struct OursSingleShot11Q;

impl PaperBaseline11Q {
    fn compute_report(&self, csv_path: &str, pivot_index: usize) -> Result<ExactHistogramReport> {
        let native_rows = load_native_rows_if_present(csv_path, pivot_index)?;
        let choice = infer_choice(csv_path, native_rows.as_deref());
        Ok(compute_paper_baseline_exact_hist_for_choice(choice))
    }
}

impl OursSingleShot11Q {
    fn compute_report(&self, csv_path: &str, pivot_index: usize) -> Result<ExactHistogramReport> {
        let native_rows = load_native_rows_if_present(csv_path, pivot_index)?;
        let report = if let Some(rows) = native_rows.as_ref() {
            compute_ours_single_shot_exact_hist_from_native_rows(rows)
        } else {
            let choice = infer_choice(csv_path, None);
            compute_ours_single_shot_exact_hist_for_choice(choice)
        };
        Ok(report)
    }
}

impl SingleShot11QAlgorithm for PaperBaseline11Q {
    fn name(&self) -> &'static str {
        "paper-baseline-11q"
    }

    fn compute_exact_hist(&self, csv_path: &str, pivot_index: usize) -> Result<BTreeMap<u32, u64>> {
        let report = self.compute_report(csv_path, pivot_index)?;
        Ok(report.histogram)
    }

    fn compute_summary(&self, csv_path: &str, pivot_index: usize) -> Result<SingleShotSummary> {
        let report = self.compute_report(csv_path, pivot_index)?;
        Ok(SingleShotSummary::from_exact_report(&report, None))
    }
}

impl SingleShot11QAlgorithm for OursSingleShot11Q {
    fn name(&self) -> &'static str {
        "ours-single-shot-11q"
    }

    fn compute_exact_hist(&self, csv_path: &str, pivot_index: usize) -> Result<BTreeMap<u32, u64>> {
        let report = self.compute_report(csv_path, pivot_index)?;
        Ok(report.histogram)
    }

    fn compute_summary(&self, csv_path: &str, pivot_index: usize) -> Result<SingleShotSummary> {
        let report = self.compute_report(csv_path, pivot_index)?;
        Ok(SingleShotSummary::from_exact_report(
            &report,
            Some(local_measurement_count_histogram()),
        ))
    }
}

pub fn local_measurement_count_histogram() -> BTreeMap<u8, u64> {
    let mut hist = BTreeMap::new();
    for prev in [Head::I, Head::X, Head::Y, Head::Z] {
        for basis in [Head::X, Head::Y, Head::Z] {
            let (_, delta) = transition_local(prev, basis);
            *hist.entry(delta).or_insert(0) += 1;
        }
    }
    hist
}

fn infer_choice(csv_path: &str, native_rows: Option<&[crate::NativeRow]>) -> MeasurementChoices {
    if let Some(rows) = native_rows {
        if let Some(choice) = infer_measurement_choice_from_native_rows(rows) {
            return choice;
        }
    }

    let trimmed = csv_path.trim();
    if trimmed.is_empty() {
        return MeasurementChoices::Gross;
    }

    infer_measurement_choice_from_csv_path(Path::new(trimmed)).unwrap_or(MeasurementChoices::Gross)
}

fn load_native_rows_if_present(
    csv_path: &str,
    pivot_index: usize,
) -> Result<Option<Vec<crate::NativeRow>>> {
    let trimmed = csv_path.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let path = Path::new(trimmed);
    if !path.exists() {
        return Ok(None);
    }
    let rows = parse_native_rows_from_csv(path, pivot_index).map_err(anyhow::Error::msg)?;
    Ok(Some(rows))
}
