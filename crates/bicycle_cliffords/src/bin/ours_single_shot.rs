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

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use bicycle_cliffords::{
    compute_ours_single_shot_exact_hist_for_choice,
    compute_ours_single_shot_exact_hist_from_native_rows, compute_pivot_scan_experiment_for_choice,
    compute_pivot_scan_experiment_from_native_rows_by_pivot,
    compute_safe_pivot_experiment_for_choice,
    compute_safe_pivot_experiment_from_native_rows_by_pivot,
    explain_ours_single_shot_target_for_code, explain_ours_single_shot_target_from_native_rows,
    infer_measurement_choice_from_csv_path, infer_measurement_choice_from_native_rows,
    parse_native_rows_from_csv, ExactHistogramReport, MeasurementChoices, NativeRow, Tail11,
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Optional dictionary CSV path. Used to infer code choice from file name.
    #[arg(long)]
    csv: Option<PathBuf>,
    /// Explicit code selection, overrides CSV-based inference.
    #[arg(long)]
    code: Option<MeasurementChoices>,
    /// Output prefix. Writes <out>_exact_hist.json and <out>_summary.json.
    #[arg(long, default_value = "ours_single_shot")]
    out: PathBuf,
    #[arg(long)]
    explain_tail: Option<String>,
    #[arg(long)]
    explain_out: Option<PathBuf>,
    /// Emit safe-pivot certification + fixed/safe histograms as separate JSON files.
    #[arg(long)]
    ours_safe_pivot_report: bool,
    /// Output JSON path for safe-pivot certification details.
    #[arg(long, default_value = "safe_pivot_certification.json")]
    safe_pivot_certification_out: PathBuf,
    /// Output JSON path for the fixed-pivot exact histogram.
    #[arg(long, default_value = "ours_fixed_pivot_exact_hist.json")]
    ours_fixed_pivot_hist_out: PathBuf,
    /// Output JSON path for the safe-pivot exact histogram.
    #[arg(long, default_value = "ours_safe_pivot_exact_hist.json")]
    ours_safe_pivot_hist_out: PathBuf,
    /// Output JSON path for safe-pivot summary stats.
    #[arg(long, default_value = "ours_safe_pivot_summary.json")]
    ours_safe_pivot_summary_out: PathBuf,
    /// Emit direct pivot recompilation scan and best-single-pivot reports.
    #[arg(long)]
    ours_pivot_scan_report: bool,
    /// Candidate pivots for empirical scan (1-based), comma-separated.
    #[arg(long, value_delimiter = ',')]
    pivot_candidates: Option<Vec<u8>>,
    /// Output JSON path for pivot scan summary.
    #[arg(long, default_value = "pivot_scan_summary.json")]
    pivot_scan_summary_out: PathBuf,
    /// Output JSON path for fixed-pivot histogram in pivot-scan mode.
    #[arg(long, default_value = "ours_single_shot_fixed_pivot_exact_hist.json")]
    ours_single_shot_fixed_pivot_hist_out: PathBuf,
    /// Output JSON path for best-single-pivot histogram.
    #[arg(
        long,
        default_value = "ours_single_shot_best_single_pivot_exact_hist.json"
    )]
    ours_single_shot_best_single_pivot_hist_out: PathBuf,
    /// Output JSON path for best-single-pivot summary.
    #[arg(
        long,
        default_value = "ours_single_shot_best_single_pivot_summary.json"
    )]
    ours_single_shot_best_single_pivot_summary_out: PathBuf,
    /// Prefix for per-pivot outputs: <prefix>_<j>_exact_hist.json and <prefix>_<j>_summary.json.
    #[arg(long, default_value = "ours_single_shot_pivot")]
    ours_single_shot_pivot_prefix: PathBuf,
}

fn with_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}{}", prefix.display(), suffix))
}

fn write_hist_only(path: &Path, report: &ExactHistogramReport) -> Result<(), Box<dyn Error>> {
    let text = serde_json::to_string_pretty(&report.histogram)?;
    fs::write(path, text)?;
    Ok(())
}

fn write_summary(path: &Path, report: &ExactHistogramReport) -> Result<(), Box<dyn Error>> {
    let text = serde_json::to_string_pretty(report)?;
    fs::write(path, text)?;
    Ok(())
}

fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    let text = serde_json::to_string_pretty(value)?;
    fs::write(path, text)?;
    Ok(())
}

fn resolve_choice(cli: &Cli) -> MeasurementChoices {
    if let Some(choice) = cli.code {
        return choice;
    }

    if let Some(csv) = cli.csv.as_ref() {
        if let Some(choice) = infer_measurement_choice_from_csv_path(csv) {
            return choice;
        }
    }

    MeasurementChoices::Gross
}

fn load_native_rows_from_csv_if_present(
    csv: Option<&Path>,
) -> Result<Option<Vec<NativeRow>>, Box<dyn Error>> {
    let Some(path) = csv else {
        return Ok(None);
    };
    if !path.exists() {
        return Ok(None);
    }
    let rows = parse_native_rows_from_csv(path, 0)?;
    Ok(Some(rows))
}

fn load_native_rows_for_all_pivots_from_csv_if_present(
    csv: Option<&Path>,
) -> Result<Option<Vec<Vec<NativeRow>>>, Box<dyn Error>> {
    let Some(path) = csv else {
        return Ok(None);
    };
    if !path.exists() {
        return Ok(None);
    }
    let mut rows_by_pivot = Vec::with_capacity(12);
    for pivot_index in 0..12usize {
        rows_by_pivot.push(parse_native_rows_from_csv(path, pivot_index)?);
    }
    Ok(Some(rows_by_pivot))
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let csv_rows = load_native_rows_from_csv_if_present(cli.csv.as_deref())?;
    let choice = if let Some(rows) = csv_rows.as_ref() {
        cli.code
            .or_else(|| infer_measurement_choice_from_native_rows(rows))
            .or_else(|| {
                cli.csv
                    .as_ref()
                    .and_then(|path| infer_measurement_choice_from_csv_path(path))
            })
            .unwrap_or(MeasurementChoices::Gross)
    } else {
        resolve_choice(&cli)
    };

    if cli.ours_safe_pivot_report && cli.ours_pivot_scan_report {
        return Err("choose only one experimental report mode: --ours-safe-pivot-report or --ours-pivot-scan-report".into());
    }

    if cli.ours_pivot_scan_report {
        let pivot_candidates = cli.pivot_candidates.clone().unwrap_or_default();
        let experiment = if let Some(rows_by_pivot) =
            load_native_rows_for_all_pivots_from_csv_if_present(cli.csv.as_deref())?
        {
            compute_pivot_scan_experiment_from_native_rows_by_pivot(
                choice.measurement(),
                &rows_by_pivot,
                &pivot_candidates,
            )?
        } else {
            compute_pivot_scan_experiment_for_choice(choice, &pivot_candidates)?
        };

        write_json(&cli.pivot_scan_summary_out, &experiment.pivot_scan_summary)?;
        write_hist_only(
            &cli.ours_single_shot_fixed_pivot_hist_out,
            &experiment.fixed_pivot_exact_hist,
        )?;
        write_hist_only(
            &cli.ours_single_shot_best_single_pivot_hist_out,
            &experiment.best_single_pivot_exact_hist,
        )?;
        write_json(
            &cli.ours_single_shot_best_single_pivot_summary_out,
            &experiment.best_single_pivot_summary,
        )?;

        for (pivot_qubit, hist) in &experiment.per_pivot_exact_hist {
            let hist_path = PathBuf::from(format!(
                "{}_{}_exact_hist.json",
                cli.ours_single_shot_pivot_prefix.display(),
                pivot_qubit
            ));
            write_hist_only(&hist_path, hist)?;

            let summary = experiment
                .pivot_scan_summary
                .pivots
                .iter()
                .find(|entry| entry.pivot_index == *pivot_qubit)
                .ok_or("missing per-pivot summary entry for valid pivot")?;
            let summary_path = PathBuf::from(format!(
                "{}_{}_summary.json",
                cli.ours_single_shot_pivot_prefix.display(),
                pivot_qubit
            ));
            write_json(&summary_path, summary)?;
        }

        println!(
            "Wrote pivot scan summary:          {}",
            cli.pivot_scan_summary_out.display()
        );
        println!(
            "Wrote fixed-pivot exact hist:      {}",
            cli.ours_single_shot_fixed_pivot_hist_out.display()
        );
        println!(
            "Wrote best-single exact hist:      {}",
            cli.ours_single_shot_best_single_pivot_hist_out.display()
        );
        println!(
            "Wrote best-single summary:         {}",
            cli.ours_single_shot_best_single_pivot_summary_out.display()
        );
        println!(
            "ours_pivot_scan ({choice}): J_emp={:?}, J_valid={:?}, fixed_mean={:?}, best_single_mean={:?}, best_pivot_by_mean={:?}",
            experiment.pivot_scan_summary.j_emp,
            experiment.pivot_scan_summary.j_valid,
            experiment.best_single_pivot_summary.fixed_pivot_mean,
            experiment.best_single_pivot_summary.best_single_pivot_mean,
            experiment.best_single_pivot_summary.best_pivot_by_mean
        );
        return Ok(());
    }

    if cli.ours_safe_pivot_report {
        let experiment = if let Some(rows_by_pivot) =
            load_native_rows_for_all_pivots_from_csv_if_present(cli.csv.as_deref())?
        {
            compute_safe_pivot_experiment_from_native_rows_by_pivot(
                choice.measurement(),
                &rows_by_pivot,
            )?
        } else {
            compute_safe_pivot_experiment_for_choice(choice)?
        };

        write_hist_only(&cli.ours_fixed_pivot_hist_out, &experiment.fixed_pivot)?;
        write_hist_only(&cli.ours_safe_pivot_hist_out, &experiment.safe_pivot)?;
        write_json(&cli.safe_pivot_certification_out, &experiment.certification)?;
        write_json(&cli.ours_safe_pivot_summary_out, &experiment.summary)?;

        println!(
            "Wrote fixed-pivot exact hist:      {}",
            cli.ours_fixed_pivot_hist_out.display()
        );
        println!(
            "Wrote safe-pivot exact hist:       {}",
            cli.ours_safe_pivot_hist_out.display()
        );
        println!(
            "Wrote safe-pivot certification:    {}",
            cli.safe_pivot_certification_out.display()
        );
        println!(
            "Wrote safe-pivot summary:          {}",
            cli.ours_safe_pivot_summary_out.display()
        );
        println!(
            "ours_safe_pivot ({choice}): J_safe={:?}, mean_fixed={:?}, mean_safe={:?}",
            experiment.summary.j_safe,
            experiment.summary.mean_fixed_pivot_cost,
            experiment.summary.mean_safe_pivot_cost
        );
        return Ok(());
    }

    let report = if let Some(rows) = csv_rows.as_ref() {
        compute_ours_single_shot_exact_hist_from_native_rows(rows)
    } else {
        compute_ours_single_shot_exact_hist_for_choice(choice)
    };
    let hist_path = with_suffix(&cli.out, "_exact_hist.json");
    let summary_path = with_suffix(&cli.out, "_summary.json");

    write_hist_only(&hist_path, &report)?;
    write_summary(&summary_path, &report)?;

    println!("Wrote ours single-shot exact hist: {}", hist_path.display());
    println!(
        "Wrote ours single-shot summary:    {}",
        summary_path.display()
    );
    println!(
        "ours_single_shot ({choice}): targets={}, reachable={}, mean={:?}, median={:?}, support={:?}",
        report.total_targets, report.reachable_targets, report.mean, report.median, report.support
    );

    if let Some(tail_label) = cli.explain_tail.as_deref() {
        let tail = Tail11::from_label(tail_label)?;
        let explain = if let Some(rows) = csv_rows.as_ref() {
            explain_ours_single_shot_target_from_native_rows(rows, tail)
        } else {
            explain_ours_single_shot_target_for_code(choice.measurement(), tail)
        }
        .ok_or("target tail is unreachable in ours_single_shot")?;
        let explain_json = serde_json::to_string_pretty(&explain)?;
        if let Some(path) = cli.explain_out.as_ref() {
            fs::write(path, explain_json)?;
            println!("Wrote target explanation: {}", path.display());
        } else {
            println!("{explain_json}");
        }
    }

    Ok(())
}
