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

use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

use bicycle_cliffords::{
    MeasurementChoices, OursSingleShot11Q, SingleShot11QAlgorithm,
    compute_pivot_scan_experiment_for_choice,
    compute_pivot_scan_experiment_from_native_rows_by_pivot,
    infer_measurement_choice_from_csv_path, infer_measurement_choice_from_native_rows,
    parse_native_rows_from_csv, resolve_csv_path,
};

fn find_arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

fn resolve_csv(args: &[String]) -> String {
    let explicit = find_arg_value(args, "--csv").or_else(|| std::env::var("BOSS_CSV_GROSS").ok());
    let resolved = resolve_csv_path(
        explicit.as_deref(),
        &["native_dictionary_gross.csv", "native.csv"],
    )
    .expect("failed to resolve CSV path for ours single-shot benchmark");
    resolved.to_string_lossy().into_owned()
}

fn resolve_pivot_index(args: &[String]) -> usize {
    find_arg_value(args, "--pivot-index")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0)
}

fn run_once<F: FnMut()>(mut f: F) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

fn format_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn infer_choice(csv: &str, pivot_index: usize) -> MeasurementChoices {
    if Path::new(csv).exists() {
        let rows = parse_native_rows_from_csv(Path::new(csv), pivot_index)
            .expect("failed to parse native rows for choice inference");
        if let Some(choice) = infer_measurement_choice_from_native_rows(&rows) {
            return choice;
        }
    }
    infer_measurement_choice_from_csv_path(Path::new(csv)).unwrap_or(MeasurementChoices::Gross)
}

fn parse_rows_for_all_pivots(csv: &str) -> Result<Vec<Vec<bicycle_cliffords::NativeRow>>, String> {
    let mut rows_by_pivot = Vec::with_capacity(12);
    for pivot_index in 0..12usize {
        rows_by_pivot.push(parse_native_rows_from_csv(Path::new(csv), pivot_index)?);
    }
    Ok(rows_by_pivot)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let csv = resolve_csv(&args);
    let pivot_index = resolve_pivot_index(&args);
    let include_pivot_scan = has_flag(&args, "--include-pivot-scan");
    let algorithm = OursSingleShot11Q;

    let exact_hist = run_once(|| {
        let hist = algorithm
            .compute_exact_hist(&csv, pivot_index)
            .expect("ours single-shot exact histogram benchmark failed");
        black_box(hist);
    });

    let summary = run_once(|| {
        let summary = algorithm
            .compute_summary(&csv, pivot_index)
            .expect("ours single-shot summary benchmark failed");
        black_box(summary);
    });

    let mut pivot_scan_ms = None;
    if include_pivot_scan {
        let choice = infer_choice(&csv, pivot_index);
        let elapsed = run_once(|| {
            if Path::new(&csv).exists() {
                let rows_by_pivot = parse_rows_for_all_pivots(&csv)
                    .expect("failed to parse native rows for all pivots");
                let report = compute_pivot_scan_experiment_from_native_rows_by_pivot(
                    choice.measurement(),
                    &rows_by_pivot,
                    &[],
                )
                .expect("pivot scan benchmark failed (csv mode)");
                black_box(report.best_single_pivot_summary.best_single_pivot_mean);
            } else {
                let report = compute_pivot_scan_experiment_for_choice(choice, &[])
                    .expect("pivot scan benchmark failed (choice mode)");
                black_box(report.best_single_pivot_summary.best_single_pivot_mean);
            }
        });
        pivot_scan_ms = Some(elapsed);
    }

    println!("benchmark=ours_single_shot_11q");
    println!("algorithm={}", algorithm.name());
    println!("csv={csv}");
    println!("pivot_index={pivot_index}");
    println!("exact_hist_ms={:.3}", format_ms(exact_hist));
    println!("summary_ms={:.3}", format_ms(summary));
    println!("pivot_scan_enabled={include_pivot_scan}");
    if let Some(elapsed) = pivot_scan_ms {
        println!("pivot_scan_best_single_ms={:.3}", format_ms(elapsed));
    }
}
