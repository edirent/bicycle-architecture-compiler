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
    PaperBaseline11Q, SingleShot11QAlgorithm, infer_measurement_choice_from_native_rows,
    parse_native_rows_from_csv, resolve_csv_path,
};

fn find_arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
}

fn resolve_csv(args: &[String]) -> String {
    let explicit = find_arg_value(args, "--csv").or_else(|| std::env::var("BOSS_CSV_GROSS").ok());
    let resolved = resolve_csv_path(
        explicit.as_deref(),
        &["native_dictionary_gross.csv", "native.csv"],
    )
    .expect("failed to resolve CSV path for paper baseline benchmark");
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let csv = resolve_csv(&args);
    let pivot_index = resolve_pivot_index(&args);
    let algorithm = PaperBaseline11Q;

    let native_read = run_once(|| {
        let rows = parse_native_rows_from_csv(Path::new(&csv), pivot_index)
            .expect("failed to parse native rows for paper baseline benchmark");
        black_box(rows);
    });

    let rows_for_infer = parse_native_rows_from_csv(Path::new(&csv), pivot_index)
        .expect("failed to parse native rows for code inference");
    let inferred_choice = infer_measurement_choice_from_native_rows(&rows_for_infer);

    let exact_hist = run_once(|| {
        let hist = algorithm
            .compute_exact_hist(&csv, pivot_index)
            .expect("paper baseline exact histogram benchmark failed");
        black_box(hist);
    });

    let summary = run_once(|| {
        let summary = algorithm
            .compute_summary(&csv, pivot_index)
            .expect("paper baseline summary benchmark failed");
        black_box(summary);
    });

    println!("benchmark=paper_baseline_11q");
    println!("algorithm={}", algorithm.name());
    println!("csv={csv}");
    println!("pivot_index={pivot_index}");
    println!("inferred_choice={inferred_choice:?}");
    println!("native_read_ms={:.3}", format_ms(native_read));
    println!("exact_hist_ms={:.3}", format_ms(exact_hist));
    println!("summary_ms={:.3}", format_ms(summary));
}
