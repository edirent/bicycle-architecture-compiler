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

use std::{
    env, error,
    fs::File,
    io,
    path::{Path, PathBuf},
};

use bicycle_cliffords::{
    CompleteMeasurementTable, MeasurementChoices, MeasurementTableBuilder, PaperBaseline11Q,
    SingleShot11QAlgorithm, infer_measurement_choice_from_csv_path,
    infer_measurement_choice_from_native_rows, native_measurement::NativeMeasurement,
    parse_native_rows_from_csv,
};
use bicycle_compiler::language::{AnglePrecision, PbcOperation};

use io::Write;

use bicycle_compiler::{PathArchitecture, optimize};
use clap::{Parser, Subcommand};
use log::{debug, info};
use serde_json::Deserializer;

#[derive(Parser)]
#[command(version, about, long_about=None)]
struct Cli {
    /// Select the bicycle code (either gross or two-gross)
    code: Option<MeasurementChoices>,
    #[command(subcommand)]
    commands: Option<Commands>,
    /// Read a cached Clifford synthesis table from the given file name
    #[arg(long)]
    measurement_table: Option<String>,
    /// Emit paper-style 11q beta histogram report and exit.
    #[arg(long)]
    paper_beta_report: bool,
    /// Optional dictionary CSV path; currently used to infer gross vs two-gross by file name.
    #[arg(long)]
    csv: Option<PathBuf>,
    /// Optional override for paper baseline histogram JSON output.
    #[arg(long)]
    paper_hist_out: Option<PathBuf>,
    /// Optional override for paper baseline summary JSON output.
    #[arg(long)]
    paper_summary_out: Option<PathBuf>,
    /// The accuracy of small angle synthesis
    #[arg(short, long, default_value_t = AnglePrecision::lit("1e-9"))]
    accuracy: AnglePrecision,
}

/// Caching commands
#[derive(Subcommand, Clone, PartialEq, Eq)]
enum Commands {
    /// Generate Clifford measurement table and save to file name
    Generate {
        /// The file name to output to
        measurement_table: String,
    },
}

fn code_tag(choice: MeasurementChoices) -> &'static str {
    match choice {
        MeasurementChoices::Gross => "gross",
        MeasurementChoices::TwoGross => "two_gross",
    }
}

fn ensure_parent_dir(path: &Path) -> Result<(), Box<dyn error::Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn emit_paper_beta_report(
    choice: MeasurementChoices,
    csv_hint: &str,
    hist_out: Option<&Path>,
    summary_out: Option<&Path>,
) -> Result<(), Box<dyn error::Error>> {
    let tag = code_tag(choice);
    let default_hist = PathBuf::from(format!(
        "results/histograms/paper_baseline_{tag}_exact_hist.json"
    ));
    let default_summary =
        PathBuf::from(format!("results/reports/paper_baseline_{tag}_summary.json"));
    let hist_path = hist_out.unwrap_or(default_hist.as_path());
    let summary_path = summary_out.unwrap_or(default_summary.as_path());

    ensure_parent_dir(hist_path)?;
    ensure_parent_dir(summary_path)?;

    let algorithm = PaperBaseline11Q;
    let histogram = algorithm.compute_exact_hist(csv_hint, 0)?;
    let summary = algorithm.compute_summary(csv_hint, 0)?;

    let hist_json = serde_json::to_string_pretty(&histogram)?;
    std::fs::write(hist_path, hist_json)?;
    let summary_json = serde_json::to_string_pretty(&summary)?;
    std::fs::write(summary_path, summary_json)?;

    println!("Wrote paper baseline exact hist: {}", hist_path.display());
    println!(
        "Wrote paper baseline summary:    {}",
        summary_path.display()
    );
    println!(
        "paper_baseline ({choice}): targets={}, reachable={}, mean={:?}, median={:?}, support={:?}",
        summary.total_targets,
        summary.reachable_targets,
        summary.mean,
        summary.median,
        summary.support
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn error::Error>> {
    // By default log INFO.
    if env::var("RUST_LOG").is_err() {
        // TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { env::set_var("RUST_LOG", "info") };
    }
    env_logger::init();

    let cli = Cli::parse();

    if cli.paper_beta_report {
        let inferred_from_rows = if let Some(path) = cli.csv.as_ref().filter(|path| path.exists()) {
            let rows = parse_native_rows_from_csv(path, 0)?;
            infer_measurement_choice_from_native_rows(&rows)
        } else {
            None
        };
        let inferred = cli
            .csv
            .as_ref()
            .and_then(|path| infer_measurement_choice_from_csv_path(path.as_path()));
        let choice = cli
            .code
            .or(inferred_from_rows)
            .or(inferred)
            .unwrap_or(MeasurementChoices::Gross);
        let csv_hint = match cli.csv.as_ref() {
            Some(path) => path.to_string_lossy().into_owned(),
            None => format!("{}_hint.csv", code_tag(choice)),
        };
        return emit_paper_beta_report(
            choice,
            &csv_hint,
            cli.paper_hist_out.as_deref(),
            cli.paper_summary_out.as_deref(),
        );
    }

    let code = cli
        .code
        .ok_or("missing required <code>; expected 'gross' or 'two-gross'")?;

    if let Some(Commands::Generate {
        measurement_table: cache_str,
    }) = cli.commands
    {
        info!("Generating measurement table.");
        let cache_path = Path::new(&cache_str);

        // Ensure that we can write a file in the desired output directory.  To do this we
        // write and delte an empty file in the parent directory of the full path of the
        // (output) cache file.  We do this in order to fail early rather than computing the
        // measurement table, only to find at the end that we cannot write the result.
        match cache_path.parent() {
            Some(cache_dir) => {
                let temp_filename = "dummy_file_check";
                let mut temp_file_path = PathBuf::from(cache_dir);
                temp_file_path.push(temp_filename);
                match File::create(&temp_file_path) {
                    Ok(_) => {
                        // Successfully created dummy file. Remove file.
                        std::fs::remove_file(temp_file_path)?;
                    }
                    Err(e) => {
                        eprintln!(
                            "Cannot create measurement_table output file in the target directory: {e}"
                        );
                        std::process::exit(1);
                    }
                }
            }
            None => {
                eprintln!("No parent directory found for {cache_str}");
                std::process::exit(1);
            }
        }

        // Create a builder and build the measurement table.
        let mut builder =
            MeasurementTableBuilder::new(NativeMeasurement::all(), code.measurement());
        builder.build();
        let measurement_table = builder.complete()?;

        // Serialize the measurement table and write to the cache file.
        let serialized =
            bitcode::serialize(&measurement_table).expect("The table should be serializable");
        info!("Done generating measurement table, writing.");
        let f = File::create(cache_path);
        match f {
            Ok(mut f) => {
                f.write_all(&serialized)
                    .expect("The serialized table should be writable to the cache");
            }
            Err(e) => {
                eprintln!(
                    "Cannot create  measurement_table output file in the target directory: {e}"
                );
                std::process::exit(1);
            }
        }
        info!("Done writing measurement table, exiting.");
        std::process::exit(0);
    }

    // Generate measurement table, from cache if given or otherwise from scratch
    let measurement_table = if let Some(cache_str) = cli.measurement_table {
        let cache_path = Path::new(&cache_str);
        let read =
            std::fs::read(cache_path).expect("The measurement table file should be readable");
        bitcode::deserialize::<CompleteMeasurementTable>(&read)?
    } else {
        let mut builder =
            MeasurementTableBuilder::new(NativeMeasurement::all(), code.measurement());
        builder.build();
        builder.complete()?
    };

    let reader = io::stdin().lock();

    // Support some streaming input from Stdin
    // The following works for (a weird version of) JSON:
    let de = Deserializer::from_reader(reader);
    let ops = de.into_iter::<PbcOperation>().map(|op| op.unwrap());
    let mut ops = ops.peekable();

    // Set the architecture based on the first operation
    let first_op = ops.peek();
    let architecture = if let Some(op) = first_op {
        PathArchitecture::for_qubits(op.basis().len())
    } else {
        // No ops, may as well terminate now.
        return Ok(());
    };

    let compiled = ops.map(|op| op.compile(&architecture, &measurement_table, cli.accuracy));

    let optimized_auts = compiled.map(optimize::remove_trivial_automorphisms);
    let mut optimized_chunked_ops = optimize::remove_duplicate_measurements_chunked(optimized_auts);
    let mut stdout = io::stdout();
    // Stop on first error
    let err: Result<(), io::Error> = optimized_chunked_ops.try_for_each(|chunk| {
        let out = serde_json::to_string(&chunk)?;
        writeln!(stdout, "{out}")
    });
    debug!("Encountered error while writing to stdout: {err:?}");

    Ok(())
}
