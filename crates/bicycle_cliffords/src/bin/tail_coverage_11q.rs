use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use bicycle_cliffords::tail_coverage11q::beta_baseline::{BetaBaseline, BetaBaselineSource};
use bicycle_cliffords::tail_coverage11q::calibration::CalibrationConfig;
use bicycle_cliffords::tail_coverage11q::mic::{MicModeResult, SeedPolicy, run_mic_benchmark};
use bicycle_cliffords::tail_coverage11q::native::load_native_rows;
use bicycle_cliffords::tail_coverage11q::report::{
    RunMetadata, build_closure_vs_cumulative_summary, build_final_absolute_class_report,
    build_final_real_native_closure_report, write_mic_mode_outputs, write_required_outputs,
};
use bicycle_cliffords::tail_coverage11q::{
    CoverageExperiment, CoverageSemantics, ModeRuleToggles, RequestedMode, RuleToggles,
    run_exact_coverage_with_toggles,
};
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
enum CliMode {
    Strict,
    Provisional,
    Both,
}

impl From<CliMode> for RequestedMode {
    fn from(value: CliMode) -> Self {
        match value {
            CliMode::Strict => RequestedMode::Strict,
            CliMode::Provisional => RequestedMode::Provisional,
            CliMode::Both => RequestedMode::Both,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
enum CliExperiment {
    Cumulative,
    #[value(name = "closure_class")]
    ClosureClass,
    Mic,
    All,
}

impl From<CliExperiment> for CoverageExperiment {
    fn from(value: CliExperiment) -> Self {
        match value {
            CliExperiment::Cumulative => CoverageExperiment::Cumulative,
            CliExperiment::ClosureClass => CoverageExperiment::ClosureClass,
            CliExperiment::Mic => CoverageExperiment::Mic,
            CliExperiment::All => CoverageExperiment::All,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
enum CliSeedPolicy {
    #[value(name = "native_stage1")]
    NativeStage1,
    #[value(name = "threshold_beta")]
    ThresholdBeta,
}

impl From<CliSeedPolicy> for SeedPolicy {
    fn from(value: CliSeedPolicy) -> Self {
        match value {
            CliSeedPolicy::NativeStage1 => SeedPolicy::NativeStage1,
            CliSeedPolicy::ThresholdBeta => SeedPolicy::ThresholdBeta,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Exact 11-qubit benchmark runner for cumulative, closure_class, and MIC experiments."
)]
struct Cli {
    /// Native rows CSV path. If omitted, tries native.csv then native_dictionary.csv.
    #[arg(long)]
    input: Option<PathBuf>,

    /// Pivot index used when parsing rows with full x_bits/z_bits columns.
    #[arg(long, default_value_t = 0)]
    pivot_index: usize,

    /// Output directory for final artifacts.
    #[arg(long, default_value = "results/final")]
    output_dir: PathBuf,

    /// Run mode: strict, provisional, or both.
    #[arg(long, value_enum, default_value_t = CliMode::Both)]
    mode: CliMode,

    /// Experiment pipeline: cumulative, closure_class, mic, or all. Default on this branch: mic.
    #[arg(long, value_enum, default_value_t = CliExperiment::Mic)]
    experiment: CliExperiment,

    /// Optional CSV with columns cost,paper_count for Figure-9 overlay comparison.
    #[arg(long)]
    figure9_csv: Option<PathBuf>,

    /// Optional per-tail baseline table path for MIC. If omitted, baseline is computed exactly in-code.
    #[arg(long)]
    beta_table: Option<PathBuf>,

    /// Optional calibration JSON for MIC class->physical-count mapping.
    #[arg(long)]
    calibration_json: Option<PathBuf>,

    /// MIC seed policy.
    #[arg(long, value_enum, default_value_t = CliSeedPolicy::NativeStage1)]
    seed_policy: CliSeedPolicy,

    /// Optional threshold used when --seed-policy threshold_beta.
    #[arg(long)]
    seed_threshold: Option<u16>,

    /// Toggle strict core rule family for both strict and provisional runs.
    #[arg(long)]
    enable_strict_core: Option<bool>,

    /// Toggle provisional bare bootstrap (class 5) for both strict and provisional runs.
    #[arg(long)]
    enable_bootstrap_5: Option<bool>,

    /// Toggle provisional handwritten bootstrap (class 6) for both strict and provisional runs.
    #[arg(long)]
    enable_bootstrap_6: Option<bool>,
}

fn hist_covered_uncovered(hist: &BTreeMap<String, u64>) -> (u64, u64) {
    let uncovered = hist.get("uncovered").copied().unwrap_or(0);
    let total: u64 = hist.values().sum();
    (total.saturating_sub(uncovered), uncovered)
}

fn ensure_stage1_validations(run: &bicycle_cliffords::tail_coverage11q::CoverageRun) -> Result<()> {
    let v = run.stage1.validation;
    if !v.histogram_total_ok {
        bail!("stage-1 histogram total does not equal 4^11");
    }
    if !v.stage1_same_q_only_ok {
        bail!("stage-1 same-Q P2 validation failed");
    }
    if !v.direct_native_mask_consistency_ok {
        bail!("stage-1 direct native mask consistency validation failed");
    }
    if !v.generators_dedup_by_head_and_tail_ok {
        bail!("stage-1 generator dedup-by-(head,tail) validation failed");
    }
    Ok(())
}

fn ensure_closure_validations(
    label: &str,
    dist: &bicycle_cliffords::tail_coverage11q::hist::ClosureDistribution,
) -> Result<()> {
    let v = dist.validations;
    if !v.baseline_formula_ok {
        bail!("{label}: baseline formula validation failed");
    }
    if !v.baseline_total_ok {
        bail!("{label}: baseline total validation failed");
    }
    if !v.any_total_ok {
        bail!("{label}: any total validation failed");
    }
    if !v.headed_total_ok {
        bail!("{label}: headed total validation failed");
    }
    if !v.weight_accounting_ok {
        bail!("{label}: weight accounting validation failed");
    }
    if !v.any_class_support_ok {
        bail!("{label}: any-class support subset validation failed");
    }
    if !v.headed_class_support_ok {
        bail!("{label}: headed-class support subset validation failed");
    }
    if !v.strict_headed_support_ok {
        bail!("{label}: strict headed support subset validation failed");
    }
    if !v.no_cumulative_bins_present {
        bail!("{label}: bins > 6 detected (cumulative leakage)");
    }
    if !v.any_vs_headed_domination_ok {
        bail!("{label}: any-vs-headed domination validation failed");
    }
    if !v.stage1_same_q_only_ok
        || !v.direct_native_mask_consistency_ok
        || !v.generators_dedup_by_head_and_tail_ok
    {
        bail!("{label}: stage-1 inherited validations failed");
    }
    Ok(())
}

fn ensure_cumulative_validations(
    label: &str,
    dist: &bicycle_cliffords::tail_coverage11q::hist::CumulativeDistribution,
) -> Result<()> {
    let v = dist.validations;
    if !v.baseline_formula_ok {
        bail!("{label}: baseline formula validation failed");
    }
    if !v.baseline_total_ok {
        bail!("{label}: baseline total validation failed");
    }
    if !v.any_total_ok {
        bail!("{label}: any total validation failed");
    }
    if !v.headed_total_ok {
        bail!("{label}: headed total validation failed");
    }
    if !v.weight_accounting_ok {
        bail!("{label}: weight accounting validation failed");
    }
    if !v.any_vs_headed_domination_ok {
        bail!("{label}: any-vs-headed domination validation failed");
    }
    if !v.stage1_same_q_only_ok
        || !v.direct_native_mask_consistency_ok
        || !v.generators_dedup_by_head_and_tail_ok
    {
        bail!("{label}: stage-1 inherited validations failed");
    }
    Ok(())
}

fn resolve_mode_toggles(cli: &Cli) -> ModeRuleToggles {
    let mut strict = RuleToggles::strict_defaults();
    let mut provisional = RuleToggles::provisional_defaults();

    if let Some(enabled) = cli.enable_strict_core {
        strict.enable_strict_core = enabled;
        provisional.enable_strict_core = enabled;
    }
    if let Some(enabled) = cli.enable_bootstrap_5 {
        strict.enable_bootstrap_5 = enabled;
        provisional.enable_bootstrap_5 = enabled;
    }
    if let Some(enabled) = cli.enable_bootstrap_6 {
        strict.enable_bootstrap_6 = enabled;
        provisional.enable_bootstrap_6 = enabled;
    }

    ModeRuleToggles {
        strict,
        provisional,
    }
}

fn ensure_mic_mode_validations(label: &str, mode_result: &MicModeResult) -> Result<()> {
    let mic_total: u64 = mode_result.mic_cost_histogram.values().sum();
    if mic_total != bicycle_cliffords::tail_coverage11q::pauli::SPACE_SIZE as u64 {
        bail!(
            "{label}: MIC histogram total mismatch (expected {}, got {})",
            bicycle_cliffords::tail_coverage11q::pauli::SPACE_SIZE,
            mic_total
        );
    }

    for i in 0..mode_result.baseline_costs.len() {
        let b = mode_result.baseline_costs[i];
        let r = mode_result.reuse_headed_costs[i];
        let m = mode_result.mic_costs[i];
        if m != b.min(r) {
            bail!("{label}: MIC formula violation at tail {i}");
        }
        if mode_result.improved[i] != (m < b) {
            bail!("{label}: improvement flag violation at tail {i}");
        }
    }

    Ok(())
}

fn semantics_from_experiment(experiment: CoverageExperiment) -> Option<CoverageSemantics> {
    match experiment {
        CoverageExperiment::Cumulative => Some(CoverageSemantics::Cumulative),
        CoverageExperiment::ClosureClass => Some(CoverageSemantics::Closure),
        CoverageExperiment::All => Some(CoverageSemantics::Both),
        CoverageExperiment::Mic => None,
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.pivot_index >= 12 {
        bail!("pivot-index must be in [0, 11], got {}", cli.pivot_index);
    }

    let (native_rows, resolved_input_path) =
        load_native_rows(cli.input.as_deref(), cli.pivot_index)?;
    let mode: RequestedMode = cli.mode.into();
    let experiment: CoverageExperiment = cli.experiment.into();
    let mode_toggles = resolve_mode_toggles(&cli);

    if let Some(semantics) = semantics_from_experiment(experiment) {
        let run = run_exact_coverage_with_toggles(&native_rows, mode, semantics, mode_toggles);
        ensure_stage1_validations(&run)?;

        if let Some(strict) = &run.closure_strict {
            ensure_closure_validations("strict closure", strict)?;
        }
        if let Some(provisional) = &run.closure_provisional {
            ensure_closure_validations("provisional closure", provisional)?;
        }
        if let Some(strict) = &run.cumulative_strict {
            ensure_cumulative_validations("strict cumulative", strict)?;
        }
        if let Some(provisional) = &run.cumulative_provisional {
            ensure_cumulative_validations("provisional cumulative", provisional)?;
        }

        let strict_any = run
            .closure_strict
            .as_ref()
            .map(|d| hist_covered_uncovered(&d.any_class_histogram));
        let strict_headed = run
            .closure_strict
            .as_ref()
            .map(|d| hist_covered_uncovered(&d.headed_class_histogram));
        let provisional_any = run
            .closure_provisional
            .as_ref()
            .map(|d| hist_covered_uncovered(&d.any_class_histogram));
        let provisional_headed = run
            .closure_provisional
            .as_ref()
            .map(|d| hist_covered_uncovered(&d.headed_class_histogram));

        let delta_weight_map = run.closure_delta.or(run.cumulative_delta).map(|delta| {
            let mut map = BTreeMap::new();
            for (w, value) in delta
                .provisional_covered_minus_strict_by_weight
                .iter()
                .enumerate()
            {
                map.insert(w.to_string(), *value);
            }
            map
        });

        let mut x_set = BTreeSet::new();
        let mut y_set = BTreeSet::new();
        let mut z_set = BTreeSet::new();
        for generator in &run.stage1.all_generators {
            match generator.head {
                bicycle_cliffords::tail_coverage11q::typed_dp::ActiveHead::X => {
                    x_set.insert(generator.tail_bits);
                }
                bicycle_cliffords::tail_coverage11q::typed_dp::ActiveHead::Y => {
                    y_set.insert(generator.tail_bits);
                }
                bicycle_cliffords::tail_coverage11q::typed_dp::ActiveHead::Z => {
                    z_set.insert(generator.tail_bits);
                }
                bicycle_cliffords::tail_coverage11q::typed_dp::ActiveHead::Bare => {}
            }
        }

        let metadata = RunMetadata {
            benchmark: "exact_11q_pauli_tail_typed_coverage".to_string(),
            mode_requested: mode.as_str().to_string(),
            semantics_requested: semantics.as_str().to_string(),
            kernel: "XYZ_native_headed".to_string(),
            space_size: bicycle_cliffords::tail_coverage11q::pauli::SPACE_SIZE as u64,
            input_native_csv: resolved_input_path.display().to_string(),
            pivot_index: cli.pivot_index,
            native_row_count: native_rows.len(),
            distinct_all_native_generators: run.stage1.all_generators.len(),
            distinct_x_kernel_generator_tails: x_set.len(),
            distinct_y_kernel_generator_tails: y_set.len(),
            distinct_z_kernel_generator_tails: z_set.len(),
            stage1_summary: run.stage1.summary,
            stage1_validation: run.stage1.validation,
            strict_any_total_covered: strict_any.map(|(c, _)| c),
            strict_any_total_uncovered: strict_any.map(|(_, u)| u),
            strict_headed_total_covered: strict_headed.map(|(c, _)| c),
            strict_headed_total_uncovered: strict_headed.map(|(_, u)| u),
            provisional_any_total_covered: provisional_any.map(|(c, _)| c),
            provisional_any_total_uncovered: provisional_any.map(|(_, u)| u),
            provisional_headed_total_covered: provisional_headed.map(|(c, _)| c),
            provisional_headed_total_uncovered: provisional_headed.map(|(_, u)| u),
            strict_provisional_monotonicity_checked: run.closure_delta.is_some()
                || run.cumulative_delta.is_some(),
            strict_provisional_monotonicity_ok: run
                .closure_delta
                .or(run.cumulative_delta)
                .map(|d| d.monotone_ok),
            provisional_total_covered_minus_strict: run
                .closure_delta
                .or(run.cumulative_delta)
                .map(|d| d.provisional_total_covered_minus_strict),
            provisional_covered_minus_strict_by_weight: delta_weight_map,
        };

        let markdown = build_final_absolute_class_report(
            run.stage1.summary,
            run.closure_strict.as_ref(),
            run.closure_provisional.as_ref(),
            run.cumulative_strict.as_ref(),
            run.cumulative_provisional.as_ref(),
            run.closure_delta,
            cli.figure9_csv.is_some(),
        );
        let closure_vs_cumulative_summary = build_closure_vs_cumulative_summary(
            run.closure_strict.as_ref(),
            run.closure_provisional.as_ref(),
            run.cumulative_strict.as_ref(),
            run.cumulative_provisional.as_ref(),
        );
        let final_real_native_report = build_final_real_native_closure_report(
            &metadata,
            run.closure_strict.as_ref(),
            run.closure_provisional.as_ref(),
            run.cumulative_strict.as_ref(),
            run.cumulative_provisional.as_ref(),
        );

        let output_paths = write_required_outputs(
            &cli.output_dir,
            run.closure_strict.as_ref(),
            run.closure_provisional.as_ref(),
            run.cumulative_strict.as_ref(),
            run.cumulative_provisional.as_ref(),
            &markdown,
            &closure_vs_cumulative_summary,
            &final_real_native_report,
            &metadata,
            cli.figure9_csv.as_deref(),
        )?;

        println!("coverage benchmark complete");
        println!("input CSV: {}", resolved_input_path.display());
        println!("native rows: {}", native_rows.len());
        println!("experiment: {}", experiment.as_str());
        println!("coverage files written under: {}", cli.output_dir.display());
        println!(
            "final report: {}",
            output_paths.final_absolute_class_report_md.display()
        );

        if let Some(figure9) = cli.figure9_csv.as_ref()
            && !output_paths.overlay_compare_with_figure9_csv.exists()
        {
            return Err(anyhow!(
                "figure9 CSV was provided ({}), but overlay file was not produced",
                figure9.display()
            ));
        }
    }

    if matches!(
        experiment,
        CoverageExperiment::Mic | CoverageExperiment::All
    ) {
        let (baseline, baseline_source) = BetaBaseline::load_or_compute(cli.beta_table.as_deref())?;
        let calibration = CalibrationConfig::load(cli.calibration_json.as_deref())?;
        let seed_policy: SeedPolicy = cli.seed_policy.into();

        let mic_run = run_mic_benchmark(
            &native_rows,
            mode,
            mode_toggles,
            baseline,
            baseline_source.clone(),
            calibration,
            seed_policy,
            cli.seed_threshold,
        )?;

        let baseline_source_label = match &baseline_source {
            BetaBaselineSource::Loaded(path) => format!("loaded from {}", path.display()),
            BetaBaselineSource::ComputedExact => {
                "computed exactly in-code from CompleteMeasurementTable(GROSS_MEASUREMENT)"
                    .to_string()
            }
        };

        match mode {
            RequestedMode::Strict => {
                let strict = mic_run
                    .strict
                    .as_ref()
                    .ok_or_else(|| anyhow!("missing strict MIC result"))?;
                ensure_mic_mode_validations("strict mic", strict)?;
                let paths = write_mic_mode_outputs(
                    &cli.output_dir,
                    strict,
                    &mic_run,
                    &baseline_source_label,
                    cli.figure9_csv.as_deref(),
                    None,
                )?;
                println!("MIC strict complete");
                println!("baseline source: {baseline_source_label}");
                println!("mic report: {}", paths.mic_vs_baseline_report_md.display());
            }
            RequestedMode::Provisional => {
                let provisional = mic_run
                    .provisional
                    .as_ref()
                    .ok_or_else(|| anyhow!("missing provisional MIC result"))?;
                ensure_mic_mode_validations("provisional mic", provisional)?;
                let paths = write_mic_mode_outputs(
                    &cli.output_dir,
                    provisional,
                    &mic_run,
                    &baseline_source_label,
                    cli.figure9_csv.as_deref(),
                    None,
                )?;
                println!("MIC provisional complete");
                println!("baseline source: {baseline_source_label}");
                println!("mic report: {}", paths.mic_vs_baseline_report_md.display());
            }
            RequestedMode::Both => {
                if let Some(strict) = mic_run.strict.as_ref() {
                    ensure_mic_mode_validations("strict mic", strict)?;
                    let paths = write_mic_mode_outputs(
                        &cli.output_dir,
                        strict,
                        &mic_run,
                        &baseline_source_label,
                        cli.figure9_csv.as_deref(),
                        Some("strict"),
                    )?;
                    println!(
                        "MIC strict report: {}",
                        paths.mic_vs_baseline_report_md.display()
                    );
                }
                if let Some(provisional) = mic_run.provisional.as_ref() {
                    ensure_mic_mode_validations("provisional mic", provisional)?;
                    let paths = write_mic_mode_outputs(
                        &cli.output_dir,
                        provisional,
                        &mic_run,
                        &baseline_source_label,
                        cli.figure9_csv.as_deref(),
                        Some("provisional"),
                    )?;
                    println!(
                        "MIC provisional report: {}",
                        paths.mic_vs_baseline_report_md.display()
                    );
                }
                println!("MIC both complete");
                println!("baseline source: {baseline_source_label}");
            }
        }
    }

    Ok(())
}
