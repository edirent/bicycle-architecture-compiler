use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use super::hist::{ClosureDistribution, CumulativeDistribution, ModeDelta};
use super::mic::{MicModeResult, MicRun, build_overlay_rows};
use super::pauli::SPACE_SIZE;
use super::stage1::{Stage1Summary, Stage1Validation};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    pub benchmark: String,
    pub mode_requested: String,
    pub semantics_requested: String,
    pub kernel: String,
    pub space_size: u64,
    pub input_native_csv: String,
    pub pivot_index: usize,
    pub native_row_count: usize,
    pub distinct_all_native_generators: usize,
    pub distinct_x_kernel_generator_tails: usize,
    pub distinct_y_kernel_generator_tails: usize,
    pub distinct_z_kernel_generator_tails: usize,
    pub stage1_summary: Stage1Summary,
    pub stage1_validation: Stage1Validation,

    pub strict_any_total_covered: Option<u64>,
    pub strict_any_total_uncovered: Option<u64>,
    pub strict_headed_total_covered: Option<u64>,
    pub strict_headed_total_uncovered: Option<u64>,

    pub provisional_any_total_covered: Option<u64>,
    pub provisional_any_total_uncovered: Option<u64>,
    pub provisional_headed_total_covered: Option<u64>,
    pub provisional_headed_total_uncovered: Option<u64>,

    pub strict_provisional_monotonicity_checked: bool,
    pub strict_provisional_monotonicity_ok: Option<bool>,
    pub provisional_total_covered_minus_strict: Option<i64>,
    pub provisional_covered_minus_strict_by_weight: Option<BTreeMap<String, i64>>,
}

#[derive(Debug, Clone)]
pub struct OutputPaths {
    pub strict_any_class_histogram_json: PathBuf,
    pub provisional_any_class_histogram_json: PathBuf,
    pub strict_headed_class_histogram_json: PathBuf,
    pub provisional_headed_class_histogram_json: PathBuf,

    pub strict_any_depth_histogram_json: PathBuf,
    pub provisional_any_depth_histogram_json: PathBuf,
    pub strict_headed_depth_histogram_json: PathBuf,
    pub provisional_headed_depth_histogram_json: PathBuf,

    pub strict_first_origin_histogram_json: PathBuf,
    pub strict_best_class_origin_histogram_json: PathBuf,
    pub provisional_first_origin_histogram_json: PathBuf,
    pub provisional_best_class_origin_histogram_json: PathBuf,

    pub strict_headed_weight_by_class_json: PathBuf,
    pub provisional_headed_weight_by_class_json: PathBuf,

    pub strict_any_cost_histogram_json: PathBuf,
    pub provisional_any_cost_histogram_json: PathBuf,
    pub strict_headed_cost_histogram_json: PathBuf,
    pub provisional_headed_cost_histogram_json: PathBuf,

    pub final_absolute_class_report_md: PathBuf,
    pub closure_vs_cumulative_summary_md: PathBuf,
    pub final_real_native_closure_report_md: PathBuf,
    pub run_metadata_json: PathBuf,
    pub overlay_compare_with_figure9_csv: PathBuf,
}

impl OutputPaths {
    pub fn in_dir(root: &Path) -> Self {
        Self {
            strict_any_class_histogram_json: root.join("strict_any_class_histogram.json"),
            provisional_any_class_histogram_json: root.join("provisional_any_class_histogram.json"),
            strict_headed_class_histogram_json: root.join("strict_headed_class_histogram.json"),
            provisional_headed_class_histogram_json: root
                .join("provisional_headed_class_histogram.json"),

            strict_any_depth_histogram_json: root.join("strict_any_depth_histogram.json"),
            provisional_any_depth_histogram_json: root.join("provisional_any_depth_histogram.json"),
            strict_headed_depth_histogram_json: root.join("strict_headed_depth_histogram.json"),
            provisional_headed_depth_histogram_json: root
                .join("provisional_headed_depth_histogram.json"),

            strict_first_origin_histogram_json: root.join("strict_first_origin_histogram.json"),
            strict_best_class_origin_histogram_json: root
                .join("strict_best_class_origin_histogram.json"),
            provisional_first_origin_histogram_json: root
                .join("provisional_first_origin_histogram.json"),
            provisional_best_class_origin_histogram_json: root
                .join("provisional_best_class_origin_histogram.json"),

            strict_headed_weight_by_class_json: root.join("strict_headed_weight_by_class.json"),
            provisional_headed_weight_by_class_json: root
                .join("provisional_headed_weight_by_class.json"),

            strict_any_cost_histogram_json: root.join("strict_any_cost_histogram.json"),
            provisional_any_cost_histogram_json: root.join("provisional_any_cost_histogram.json"),
            strict_headed_cost_histogram_json: root.join("strict_headed_cost_histogram.json"),
            provisional_headed_cost_histogram_json: root
                .join("provisional_headed_cost_histogram.json"),

            final_absolute_class_report_md: root.join("final_absolute_class_report.md"),
            closure_vs_cumulative_summary_md: root.join("closure_vs_cumulative_summary.md"),
            final_real_native_closure_report_md: root.join("final_real_native_closure_report.md"),
            run_metadata_json: root.join("run_metadata.json"),
            overlay_compare_with_figure9_csv: root.join("overlay_compare_with_figure9.csv"),
        }
    }
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    ensure_parent(path)?;
    fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn format_histogram(hist: &BTreeMap<String, u64>) -> String {
    let mut numeric = hist
        .iter()
        .filter_map(|(k, v)| k.parse::<u32>().ok().map(|cost| (cost, *v)))
        .collect::<Vec<_>>();
    numeric.sort_unstable_by_key(|(cost, _)| *cost);

    let mut parts = numeric
        .into_iter()
        .map(|(cost, count)| format!("{cost}: {count}"))
        .collect::<Vec<_>>();

    if let Some(uncovered) = hist.get("uncovered") {
        parts.push(format!("uncovered: {uncovered}"));
    }
    parts.join(", ")
}

fn format_named_histogram(hist: &BTreeMap<String, u64>) -> String {
    hist.iter()
        .map(|(k, v)| format!("{k}: {v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn max_numeric_key(hist: &BTreeMap<String, u64>) -> Option<u32> {
    hist.keys().filter_map(|k| k.parse::<u32>().ok()).max()
}

fn format_weight_by_class_lines(
    matrix: &BTreeMap<String, BTreeMap<String, u64>>,
    class_keys: &[&str],
) -> Vec<String> {
    let mut out = Vec::new();
    let mut weights = matrix
        .keys()
        .filter_map(|k| k.parse::<usize>().ok())
        .collect::<Vec<_>>();
    weights.sort_unstable();

    for w in weights {
        let row = matrix
            .get(&w.to_string())
            .cloned()
            .unwrap_or_else(BTreeMap::new);
        let baseline: u64 = row.values().sum();
        let mut abs_parts = Vec::new();
        let mut frac_parts = Vec::new();
        for class_key in class_keys {
            let count = row.get(*class_key).copied().unwrap_or(0);
            abs_parts.push(format!("{class_key}={count}"));
            let frac = if baseline == 0 {
                0.0
            } else {
                count as f64 / baseline as f64
            };
            frac_parts.push(format!("{class_key}={frac:.6}"));
        }
        out.push(format!(
            "- w={w}: [{}]; fractions [{}]",
            abs_parts.join(", "),
            frac_parts.join(", ")
        ));
    }
    out
}

fn format_depth_stats_line(label: &str, depth_hist: &BTreeMap<String, u64>) -> String {
    let mut values = Vec::<u16>::new();
    for (k, v) in depth_hist {
        if k == "uncovered" {
            continue;
        }
        if let Ok(depth) = k.parse::<u16>() {
            for _ in 0..*v {
                values.push(depth);
            }
        }
    }
    values.sort_unstable();
    if values.is_empty() {
        return format!("- {label}: no covered states");
    }
    let count = values.len();
    let sum: u64 = values.iter().map(|&d| d as u64).sum();
    let mean = sum as f64 / count as f64;
    let median = values[(count - 1) / 2];
    let p90_idx = ((count as f64 * 0.9).ceil() as usize)
        .saturating_sub(1)
        .min(count - 1);
    let p90 = values[p90_idx];
    format!("- {label}: mean={mean:.6}, median={median}, p90={p90}")
}

fn parse_figure9_csv(path: &Path) -> Result<BTreeMap<u32, u64>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed reading Figure-9 CSV '{}'", path.display()))?;

    let mut lines = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'));

    let header = lines
        .next()
        .ok_or_else(|| anyhow!("figure9 CSV is empty: {}", path.display()))?;
    let columns = header
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .collect::<Vec<_>>();

    let cost_idx = columns
        .iter()
        .position(|c| c == "cost")
        .ok_or_else(|| anyhow!("figure9 CSV missing 'cost' column"))?;
    let count_idx = columns
        .iter()
        .position(|c| c == "paper_count")
        .ok_or_else(|| anyhow!("figure9 CSV missing 'paper_count' column"))?;

    let mut out = BTreeMap::new();
    for (line_no, line) in lines.enumerate() {
        let fields = line.split(',').map(|s| s.trim()).collect::<Vec<_>>();
        let cost = fields
            .get(cost_idx)
            .ok_or_else(|| anyhow!("missing cost on figure9 line {}", line_no + 2))?
            .parse::<u32>()
            .with_context(|| format!("invalid cost on figure9 line {}", line_no + 2))?;
        let count = fields
            .get(count_idx)
            .ok_or_else(|| anyhow!("missing paper_count on figure9 line {}", line_no + 2))?
            .parse::<u64>()
            .with_context(|| format!("invalid paper_count on figure9 line {}", line_no + 2))?;
        out.insert(cost, count);
    }
    Ok(out)
}

fn write_overlay_csv(
    path: &Path,
    figure9_csv: &Path,
    ours_hist_any: &BTreeMap<String, u64>,
) -> Result<()> {
    let paper = parse_figure9_csv(figure9_csv)?;

    let ours = ours_hist_any
        .iter()
        .filter_map(|(k, v)| k.parse::<u32>().ok().map(|cost| (cost, *v)))
        .collect::<BTreeMap<_, _>>();

    let mut costs = paper.keys().copied().collect::<Vec<_>>();
    costs.extend(ours.keys().copied());
    costs.sort_unstable();
    costs.dedup();

    let mut lines = Vec::new();
    lines.push("cost,paper_count,closure_class_count".to_string());
    for cost in costs {
        lines.push(format!(
            "{cost},{},{}",
            paper.get(&cost).copied().unwrap_or(0),
            ours.get(&cost).copied().unwrap_or(0)
        ));
    }

    ensure_parent(path)?;
    fs::write(path, lines.join("\n"))?;
    Ok(())
}

fn write_closure_histograms(
    paths: &OutputPaths,
    strict: Option<&ClosureDistribution>,
    provisional: Option<&ClosureDistribution>,
) -> Result<()> {
    if let Some(strict) = strict {
        write_json(
            &paths.strict_any_class_histogram_json,
            &strict.any_class_histogram,
        )?;
        write_json(
            &paths.strict_headed_class_histogram_json,
            &strict.headed_class_histogram,
        )?;
        write_json(
            &paths.strict_any_depth_histogram_json,
            &strict.any_depth_histogram,
        )?;
        write_json(
            &paths.strict_headed_depth_histogram_json,
            &strict.headed_depth_histogram,
        )?;
        write_json(
            &paths.strict_first_origin_histogram_json,
            &strict.first_origin_histogram,
        )?;
        write_json(
            &paths.strict_best_class_origin_histogram_json,
            &strict.best_class_origin_histogram,
        )?;
        write_json(
            &paths.strict_headed_weight_by_class_json,
            &strict.headed_weight_by_class,
        )?;
    }

    if let Some(provisional) = provisional {
        write_json(
            &paths.provisional_any_class_histogram_json,
            &provisional.any_class_histogram,
        )?;
        write_json(
            &paths.provisional_headed_class_histogram_json,
            &provisional.headed_class_histogram,
        )?;
        write_json(
            &paths.provisional_any_depth_histogram_json,
            &provisional.any_depth_histogram,
        )?;
        write_json(
            &paths.provisional_headed_depth_histogram_json,
            &provisional.headed_depth_histogram,
        )?;
        write_json(
            &paths.provisional_first_origin_histogram_json,
            &provisional.first_origin_histogram,
        )?;
        write_json(
            &paths.provisional_best_class_origin_histogram_json,
            &provisional.best_class_origin_histogram,
        )?;
        write_json(
            &paths.provisional_headed_weight_by_class_json,
            &provisional.headed_weight_by_class,
        )?;
    }

    Ok(())
}

fn write_cumulative_histograms(
    paths: &OutputPaths,
    strict: Option<&CumulativeDistribution>,
    provisional: Option<&CumulativeDistribution>,
) -> Result<()> {
    if let Some(strict) = strict {
        write_json(
            &paths.strict_any_cost_histogram_json,
            &strict.any_cost_histogram,
        )?;
        write_json(
            &paths.strict_headed_cost_histogram_json,
            &strict.headed_cost_histogram,
        )?;
    }

    if let Some(provisional) = provisional {
        write_json(
            &paths.provisional_any_cost_histogram_json,
            &provisional.any_cost_histogram,
        )?;
        write_json(
            &paths.provisional_headed_cost_histogram_json,
            &provisional.headed_cost_histogram,
        )?;
    }

    Ok(())
}

pub fn build_final_absolute_class_report(
    stage1_summary: Stage1Summary,
    closure_strict: Option<&ClosureDistribution>,
    closure_provisional: Option<&ClosureDistribution>,
    cumulative_strict: Option<&CumulativeDistribution>,
    cumulative_provisional: Option<&CumulativeDistribution>,
    closure_delta: Option<ModeDelta>,
    figure9_csv_supplied: bool,
) -> String {
    let mut md = String::new();
    md.push_str("# 11Q Tail Coverage Closure-Class Report\n\n");

    md.push_str("This is a closure-class benchmark.\n\n");
    md.push_str("- Costs 4 / 5 / 6 are absolute classes, not cumulative path lengths.\n");
    md.push_str("- Closure depth is reported separately and is NOT a cost.\n");
    md.push_str("- Closure reachability is computed as a least fixed point (saturation).\n\n");

    md.push_str("## Stage-1 Summary\n\n");
    md.push_str(&format!("- class 1: {}\n", stage1_summary.class_1));
    md.push_str(&format!("- class 2: {}\n", stage1_summary.class_2));
    md.push_str(&format!("- class 4: {}\n", stage1_summary.class_4));
    md.push_str(&format!("- uncovered: {}\n", stage1_summary.uncovered));
    md.push_str(&format!(
        "- covered fraction: {:.8}\n\n",
        stage1_summary.covered_fraction
    ));

    md.push_str("## Closure Histograms\n\n");
    if let Some(strict) = closure_strict {
        md.push_str(&format!(
            "- strict any class histogram: {}\n",
            format_histogram(&strict.any_class_histogram)
        ));
        md.push_str(&format!(
            "- strict headed class histogram: {}\n",
            format_histogram(&strict.headed_class_histogram)
        ));
    } else {
        md.push_str("- strict mode not run\n");
    }

    if let Some(provisional) = closure_provisional {
        md.push_str(&format!(
            "- provisional any class histogram: {}\n",
            format_histogram(&provisional.any_class_histogram)
        ));
        md.push_str(&format!(
            "- provisional headed class histogram: {}\n",
            format_histogram(&provisional.headed_class_histogram)
        ));
    } else {
        md.push_str("- provisional mode not run\n");
    }

    if let Some(delta) = closure_delta {
        md.push_str(&format!(
            "- provisional covered minus strict covered (any collapse): {}\n",
            delta.provisional_total_covered_minus_strict
        ));
    }
    md.push('\n');

    md.push_str("## Legacy Cumulative Comparison\n\n");
    if let Some(strict) = cumulative_strict {
        md.push_str(&format!(
            "- strict cumulative any histogram: {}\n",
            format_histogram(&strict.any_cost_histogram)
        ));
    }
    if let Some(provisional) = cumulative_provisional {
        md.push_str(&format!(
            "- provisional cumulative any histogram: {}\n",
            format_histogram(&provisional.any_cost_histogram)
        ));
    }
    if cumulative_strict.is_none() && cumulative_provisional.is_none() {
        md.push_str("- cumulative semantics not run in this invocation\n");
    }

    md.push_str("\n## Figure-9 Overlay Note\n\n");
    if figure9_csv_supplied {
        md.push_str("- Figure-9 CSV overlay was requested.\n");
        md.push_str("- Overlay uses `closure_class` labels and is not the paper's generic-synthesis axis.\n");
    } else {
        md.push_str("- Figure-9 CSV overlay not requested.\n");
    }

    md.push_str("\n## Space\n\n");
    md.push_str(&format!("- total tail states: {}\n", SPACE_SIZE));

    md
}

fn grouped_origin_counts(hist: &BTreeMap<String, u64>) -> (u64, u64, u64, u64) {
    let seed = hist
        .iter()
        .filter(|(k, _)| k.starts_with("seed_"))
        .map(|(_, v)| *v)
        .sum();
    let strict_core = hist.get("strict_core").copied().unwrap_or(0);
    let bootstrap5 = hist
        .get("provisional_bare_bootstrap_5")
        .copied()
        .unwrap_or(0);
    let bootstrap6 = hist
        .get("provisional_extra_bootstrap_6")
        .copied()
        .unwrap_or(0);
    (seed, strict_core, bootstrap5, bootstrap6)
}

pub fn build_closure_vs_cumulative_summary(
    closure_strict: Option<&ClosureDistribution>,
    closure_provisional: Option<&ClosureDistribution>,
    cumulative_strict: Option<&CumulativeDistribution>,
    cumulative_provisional: Option<&CumulativeDistribution>,
) -> String {
    let mut md = String::new();
    md.push_str("# Closure vs Cumulative Summary\n\n");
    md.push_str("- Closure-class semantics assigns absolute classes {1,2,4,5,6} and does not accumulate predecessor class.\n");
    md.push_str(
        "- Cumulative semantics uses path-length-like accumulation (`d(next)=d(cur)+delta`).\n\n",
    );

    md.push_str("## Any-Collapse Histograms\n\n");
    if let Some(strict) = closure_strict {
        md.push_str(&format!(
            "- closure strict: {}\n",
            format_histogram(&strict.any_class_histogram)
        ));
    }
    if let Some(provisional) = closure_provisional {
        md.push_str(&format!(
            "- closure provisional: {}\n",
            format_histogram(&provisional.any_class_histogram)
        ));
    }
    if let Some(strict) = cumulative_strict {
        md.push_str(&format!(
            "- cumulative strict: {}\n",
            format_histogram(&strict.any_cost_histogram)
        ));
    }
    if let Some(provisional) = cumulative_provisional {
        md.push_str(&format!(
            "- cumulative provisional: {}\n",
            format_histogram(&provisional.any_cost_histogram)
        ));
    }
    md.push('\n');
    md
}

pub fn build_final_real_native_closure_report(
    metadata: &RunMetadata,
    closure_strict: Option<&ClosureDistribution>,
    closure_provisional: Option<&ClosureDistribution>,
    cumulative_strict: Option<&CumulativeDistribution>,
    cumulative_provisional: Option<&CumulativeDistribution>,
) -> String {
    let mut md = String::new();
    md.push_str("# Final Real Native Closure Report\n\n");
    md.push_str("This is a closure-class benchmark.\n\n");
    md.push_str("- Costs 4 / 5 / 6 are absolute classes, not cumulative path lengths.\n");
    md.push_str("- Closure depth is reported separately and is NOT a cost.\n\n");

    md.push_str("## Real input summary\n\n");
    md.push_str(&format!(
        "- native row count: {}\n",
        metadata.native_row_count
    ));
    md.push_str(&format!(
        "- distinct generators by head: X={}, Y={}, Z={}\n",
        metadata.distinct_x_kernel_generator_tails,
        metadata.distinct_y_kernel_generator_tails,
        metadata.distinct_z_kernel_generator_tails
    ));
    md.push_str(&format!("- pivot index: {}\n\n", metadata.pivot_index));

    md.push_str("## Closure-class histograms\n\n");
    if let Some(strict) = closure_strict {
        md.push_str(&format!(
            "- strict_any: {}\n",
            format_histogram(&strict.any_class_histogram)
        ));
        md.push_str(&format!(
            "- strict_headed: {}\n",
            format_histogram(&strict.headed_class_histogram)
        ));
    }
    if let Some(provisional) = closure_provisional {
        md.push_str(&format!(
            "- provisional_any: {}\n",
            format_histogram(&provisional.any_class_histogram)
        ));
        md.push_str(&format!(
            "- provisional_headed: {}\n",
            format_histogram(&provisional.headed_class_histogram)
        ));
    }
    md.push('\n');

    md.push_str("## Weight-by-class analysis (headed collapse)\n\n");
    if let Some(strict) = closure_strict {
        md.push_str("### strict H_headed[w][class]\n\n");
        for line in format_weight_by_class_lines(
            &strict.headed_weight_by_class,
            &["2", "4", "5", "6", "uncovered"],
        ) {
            md.push_str(&line);
            md.push('\n');
        }
        md.push('\n');
    }
    if let Some(provisional) = closure_provisional {
        md.push_str("### provisional H_headed[w][class]\n\n");
        for line in format_weight_by_class_lines(
            &provisional.headed_weight_by_class,
            &["2", "4", "5", "6", "uncovered"],
        ) {
            md.push_str(&line);
            md.push('\n');
        }
        md.push('\n');
    }

    md.push_str("## Discovery-origin analysis\n\n");
    if let Some(strict) = closure_strict {
        let (seed, core, b5, b6) = grouped_origin_counts(&strict.first_origin_histogram);
        md.push_str(&format!(
            "- strict first-discovery grouped: stage1_seeds={seed}, strict_core={core}, bootstrap5={b5}, bootstrap6={b6}\n"
        ));
    }
    if let Some(provisional) = closure_provisional {
        let (seed, core, b5, b6) = grouped_origin_counts(&provisional.first_origin_histogram);
        md.push_str(&format!(
            "- provisional first-discovery grouped: stage1_seeds={seed}, strict_core={core}, bootstrap5={b5}, bootstrap6={b6}\n"
        ));
    }
    md.push('\n');

    md.push_str("## Best-class-origin analysis\n\n");
    if let Some(strict) = closure_strict {
        let (seed, core, b5, b6) = grouped_origin_counts(&strict.best_class_origin_histogram);
        md.push_str(&format!(
            "- strict best-class grouped: stage1={seed}, strict_core={core}, bootstrap5={b5}, bootstrap6={b6}\n"
        ));
        md.push_str(&format!(
            "- strict class-improvement by origin: {}\n",
            format_named_histogram(&strict.class_improvement_origin_histogram)
        ));
    }
    if let Some(provisional) = closure_provisional {
        let (seed, core, b5, b6) = grouped_origin_counts(&provisional.best_class_origin_histogram);
        md.push_str(&format!(
            "- provisional best-class grouped: stage1={seed}, strict_core={core}, bootstrap5={b5}, bootstrap6={b6}\n"
        ));
        md.push_str(&format!(
            "- provisional class-improvement by origin: {}\n",
            format_named_histogram(&provisional.class_improvement_origin_histogram)
        ));
    }
    md.push('\n');

    md.push_str("## Closure-depth analysis\n\n");
    if let Some(strict) = closure_strict {
        md.push_str(&format!(
            "- strict any depth histogram: {}\n",
            format_histogram(&strict.any_depth_histogram)
        ));
        md.push_str(&format!(
            "- strict headed depth histogram: {}\n",
            format_histogram(&strict.headed_depth_histogram)
        ));
        md.push_str(&format_depth_stats_line(
            "strict any",
            &strict.any_depth_histogram,
        ));
        md.push('\n');
        md.push_str(&format_depth_stats_line(
            "strict headed",
            &strict.headed_depth_histogram,
        ));
        md.push('\n');
    }
    if let Some(provisional) = closure_provisional {
        md.push_str(&format!(
            "- provisional any depth histogram: {}\n",
            format_histogram(&provisional.any_depth_histogram)
        ));
        md.push_str(&format!(
            "- provisional headed depth histogram: {}\n",
            format_histogram(&provisional.headed_depth_histogram)
        ));
        md.push_str(&format_depth_stats_line(
            "provisional any",
            &provisional.any_depth_histogram,
        ));
        md.push('\n');
        md.push_str(&format_depth_stats_line(
            "provisional headed",
            &provisional.headed_depth_histogram,
        ));
        md.push('\n');
    }
    md.push('\n');

    md.push_str("## Closure vs cumulative comparison\n\n");
    if let Some(strict) = closure_strict {
        md.push_str(&format!(
            "- closure strict any: {}\n",
            format_histogram(&strict.any_class_histogram)
        ));
    }
    if let Some(strict) = cumulative_strict {
        md.push_str(&format!(
            "- cumulative strict any: {}\n",
            format_histogram(&strict.any_cost_histogram)
        ));
    }
    if let Some(provisional) = closure_provisional {
        md.push_str(&format!(
            "- closure provisional any: {}\n",
            format_histogram(&provisional.any_class_histogram)
        ));
    }
    if let Some(provisional) = cumulative_provisional {
        md.push_str(&format!(
            "- cumulative provisional any: {}\n",
            format_histogram(&provisional.any_cost_histogram)
        ));
    }
    md.push('\n');
    md.push_str("- Closure class is left-bounded by absolute classes, while cumulative is path-length-like and can create larger bins.\n\n");

    md.push_str("## Conclusion\n\n");
    let closure_hist_for_shift = closure_provisional
        .map(|d| &d.any_class_histogram)
        .or_else(|| closure_strict.map(|d| &d.any_class_histogram));
    let cumulative_hist_for_shift = cumulative_provisional
        .map(|d| &d.any_cost_histogram)
        .or_else(|| cumulative_strict.map(|d| &d.any_cost_histogram));
    let closure_max = closure_hist_for_shift.and_then(max_numeric_key);
    let cumulative_max = cumulative_hist_for_shift.and_then(max_numeric_key);
    let left_shift_real =
        matches!((closure_max, cumulative_max), (Some(c), Some(k)) if c <= 6 && k > 6);

    let bootstrap_materially_changes = match (closure_strict, closure_provisional) {
        (Some(s), Some(p)) => {
            s.any_class_histogram != p.any_class_histogram
                || s.headed_class_histogram != p.headed_class_histogram
        }
        _ => false,
    };

    let (class4_fraction, depth_ge2_fraction) =
        if let Some(closure) = closure_provisional.or(closure_strict) {
            let total = closure.any_class_histogram.values().sum::<u64>().max(1);
            let class4 = closure.any_class_histogram.get("4").copied().unwrap_or(0);
            let depth_ge2 = closure
                .any_depth_histogram
                .iter()
                .filter_map(|(k, v)| {
                    if k == "uncovered" {
                        None
                    } else {
                        k.parse::<u16>().ok().and_then(|d| (d >= 2).then_some(*v))
                    }
                })
                .sum::<u64>();
            (
                class4 as f64 / total as f64,
                depth_ge2 as f64 / total as f64,
            )
        } else {
            (0.0, 0.0)
        };

    md.push_str(&format!(
        "- Left shift on full real input: {} (closure max bin {:?}, cumulative max bin {:?}).\n",
        if left_shift_real { "YES" } else { "NO" },
        closure_max,
        cumulative_max
    ));
    md.push_str(&format!(
        "- Bootstrap 5/6 materially changes final class distribution: {}.\n",
        if bootstrap_materially_changes {
            "YES"
        } else {
            "NO"
        }
    ));
    md.push_str(&format!(
        "- Most states are class 4: {} (class-4 fraction {:.6}); many are discovered at depth >=2: {} (fraction {:.6}).\n",
        if class4_fraction > 0.9 { "YES" } else { "NO" },
        class4_fraction,
        if depth_ge2_fraction > 0.5 { "YES" } else { "NO" },
        depth_ge2_fraction
    ));
    md
}

pub fn write_required_outputs(
    root: &Path,
    closure_strict: Option<&ClosureDistribution>,
    closure_provisional: Option<&ClosureDistribution>,
    cumulative_strict: Option<&CumulativeDistribution>,
    cumulative_provisional: Option<&CumulativeDistribution>,
    markdown_report: &str,
    closure_vs_cumulative_summary_md: &str,
    final_real_native_closure_md: &str,
    metadata: &RunMetadata,
    figure9_csv: Option<&Path>,
) -> Result<OutputPaths> {
    fs::create_dir_all(root)?;
    let paths = OutputPaths::in_dir(root);

    write_closure_histograms(&paths, closure_strict, closure_provisional)?;
    write_cumulative_histograms(&paths, cumulative_strict, cumulative_provisional)?;

    ensure_parent(&paths.final_absolute_class_report_md)?;
    fs::write(&paths.final_absolute_class_report_md, markdown_report)?;
    ensure_parent(&paths.closure_vs_cumulative_summary_md)?;
    fs::write(
        &paths.closure_vs_cumulative_summary_md,
        closure_vs_cumulative_summary_md,
    )?;
    ensure_parent(&paths.final_real_native_closure_report_md)?;
    fs::write(
        &paths.final_real_native_closure_report_md,
        final_real_native_closure_md,
    )?;

    write_json(&paths.run_metadata_json, metadata)?;

    if let (Some(figure9), Some(closure_any_hist)) = (
        figure9_csv,
        closure_provisional
            .map(|d| &d.any_class_histogram)
            .or_else(|| closure_strict.map(|d| &d.any_class_histogram)),
    ) {
        write_overlay_csv(
            &paths.overlay_compare_with_figure9_csv,
            figure9,
            closure_any_hist,
        )?;
    }

    Ok(paths)
}

#[derive(Debug, Clone)]
pub struct MicOutputPaths {
    pub baseline_beta_histogram_json: PathBuf,
    pub reuse_headed_cost_histogram_json: PathBuf,
    pub mic_cost_histogram_json: PathBuf,
    pub mic_cdf_csv: PathBuf,
    pub mic_weight_by_cost_json: PathBuf,
    pub mic_improvement_summary_json: PathBuf,
    pub mic_origin_summary_json: PathBuf,
    pub mic_depth_summary_json: PathBuf,
    pub mic_vs_baseline_report_md: PathBuf,
    pub mic_overlay_compare_with_figure9_csv: PathBuf,
}

impl MicOutputPaths {
    pub fn in_dir(root: &Path, prefix: Option<&str>) -> Self {
        let name = |base: &str| -> PathBuf {
            if let Some(prefix) = prefix {
                root.join(format!("{prefix}_{base}"))
            } else {
                root.join(base)
            }
        };

        Self {
            baseline_beta_histogram_json: name("baseline_beta_histogram.json"),
            reuse_headed_cost_histogram_json: name("reuse_headed_cost_histogram.json"),
            mic_cost_histogram_json: name("mic_cost_histogram.json"),
            mic_cdf_csv: name("mic_cdf.csv"),
            mic_weight_by_cost_json: name("mic_weight_by_cost.json"),
            mic_improvement_summary_json: name("mic_improvement_summary.json"),
            mic_origin_summary_json: name("mic_origin_summary.json"),
            mic_depth_summary_json: name("mic_depth_summary.json"),
            mic_vs_baseline_report_md: name("mic_vs_baseline_report.md"),
            mic_overlay_compare_with_figure9_csv: name("mic_overlay_compare_with_figure9.csv"),
        }
    }
}

fn write_mic_cdf_csv(path: &Path, cdf: &[super::mic::MicCdfRow]) -> Result<()> {
    let mut lines = Vec::with_capacity(cdf.len() + 1);
    lines.push("cost,baseline_cdf,mic_cdf,baseline_leq_count,mic_leq_count".to_string());
    for row in cdf {
        lines.push(format!(
            "{},{:.12},{:.12},{},{}",
            row.cost, row.baseline_cdf, row.mic_cdf, row.baseline_leq_count, row.mic_leq_count
        ));
    }
    ensure_parent(path)?;
    fs::write(path, lines.join("\n"))?;
    Ok(())
}

fn write_mic_overlay_csv(
    path: &Path,
    mode_result: &MicModeResult,
    figure9_csv: Option<&Path>,
) -> Result<()> {
    let paper_hist = if let Some(path) = figure9_csv {
        parse_figure9_csv(path)?
            .into_iter()
            .filter_map(|(k, v)| u16::try_from(k).ok().map(|k16| (k16, v)))
            .collect::<BTreeMap<_, _>>()
    } else {
        BTreeMap::<u16, u64>::new()
    };

    let rows = build_overlay_rows(
        &paper_hist,
        &mode_result.baseline_beta_histogram,
        &mode_result.mic_cost_histogram,
    );

    let mut lines = Vec::with_capacity(rows.len() + 1);
    lines.push("cost,paper_count,baseline_count,mic_count".to_string());
    for row in rows {
        lines.push(format!(
            "{},{},{},{}",
            row.cost, row.paper_count, row.baseline_count, row.mic_count
        ));
    }
    ensure_parent(path)?;
    fs::write(path, lines.join("\n"))?;
    Ok(())
}

pub fn build_mic_vs_baseline_report(
    mic_run: &MicRun,
    mode_result: &MicModeResult,
    baseline_source_label: &str,
) -> String {
    let mut md = String::new();
    md.push_str("# MIC vs Baseline Report\n\n");

    md.push_str("## Baseline semantics\n\n");
    md.push_str("- `beta(P)` is isolated synthesis cost with pivot-head minimization over `Q in {X,Y,Z}`.\n");
    md.push_str("- Unit: in-module bicycle measurement count.\n");
    md.push_str(&format!("- Baseline source: {baseline_source_label}\n\n"));

    md.push_str("## Reuse semantics\n\n");
    md.push_str("- Reuse cost comes from closure-reachable headed states only (`X/Y/Z`), then calibrated to the same physical unit.\n");
    md.push_str("- Bare states are witnesses only and are excluded from headed reuse cost.\n\n");

    md.push_str("## Final MIC definition\n\n");
    md.push_str("- `MIC(P) = min(beta_isolated(P), intro_cost_reuse(P))`\n\n");

    md.push_str("## Histograms\n\n");
    md.push_str(&format!(
        "- baseline histogram: {}\n",
        format_histogram(&mode_result.baseline_beta_histogram)
    ));
    md.push_str(&format!(
        "- reuse-only headed histogram: {}\n",
        format_histogram(&mode_result.reuse_headed_cost_histogram)
    ));
    md.push_str(&format!(
        "- final MIC histogram: {}\n\n",
        format_histogram(&mode_result.mic_cost_histogram)
    ));

    md.push_str("## CDF comparison\n\n");
    if let (Some(first), Some(last)) = (mode_result.mic_cdf.first(), mode_result.mic_cdf.last()) {
        md.push_str(&format!(
            "- CDF axis min/max cost: {} -> {}\n",
            first.cost, last.cost
        ));
    } else {
        md.push_str("- CDF unavailable (empty axis)\n");
    }
    md.push('\n');

    let imp = &mode_result.mic_improvement_summary;
    md.push_str("## Improvement statistics\n\n");
    md.push_str(&format!("- improved count: {}\n", imp.improved_count));
    md.push_str(&format!(
        "- improved fraction: {:.8}\n",
        imp.improved_fraction
    ));
    md.push_str(&format!(
        "- mean improvement among improved tails: {:.6}\n",
        imp.mean_improvement_among_improved
    ));
    md.push_str(&format!(
        "- median improvement among improved tails: {}\n",
        imp.median_improvement_among_improved
    ));
    md.push_str(&format!(
        "- p90 improvement among improved tails: {}\n\n",
        imp.p90_improvement_among_improved
    ));

    md.push_str("## Left-tail mass\n\n");
    md.push_str(&format!(
        "- Pr[cost <= 7]: baseline={:.8}, MIC={:.8}\n",
        imp.left_tail_mass_baseline_leq_7, imp.left_tail_mass_mic_leq_7
    ));
    md.push_str(&format!(
        "- Pr[cost <= 13]: baseline={:.8}, MIC={:.8}\n",
        imp.left_tail_mass_baseline_leq_13, imp.left_tail_mass_mic_leq_13
    ));
    md.push_str(&format!(
        "- Pr[cost <= 19]: baseline={:.8}, MIC={:.8}\n\n",
        imp.left_tail_mass_baseline_leq_19, imp.left_tail_mass_mic_leq_19
    ));

    md.push_str("## Weight-resolved improvement\n\n");
    for w in 0..super::pauli::SUPPORT_WEIGHT_BUCKETS {
        if let Some(row) = mode_result.weight_resolved_summary.get(&w.to_string()) {
            md.push_str(&format!(
                "- w={w}: baseline_mean={:.6}, mic_mean={:.6}, improved_fraction={:.6}, shift_leq7={:.6}, shift_leq13={:.6}, shift_leq19={:.6}\n",
                row.baseline_mean_cost,
                row.mic_mean_cost,
                row.improved_fraction,
                row.mic_left_tail_mass_leq_7 - row.baseline_left_tail_mass_leq_7,
                row.mic_left_tail_mass_leq_13 - row.baseline_left_tail_mass_leq_13,
                row.mic_left_tail_mass_leq_19 - row.baseline_left_tail_mass_leq_19,
            ));
        }
    }
    md.push('\n');

    md.push_str("## Origin/depth analysis\n\n");
    md.push_str(&format!(
        "- reused tails (MIC decided by reuse): {}\n",
        mode_result.mic_origin_summary.reused_tail_count
    ));
    md.push_str(&format!(
        "- reused by group: strict_core={}, bootstrap5={}, bootstrap6={}\n",
        mode_result
            .mic_origin_summary
            .reused_by_group
            .get("strict_core")
            .copied()
            .unwrap_or(0),
        mode_result
            .mic_origin_summary
            .reused_by_group
            .get("bootstrap5")
            .copied()
            .unwrap_or(0),
        mode_result
            .mic_origin_summary
            .reused_by_group
            .get("bootstrap6")
            .copied()
            .unwrap_or(0),
    ));
    md.push_str(&format!(
        "- reused depth summary: mean={:.6}, median={}, p90={}\n\n",
        mode_result.mic_depth_summary.mean_depth,
        mode_result.mic_depth_summary.median_depth,
        mode_result.mic_depth_summary.p90_depth
    ));

    md.push_str("## Plain-language conclusion\n\n");
    let baseline_mean = mode_result
        .baseline_costs
        .iter()
        .map(|&v| v as u64)
        .sum::<u64>() as f64
        / mode_result.baseline_costs.len().max(1) as f64;
    let mic_mean = mode_result.mic_costs.iter().map(|&v| v as u64).sum::<u64>() as f64
        / mode_result.mic_costs.len().max(1) as f64;
    let left_shift = imp.left_tail_mass_mic_leq_7 > imp.left_tail_mass_baseline_leq_7
        || mic_mean < baseline_mean;
    let strict_core = mode_result
        .mic_origin_summary
        .reused_by_group
        .get("strict_core")
        .copied()
        .unwrap_or(0);
    let bootstrap = mode_result
        .mic_origin_summary
        .reused_by_group
        .get("bootstrap5")
        .copied()
        .unwrap_or(0)
        + mode_result
            .mic_origin_summary
            .reused_by_group
            .get("bootstrap6")
            .copied()
            .unwrap_or(0);
    let depth_is_deep = mode_result.mic_depth_summary.median_depth >= 3;

    md.push_str(&format!(
        "- MIC left-shift relative to isolated baseline: {}.\n",
        if left_shift { "YES" } else { "NO" }
    ));
    md.push_str(&format!(
        "- Gain source is mostly {}.\n",
        if strict_core >= bootstrap {
            "strict core"
        } else {
            "bootstrap rules"
        }
    ));
    md.push_str(&format!(
        "- Reused tails are mostly {} in closure depth.\n",
        if depth_is_deep { "deep" } else { "shallow" }
    ));

    md.push_str(&format!(
        "\n- mode: {}\n- seed policy: {}\n- seed threshold: {:?}\n",
        mode_result.mode.as_str(),
        mic_run.seed_policy.as_str(),
        mic_run.seed_threshold
    ));
    md
}

pub fn write_mic_mode_outputs(
    root: &Path,
    mode_result: &MicModeResult,
    mic_run: &MicRun,
    baseline_source_label: &str,
    figure9_csv: Option<&Path>,
    prefix: Option<&str>,
) -> Result<MicOutputPaths> {
    fs::create_dir_all(root)?;
    let paths = MicOutputPaths::in_dir(root, prefix);

    write_json(
        &paths.baseline_beta_histogram_json,
        &mode_result.baseline_beta_histogram,
    )?;
    write_json(
        &paths.reuse_headed_cost_histogram_json,
        &mode_result.reuse_headed_cost_histogram,
    )?;
    write_json(
        &paths.mic_cost_histogram_json,
        &mode_result.mic_cost_histogram,
    )?;
    write_mic_cdf_csv(&paths.mic_cdf_csv, &mode_result.mic_cdf)?;
    write_json(
        &paths.mic_weight_by_cost_json,
        &mode_result.mic_weight_by_cost,
    )?;
    write_json(
        &paths.mic_improvement_summary_json,
        &mode_result.mic_improvement_summary,
    )?;
    write_json(
        &paths.mic_origin_summary_json,
        &mode_result.mic_origin_summary,
    )?;
    write_json(
        &paths.mic_depth_summary_json,
        &mode_result.mic_depth_summary,
    )?;

    let report = build_mic_vs_baseline_report(mic_run, mode_result, baseline_source_label);
    ensure_parent(&paths.mic_vs_baseline_report_md)?;
    fs::write(&paths.mic_vs_baseline_report_md, report)?;

    write_mic_overlay_csv(
        &paths.mic_overlay_compare_with_figure9_csv,
        mode_result,
        figure9_csv,
    )?;

    Ok(paths)
}
