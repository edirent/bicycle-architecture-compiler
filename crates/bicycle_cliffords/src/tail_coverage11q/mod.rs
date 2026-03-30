use serde::{Deserialize, Serialize};

use self::hist::{
    ClosureDistribution, CumulativeDistribution, ModeDelta, build_closure_distribution,
    build_cumulative_distribution, compute_mode_delta,
};
use self::pauli::build_support_weight_lut;
use self::stage1::{Stage1Result, build_stage1};
use self::typed_closure::run_typed_closure;
use self::typed_dp::run_typed_dp;

pub mod beta_baseline;
pub mod calibration;
pub mod hist;
pub mod mic;
pub mod native;
pub mod pauli;
pub mod report;
pub mod stage1;
pub mod typed_closure;
pub mod typed_dp;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Head {
    I,
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRow {
    pub index: usize,
    pub head: Head,
    pub tail_bits: u32,
    pub metadata: Option<String>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum CoverageMode {
    Strict,
    Provisional,
}

impl CoverageMode {
    pub fn as_str(self) -> &'static str {
        match self {
            CoverageMode::Strict => "strict",
            CoverageMode::Provisional => "provisional",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum RequestedMode {
    Strict,
    Provisional,
    Both,
}

impl RequestedMode {
    pub fn as_str(self) -> &'static str {
        match self {
            RequestedMode::Strict => "strict",
            RequestedMode::Provisional => "provisional",
            RequestedMode::Both => "both",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum CoverageSemantics {
    Cumulative,
    Closure,
    Both,
}

impl CoverageSemantics {
    pub fn as_str(self) -> &'static str {
        match self {
            CoverageSemantics::Cumulative => "cumulative",
            CoverageSemantics::Closure => "closure",
            CoverageSemantics::Both => "both",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum CoverageExperiment {
    Cumulative,
    ClosureClass,
    Mic,
    All,
}

impl CoverageExperiment {
    pub fn as_str(self) -> &'static str {
        match self {
            CoverageExperiment::Cumulative => "cumulative",
            CoverageExperiment::ClosureClass => "closure_class",
            CoverageExperiment::Mic => "mic",
            CoverageExperiment::All => "all",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct RuleToggles {
    pub enable_strict_core: bool,
    pub enable_bootstrap_5: bool,
    pub enable_bootstrap_6: bool,
}

impl RuleToggles {
    pub const fn strict_defaults() -> Self {
        Self {
            enable_strict_core: true,
            enable_bootstrap_5: false,
            enable_bootstrap_6: false,
        }
    }

    pub const fn provisional_defaults() -> Self {
        Self {
            enable_strict_core: true,
            enable_bootstrap_5: true,
            enable_bootstrap_6: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModeRuleToggles {
    pub strict: RuleToggles,
    pub provisional: RuleToggles,
}

impl Default for ModeRuleToggles {
    fn default() -> Self {
        Self {
            strict: RuleToggles::strict_defaults(),
            provisional: RuleToggles::provisional_defaults(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoverageRun {
    pub stage1: Stage1Result,
    pub cumulative_strict: Option<CumulativeDistribution>,
    pub cumulative_provisional: Option<CumulativeDistribution>,
    pub closure_strict: Option<ClosureDistribution>,
    pub closure_provisional: Option<ClosureDistribution>,
    pub cumulative_delta: Option<ModeDelta>,
    pub closure_delta: Option<ModeDelta>,
}

pub fn run_exact_coverage(
    native_rows: &[NativeRow],
    requested_mode: RequestedMode,
    semantics: CoverageSemantics,
) -> CoverageRun {
    run_exact_coverage_with_toggles(
        native_rows,
        requested_mode,
        semantics,
        ModeRuleToggles::default(),
    )
}

pub fn run_exact_coverage_with_toggles(
    native_rows: &[NativeRow],
    requested_mode: RequestedMode,
    semantics: CoverageSemantics,
    mode_toggles: ModeRuleToggles,
) -> CoverageRun {
    let support_weight_lut = build_support_weight_lut();
    let stage1 = build_stage1(native_rows);

    let run_cumulative = matches!(
        semantics,
        CoverageSemantics::Cumulative | CoverageSemantics::Both
    );
    let run_closure = matches!(
        semantics,
        CoverageSemantics::Closure | CoverageSemantics::Both
    );

    let cumulative_strict_raw = (run_cumulative
        && matches!(requested_mode, RequestedMode::Strict | RequestedMode::Both))
    .then(|| {
        run_typed_dp(
            CoverageMode::Strict,
            mode_toggles.strict,
            &stage1.all_generators,
            &stage1.seeds,
        )
    });

    let cumulative_provisional_raw = (run_cumulative
        && matches!(
            requested_mode,
            RequestedMode::Provisional | RequestedMode::Both
        ))
    .then(|| {
        run_typed_dp(
            CoverageMode::Provisional,
            mode_toggles.provisional,
            &stage1.all_generators,
            &stage1.seeds,
        )
    });

    let mut cumulative_strict = cumulative_strict_raw.as_ref().map(|result| {
        build_cumulative_distribution(result, &support_weight_lut, stage1.validation, true)
    });
    let mut cumulative_provisional = cumulative_provisional_raw.as_ref().map(|result| {
        build_cumulative_distribution(result, &support_weight_lut, stage1.validation, true)
    });

    let cumulative_delta =
        if let (Some(strict), Some(provisional)) = (&cumulative_strict, &cumulative_provisional) {
            let delta = compute_mode_delta(
                &strict.any_weight_distribution,
                &provisional.any_weight_distribution,
            );
            Some(delta)
        } else {
            None
        };

    if let Some(delta) = cumulative_delta {
        if let Some(strict) = &mut cumulative_strict {
            strict.validations.strict_provisional_monotone_ok = delta.monotone_ok;
        }
        if let Some(provisional) = &mut cumulative_provisional {
            provisional.validations.strict_provisional_monotone_ok = delta.monotone_ok;
        }
    }

    let closure_strict_raw = (run_closure
        && matches!(requested_mode, RequestedMode::Strict | RequestedMode::Both))
    .then(|| {
        run_typed_closure(
            CoverageMode::Strict,
            mode_toggles.strict,
            &stage1.all_generators,
            &stage1.seeds,
        )
    });

    let closure_provisional_raw = (run_closure
        && matches!(
            requested_mode,
            RequestedMode::Provisional | RequestedMode::Both
        ))
    .then(|| {
        run_typed_closure(
            CoverageMode::Provisional,
            mode_toggles.provisional,
            &stage1.all_generators,
            &stage1.seeds,
        )
    });

    let mut closure_strict = closure_strict_raw.as_ref().map(|result| {
        build_closure_distribution(result, &support_weight_lut, stage1.validation, true)
    });
    let mut closure_provisional = closure_provisional_raw.as_ref().map(|result| {
        build_closure_distribution(result, &support_weight_lut, stage1.validation, true)
    });

    let closure_delta =
        if let (Some(strict), Some(provisional)) = (&closure_strict, &closure_provisional) {
            Some(compute_mode_delta(
                &strict.any_weight_distribution,
                &provisional.any_weight_distribution,
            ))
        } else {
            None
        };

    if let Some(delta) = closure_delta {
        if let Some(strict) = &mut closure_strict {
            strict.validations.strict_provisional_monotone_ok = delta.monotone_ok;
        }
        if let Some(provisional) = &mut closure_provisional {
            provisional.validations.strict_provisional_monotone_ok = delta.monotone_ok;
        }
    }

    CoverageRun {
        stage1,
        cumulative_strict,
        cumulative_provisional,
        closure_strict,
        closure_provisional,
        cumulative_delta,
        closure_delta,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;
    use crate::tail_coverage11q::hist::build_closure_distribution;
    use crate::tail_coverage11q::stage1::Stage1Validation;
    use crate::tail_coverage11q::typed_closure::{
        INF_CLASS, INF_DEPTH, OriginKind, TypedClosureResult, emitted_transitions_for_generator,
        mark_state_with_outcome,
    };
    use crate::tail_coverage11q::typed_dp::{
        ActiveHead, HEAD_STATE_COUNT, NativeGenerator, SeedState, state_index,
    };

    fn row(index: usize, head: Head, tail_bits: u32) -> NativeRow {
        NativeRow {
            index,
            head,
            tail_bits,
            metadata: None,
        }
    }

    fn stage1_ok() -> Stage1Validation {
        Stage1Validation {
            histogram_total_ok: true,
            stage1_cost_ordering_ok: true,
            stage1_same_q_only_ok: true,
            direct_native_mask_consistency_ok: true,
            generators_dedup_by_head_and_tail_ok: true,
        }
    }

    fn key_subset(hist: &std::collections::BTreeMap<String, u64>, allowed: &[&str]) -> bool {
        hist.keys().all(|k| allowed.contains(&k.as_str()))
    }

    #[test]
    fn core_transition_assigns_absolute_4() {
        let transitions = emitted_transitions_for_generator(
            CoverageMode::Strict,
            RuleToggles::strict_defaults(),
            ActiveHead::X,
            0,
            NativeGenerator {
                head: ActiveHead::Y,
                tail_bits: 0,
            },
        );

        assert!(
            transitions
                .iter()
                .any(|(head, tail, class)| *head == ActiveHead::Z && *tail == 0 && *class == 4)
        );
        assert!(!transitions.iter().any(|(_, _, class)| *class == 6));
    }

    #[test]
    fn two_step_closure_still_yields_class_4() {
        let first = emitted_transitions_for_generator(
            CoverageMode::Strict,
            RuleToggles::strict_defaults(),
            ActiveHead::X,
            0,
            NativeGenerator {
                head: ActiveHead::Y,
                tail_bits: 1,
            },
        );
        let (b_head, b_tail, b_class) = first[0];
        assert_eq!(b_class, 4);

        let second = emitted_transitions_for_generator(
            CoverageMode::Strict,
            RuleToggles::strict_defaults(),
            b_head,
            b_tail,
            NativeGenerator {
                head: ActiveHead::X,
                tail_bits: 0,
            },
        );
        assert!(second.iter().any(|(_, _, class)| *class == 4));
        assert!(!second.iter().any(|(_, _, class)| *class == 10));
    }

    #[test]
    fn bare_bootstrap_assigns_absolute_5() {
        let z0 = 1u32 << 11;
        let transitions = emitted_transitions_for_generator(
            CoverageMode::Provisional,
            RuleToggles::provisional_defaults(),
            ActiveHead::Bare,
            z0,
            NativeGenerator {
                head: ActiveHead::X,
                tail_bits: 1,
            },
        );
        assert!(
            transitions
                .iter()
                .filter(|(_, _, class)| *class == 5)
                .count()
                >= 2
        );
        assert!(!transitions.iter().any(|(_, _, class)| *class > 6));
    }

    #[test]
    fn handwritten_branch_assigns_absolute_6() {
        let z0 = 1u32 << 11;
        let transitions = emitted_transitions_for_generator(
            CoverageMode::Provisional,
            RuleToggles::provisional_defaults(),
            ActiveHead::X,
            z0,
            NativeGenerator {
                head: ActiveHead::X,
                tail_bits: 1,
            },
        );

        assert!(transitions.iter().any(|(_, _, class)| *class == 6));
        assert!(!transitions.iter().any(|(_, _, class)| *class > 6));
    }

    #[test]
    fn strict_headed_support_subset() {
        let rows = vec![
            row(0, Head::I, 0),
            row(1, Head::X, 1),
            row(2, Head::Y, 2),
            row(3, Head::Z, 4),
        ];
        let run = run_exact_coverage(&rows, RequestedMode::Strict, CoverageSemantics::Closure);
        let strict = run.closure_strict.expect("strict closure present");
        assert!(key_subset(
            &strict.headed_class_histogram,
            &["2", "4", "uncovered"]
        ));
    }

    #[test]
    fn provisional_headed_support_subset() {
        let rows = vec![
            row(0, Head::I, 1u32 << 11),
            row(1, Head::X, 1),
            row(2, Head::Y, 2),
            row(3, Head::Z, 4),
        ];
        let run = run_exact_coverage(
            &rows,
            RequestedMode::Provisional,
            CoverageSemantics::Closure,
        );
        let provisional = run
            .closure_provisional
            .expect("provisional closure present");
        assert!(key_subset(
            &provisional.headed_class_histogram,
            &["2", "4", "5", "6", "uncovered"]
        ));
    }

    #[test]
    fn any_support_subset() {
        let rows = vec![
            row(0, Head::I, 1u32 << 11),
            row(1, Head::X, 1),
            row(2, Head::Y, 2),
            row(3, Head::Z, 4),
        ];
        let run = run_exact_coverage(
            &rows,
            RequestedMode::Provisional,
            CoverageSemantics::Closure,
        );
        let provisional = run
            .closure_provisional
            .expect("provisional closure present");
        assert!(key_subset(
            &provisional.any_class_histogram,
            &["1", "2", "4", "5", "6", "uncovered"]
        ));
    }

    #[test]
    fn closure_depth_separated_from_class() {
        let mut reachable = vec![false; HEAD_STATE_COUNT * pauli::SPACE_SIZE];
        let mut best_class = vec![INF_CLASS; HEAD_STATE_COUNT * pauli::SPACE_SIZE];
        let mut closure_depth = vec![INF_DEPTH; HEAD_STATE_COUNT * pauli::SPACE_SIZE];
        let mut first_origin = vec![OriginKind::Unset; HEAD_STATE_COUNT * pauli::SPACE_SIZE];
        let mut best_class_origin = vec![OriginKind::Unset; HEAD_STATE_COUNT * pauli::SPACE_SIZE];
        let mut queue = VecDeque::<u32>::new();

        let outcome_a = mark_state_with_outcome(
            &mut reachable,
            &mut best_class,
            &mut closure_depth,
            &mut first_origin,
            &mut best_class_origin,
            &mut queue,
            ActiveHead::X,
            9,
            6,
            3,
            OriginKind::ProvisionalExtraBootstrap6,
        );
        assert!(outcome_a.discovered);

        let outcome_b = mark_state_with_outcome(
            &mut reachable,
            &mut best_class,
            &mut closure_depth,
            &mut first_origin,
            &mut best_class_origin,
            &mut queue,
            ActiveHead::X,
            9,
            4,
            9,
            OriginKind::StrictCore,
        );
        assert!(outcome_b.class_improved);
        assert!(!outcome_b.discovered);

        let mut best_any = vec![INF_CLASS; pauli::SPACE_SIZE];
        let mut best_headed = vec![INF_CLASS; pauli::SPACE_SIZE];
        let mut depth_any = vec![INF_DEPTH; pauli::SPACE_SIZE];
        let mut depth_headed = vec![INF_DEPTH; pauli::SPACE_SIZE];

        best_any[9] = 4;
        best_headed[9] = 4;
        depth_any[9] = 3;
        depth_headed[9] = 3;

        let closure = TypedClosureResult {
            mode: CoverageMode::Strict,
            best_class_any: best_any,
            best_class_headed: best_headed,
            reachable_by_head_and_weight: [[0u64; pauli::SUPPORT_WEIGHT_BUCKETS]; HEAD_STATE_COUNT],
            depth_hist_any: std::collections::BTreeMap::new(),
            depth_hist_headed: std::collections::BTreeMap::new(),
            first_origin_histogram: std::collections::BTreeMap::new(),
            best_class_origin_histogram: std::collections::BTreeMap::new(),
            class_improvement_origin_histogram: std::collections::BTreeMap::new(),
            best_depth_any: depth_any,
            best_depth_headed: depth_headed,
            best_class_by_state: best_class,
            closure_depth_by_state: closure_depth,
            reachable_by_state: reachable,
            first_origin_by_state: first_origin,
            best_class_origin_by_state: best_class_origin,
            stats: Default::default(),
        };

        let dist = build_closure_distribution(
            &closure,
            &pauli::build_support_weight_lut(),
            stage1_ok(),
            true,
        );

        assert_eq!(dist.any_class_histogram.get("4").copied().unwrap_or(0), 1);
        assert_eq!(dist.any_depth_histogram.get("3").copied().unwrap_or(0), 1);
    }

    #[test]
    fn class_improvement_without_requeue() {
        let out = typed_closure::run_typed_closure(
            CoverageMode::Strict,
            RuleToggles::strict_defaults(),
            &[],
            &[
                SeedState {
                    head: ActiveHead::X,
                    tail_bits: 0,
                    class: 6,
                },
                SeedState {
                    head: ActiveHead::X,
                    tail_bits: 0,
                    class: 4,
                },
            ],
        );

        let idx = state_index(ActiveHead::X, 0);
        assert_eq!(out.best_class_by_state[idx], 4);
        assert_eq!(out.stats.enqueue_count, 1);
        assert!(out.stats.class_improvements_without_requeue >= 1);
    }

    #[test]
    fn no_cumulative_bins_present() {
        let rows = vec![
            row(0, Head::I, 1u32 << 11),
            row(1, Head::X, 1),
            row(2, Head::Y, 2),
            row(3, Head::Z, 4),
        ];
        let run = run_exact_coverage(
            &rows,
            RequestedMode::Provisional,
            CoverageSemantics::Closure,
        );
        let provisional = run
            .closure_provisional
            .expect("provisional closure present");

        assert!(provisional.validations.no_cumulative_bins_present);
        assert!(
            provisional
                .any_class_histogram
                .keys()
                .filter(|k| k.as_str() != "uncovered")
                .all(|k| k.parse::<u32>().is_ok_and(|v| v <= 6))
        );
    }

    #[test]
    fn toggles_disable_rule_families() {
        let rows = vec![
            row(0, Head::I, 1u32 << 11),
            row(1, Head::X, 1),
            row(2, Head::Y, 2),
            row(3, Head::Z, 4),
        ];

        let disabled = ModeRuleToggles {
            strict: RuleToggles {
                enable_strict_core: false,
                enable_bootstrap_5: false,
                enable_bootstrap_6: false,
            },
            provisional: RuleToggles {
                enable_strict_core: false,
                enable_bootstrap_5: false,
                enable_bootstrap_6: false,
            },
        };

        let run = run_exact_coverage_with_toggles(
            &rows,
            RequestedMode::Both,
            CoverageSemantics::Closure,
            disabled,
        );

        let strict = run.closure_strict.expect("strict closure present");
        let provisional = run
            .closure_provisional
            .expect("provisional closure present");

        assert!(
            strict
                .headed_class_histogram
                .keys()
                .all(|k| k == "2" || k == "uncovered")
        );
        assert!(
            provisional
                .headed_class_histogram
                .keys()
                .all(|k| k == "2" || k == "uncovered")
        );
    }
}
