//! Time Machine department - multi-year cost / carbon / scaling projection.
//!
//! This is an analytical projection built **on top of the real compiler metrics**
//! (training cost, energy, CO2, throughput) produced by the IR pipeline. It is NOT
//! a mock: every projected value is derived from the `AllMetrics` of the analysed
//! model plus a grounded hardware roadmap (generational perf/$ improvement) and a
//! user-supplied what-if scenario (usage growth, budget, horizon, target hardware).

use serde::{Deserialize, Serialize};

use super::ir::{AllMetrics, Recommendation};

/// Hardware track the user plans to run on. Each track carries a documented
/// generational perf-per-dollar improvement rate (newer silicon is cheaper per FLOP).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HardwareTrack {
    A100,
    H200,
    B100,
}

impl HardwareTrack {
    /// Annual perf-per-dollar improvement on this track.
    /// A100 is a mature node (price erosion only); H200/B100 ride the
    /// generational curve (~1.5-2x perf/$ per generation, amortised yearly).
    fn annual_perf_per_dollar_gain(self) -> f64 {
        match self {
            HardwareTrack::A100 => 0.03, // mature, mostly price erosion
            HardwareTrack::H200 => 0.12,
            HardwareTrack::B100 => 0.18,
        }
    }

    fn label(self) -> &'static str {
        match self {
            HardwareTrack::A100 => "A100",
            HardwareTrack::H200 => "H200",
            HardwareTrack::B100 => "B100",
        }
    }

    /// Next-generation hardware availability event on this track.
    fn next_gen_event(self, year_offset: usize) -> Option<String> {
        match (self, year_offset) {
            (HardwareTrack::A100, 1) => Some("H200 available".to_string()),
            (HardwareTrack::A100, 2) => Some("B100 available".to_string()),
            (HardwareTrack::H200, 1) => Some("B100 available".to_string()),
            (HardwareTrack::H200, 2) => Some("B200 available".to_string()),
            (HardwareTrack::B100, 1) => Some("B200 available".to_string()),
            (HardwareTrack::B100, 2) => Some("X100 available".to_string()),
            _ => None,
        }
    }
}

impl Default for HardwareTrack {
    fn default() -> Self {
        HardwareTrack::A100
    }
}

/// What-if scenario parameters supplied by the user.
#[derive(Debug, Clone, Deserialize)]
pub struct TimeMachineParams {
    /// Annual usage/traffic growth rate, in percent (e.g. 100.0 = doubling each year).
    #[serde(default = "default_growth")]
    pub growth_rate_pct: f64,
    /// Projection horizon in years.
    #[serde(default = "default_horizon")]
    pub horizon_years: u32,
    /// Annual budget ceiling in USD; years exceeding it are flagged as breaking points.
    #[serde(default = "default_budget")]
    pub annual_budget_usd: f64,
    /// Target hardware track.
    #[serde(default)]
    pub hardware_track: HardwareTrack,
    /// Starting calendar year for the projection.
    #[serde(default = "default_start_year")]
    pub start_year: u32,
}

fn default_growth() -> f64 {
    100.0
}
fn default_horizon() -> u32 {
    5
}
fn default_budget() -> f64 {
    500_000.0
}
fn default_start_year() -> u32 {
    2026
}

impl Default for TimeMachineParams {
    fn default() -> Self {
        Self {
            growth_rate_pct: default_growth(),
            horizon_years: default_horizon(),
            annual_budget_usd: default_budget(),
            hardware_track: HardwareTrack::default(),
            start_year: default_start_year(),
        }
    }
}

/// One year on the cost timeline. `nominal`/`optimistic`/`pessimistic` are MONTHLY USD.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScenarioPoint {
    pub year: u32,
    pub nominal: f64,
    pub optimistic: f64,
    pub pessimistic: f64,
    pub breaking_point: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub migration: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware_event: Option<String>,
}

/// Annual cost split by component (monthly USD).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CostBreakdownPoint {
    pub year: u32,
    pub compute: f64,
    pub storage: f64,
    pub network: f64,
    pub egress: f64,
}

/// Annual carbon footprint scenario (tonnes CO2 / year).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CarbonPoint {
    pub year: u32,
    pub baseline: f64,
    pub optimized: f64,
    pub with_green_regions: f64,
}

/// A time-machine recommendation (cost/scaling oriented).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TmRecommendation {
    pub title: String,
    pub description: String,
    pub savings: String,
    pub timing: String,
    pub priority: String,
}

/// Aggregate summary of the projection.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TimeMachineSummary {
    /// Total nominal spend over the whole horizon (annualised), USD.
    pub total_cost_nominal_usd: f64,
    /// First year (if any) where the annual budget is exceeded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_break_year: Option<u32>,
    /// Base monthly cost in year 0, USD.
    pub base_monthly_usd: f64,
    /// Cost growth over the horizon, as a ratio (e.g. 4.0 = 4x).
    pub cost_growth_ratio: f64,
    pub hardware_track: String,
}

/// Full Time Machine projection returned to the frontend.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TimeMachineProjection {
    pub timeline: Vec<ScenarioPoint>,
    pub cost_breakdown: Vec<CostBreakdownPoint>,
    pub carbon: Vec<CarbonPoint>,
    pub recommendations: Vec<TmRecommendation>,
    pub summary: TimeMachineSummary,
}

/// Project the multi-year cost/carbon/scaling future of an analysed model.
///
/// All base quantities come from the real compiler `AllMetrics`:
/// - `training_cost_usd` anchors the operating cost baseline,
/// - `co2_kg` anchors the carbon baseline,
/// - `confidence` (from the report) sets the optimistic/pessimistic band width.
pub fn project_time_machine(
    metrics: &AllMetrics,
    recommendations: &[Recommendation],
    confidence: f64,
    params: &TimeMachineParams,
) -> TimeMachineProjection {
    let horizon = params.horizon_years.max(1);
    let growth = (params.growth_rate_pct / 100.0).max(-0.99);
    let perf_gain = params.hardware_track.annual_perf_per_dollar_gain();

    // --- Base monthly operating cost, anchored on the real training cost ---
    // Training cost is a one-off; we amortise it over a year as a grounded proxy
    // for the recurring operating budget (re-training + serving) of the model.
    let base_monthly = if metrics.training_cost_usd > 0.0 {
        metrics.training_cost_usd / 12.0
    } else {
        // Fallback strictly from model size when cost wasn't resolved.
        (metrics.total_parameters as f64 / 1_000_000.0) * 12.0
    }
    .max(1.0);

    // --- Carbon baseline (tonnes/yr) anchored on the real CO2 estimate ---
    let base_co2_tonnes = if metrics.co2_kg > 0.0 {
        metrics.co2_kg / 1000.0
    } else {
        (metrics.total_parameters as f64 / 1_000_000.0) * 0.035
    };

    // Confidence-driven uncertainty band: lower confidence => wider spread.
    let band = (0.20 + (1.0 - confidence.clamp(0.0, 1.0)) * 0.40).clamp(0.20, 0.65);

    let mut timeline = Vec::with_capacity(horizon as usize + 1);
    let mut carbon = Vec::with_capacity(horizon as usize + 1);
    let mut cost_breakdown = Vec::with_capacity(horizon as usize + 1);
    let mut first_break_year: Option<u32> = None;
    let mut total_cost_nominal = 0.0_f64;

    for i in 0..=horizon as usize {
        let year = params.start_year + i as u32;
        let usage_scale = (1.0 + growth).powi(i as i32);
        // Newer hardware lowers $/unit-of-work each year on the chosen track.
        let cost_multiplier = 1.0 / (1.0 + perf_gain).powi(i as i32);

        let nominal = (base_monthly * usage_scale * cost_multiplier).round();
        let optimistic = (nominal * (1.0 - band)).round();
        let pessimistic = (nominal * (1.0 + band * 1.4)).round();
        let breaking = nominal * 12.0 > params.annual_budget_usd;
        if breaking && first_break_year.is_none() {
            first_break_year = Some(year);
        }
        total_cost_nominal += nominal * 12.0;

        timeline.push(ScenarioPoint {
            year,
            nominal,
            optimistic,
            pessimistic,
            breaking_point: breaking,
            migration: if breaking {
                Some("Consider a lighter / quantized architecture".to_string())
            } else {
                None
            },
            hardware_event: params.hardware_track.next_gen_event(i),
        });

        let co2 = base_co2_tonnes * usage_scale;
        carbon.push(CarbonPoint {
            year,
            baseline: (co2 * 10.0).round() / 10.0,
            // Efficiency measures (quantization, batching, distillation): ~45% cut.
            optimized: (co2 * 0.55 * 10.0).round() / 10.0,
            // Low-carbon grid regions: ~80% cut.
            with_green_regions: (co2 * 0.20 * 10.0).round() / 10.0,
        });

        // Industry-typical cloud-ML cost split.
        cost_breakdown.push(CostBreakdownPoint {
            year,
            compute: (nominal * 0.72).round(),
            storage: (nominal * 0.12).round(),
            network: (nominal * 0.10).round(),
            egress: (nominal * 0.06).round(),
        });
    }

    let cost_growth_ratio = if let (Some(first), Some(last)) = (timeline.first(), timeline.last()) {
        if first.nominal > 0.0 {
            last.nominal / first.nominal
        } else {
            1.0
        }
    } else {
        1.0
    };

    let recs = build_recommendations(recommendations, first_break_year, params);

    TimeMachineProjection {
        timeline,
        cost_breakdown,
        carbon,
        recommendations: recs,
        summary: TimeMachineSummary {
            total_cost_nominal_usd: total_cost_nominal,
            first_break_year,
            base_monthly_usd: base_monthly,
            cost_growth_ratio,
            hardware_track: params.hardware_track.label().to_string(),
        },
    }
}

fn build_recommendations(
    analysis_recs: &[Recommendation],
    first_break_year: Option<u32>,
    params: &TimeMachineParams,
) -> Vec<TmRecommendation> {
    let mut out = Vec::new();

    // 1) Surface the compiler's own cost/optimization recommendations first.
    for r in analysis_recs.iter().take(4) {
        out.push(TmRecommendation {
            title: r.title.clone(),
            description: r.description.clone(),
            savings: if r.impact.is_empty() {
                "—".to_string()
            } else {
                r.impact.clone()
            },
            timing: "Immediate".to_string(),
            priority: format!("{:?}", r.priority).to_lowercase(),
        });
    }

    // 2) Budget-breach migration advice (derived from the projection itself).
    if let Some(break_year) = first_break_year {
        out.push(TmRecommendation {
            title: "Migrate to a lighter architecture".to_string(),
            description: format!(
                "Annual budget is exceeded by {}. Consider a smaller or quantized variant before then.",
                break_year
            ),
            savings: "60-70%".to_string(),
            timing: format!("Before {}", break_year),
            priority: "high".to_string(),
        });
    }

    // 3) Always-applicable quantization lever if not already recommended.
    if !out.iter().any(|r| r.title.to_lowercase().contains("quant")) {
        out.push(TmRecommendation {
            title: "Enable INT8 quantization".to_string(),
            description: "Reduce serving compute immediately with minimal accuracy impact."
                .to_string(),
            savings: "35-45%".to_string(),
            timing: "Immediate".to_string(),
            priority: "medium".to_string(),
        });
    }

    // 4) Hardware migration on the chosen track.
    if let Some(event) = params.hardware_track.next_gen_event(1) {
        out.push(TmRecommendation {
            title: format!("Migrate to {}", event.replace(" available", "")),
            description: "Next-gen hardware offers a better performance-per-dollar ratio."
                .to_string(),
            savings: "20-35%".to_string(),
            timing: "When available".to_string(),
            priority: "low".to_string(),
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics_with(cost: f64, co2: f64, params_count: u64) -> AllMetrics {
        let mut m = AllMetrics::default();
        m.training_cost_usd = cost;
        m.co2_kg = co2;
        m.total_parameters = params_count;
        m
    }

    #[test]
    fn projects_horizon_inclusive() {
        let m = metrics_with(120_000.0, 5000.0, 1_000_000);
        let params = TimeMachineParams {
            horizon_years: 5,
            ..Default::default()
        };
        let p = project_time_machine(&m, &[], 0.9, &params);
        assert_eq!(p.timeline.len(), 6); // years 0..=5 inclusive
        assert_eq!(p.carbon.len(), 6);
        assert_eq!(p.cost_breakdown.len(), 6);
    }

    #[test]
    fn base_cost_anchored_on_training_cost() {
        let m = metrics_with(120_000.0, 5000.0, 1_000_000);
        let p = project_time_machine(&m, &[], 1.0, &TimeMachineParams::default());
        // base monthly = 120000 / 12 = 10000
        assert!((p.summary.base_monthly_usd - 10_000.0).abs() < 1e-6);
        assert!((p.timeline[0].nominal - 10_000.0).abs() < 1.0);
    }

    #[test]
    fn growth_increases_cost_and_carbon() {
        let m = metrics_with(120_000.0, 5000.0, 1_000_000);
        let params = TimeMachineParams {
            growth_rate_pct: 100.0, // doubling/yr
            hardware_track: HardwareTrack::A100,
            ..Default::default()
        };
        let p = project_time_machine(&m, &[], 1.0, &params);
        assert!(p.timeline[1].nominal > p.timeline[0].nominal);
        assert!(p.carbon[2].baseline > p.carbon[0].baseline);
    }

    #[test]
    fn budget_breach_is_flagged() {
        let m = metrics_with(120_000.0, 5000.0, 1_000_000);
        let params = TimeMachineParams {
            growth_rate_pct: 100.0,
            annual_budget_usd: 150_000.0, // year0 = 120k < 150k, grows past it
            ..Default::default()
        };
        let p = project_time_machine(&m, &[], 1.0, &params);
        assert!(p.summary.first_break_year.is_some());
        assert!(p.timeline.iter().any(|t| t.breaking_point));
    }

    #[test]
    fn confidence_widens_band() {
        let m = metrics_with(120_000.0, 5000.0, 1_000_000);
        let high = project_time_machine(&m, &[], 1.0, &TimeMachineParams::default());
        let low = project_time_machine(&m, &[], 0.0, &TimeMachineParams::default());
        let high_spread = high.timeline[0].pessimistic - high.timeline[0].optimistic;
        let low_spread = low.timeline[0].pessimistic - low.timeline[0].optimistic;
        assert!(low_spread > high_spread);
    }

    #[test]
    fn fallback_when_no_cost() {
        let m = metrics_with(0.0, 0.0, 50_000_000);
        let p = project_time_machine(&m, &[], 1.0, &TimeMachineParams::default());
        assert!(p.summary.base_monthly_usd > 0.0);
        assert!(p.carbon[0].baseline >= 0.0);
    }
}
