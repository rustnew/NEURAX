//! Cost metrics utilities

use super::CostMetrics;

/// Format cost for display
pub fn format_cost(usd: f64) -> String {
    if usd >= 1_000_000.0 {
        format!("${:.2}M", usd / 1_000_000.0)
    } else if usd >= 1_000.0 {
        format!("${:.2}K", usd / 1_000.0)
    } else {
        format!("${:.2}", usd)
    }
}

/// Format energy for display
pub fn format_energy(kwh: f64) -> String {
    if kwh >= 1_000_000.0 {
        format!("{:.2} GWh", kwh / 1_000_000.0)
    } else if kwh >= 1_000.0 {
        format!("{:.2} MWh", kwh / 1_000.0)
    } else {
        format!("{:.2} kWh", kwh)
    }
}

/// Format CO2 for display
pub fn format_co2(kg: f64) -> String {
    if kg >= 1_000.0 {
        format!("{:.2} tonnes CO2", kg / 1_000.0)
    } else {
        format!("{:.2} kg CO2", kg)
    }
}

/// Get cost breakdown
pub fn cost_breakdown(metrics: &CostMetrics) -> CostBreakdown {
    let _total = metrics.training_cost_usd;
    
    CostBreakdown {
        compute_pct: 70.0,
        storage_pct: 10.0,
        network_pct: 10.0,
        other_pct: 10.0,
    }
}

#[derive(Debug, Clone)]
pub struct CostBreakdown {
    pub compute_pct: f64,
    pub storage_pct: f64,
    pub network_pct: f64,
    pub other_pct: f64,
}
