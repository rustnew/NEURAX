//! Report formatting utilities

use super::{ReportIR, Severity, Priority};
use std::fmt::Write;

/// Format report as Markdown (no panics, silently handles write errors)
pub fn format_markdown(report: &ReportIR) -> String {
    let mut output = String::new();
    
    // Helper macro for safe writes
    macro_rules! safe_write {
        ($buf:expr, $($arg:tt)*) => {
            let _ = write!($buf, $($arg)*);
        };
    }
    
    macro_rules! safe_writeln {
        ($buf:expr) => {
            let _ = writeln!($buf);
        };
        ($buf:expr, $($arg:tt)*) => {
            let _ = writeln!($buf, $($arg)*);
        };
    }
    
    // Header
    safe_writeln!(output, "# NEURAX Analysis Report");
    safe_writeln!(output);
    
    // Metadata
    safe_writeln!(output, "## Model Information");
    safe_writeln!(output, "| Property | Value |");
    safe_writeln!(output, "|----------|-------|");
    safe_writeln!(output, "| Model Name | {} |", report.metadata.model_name);
    safe_writeln!(output, "| Model Type | {} |", report.metadata.model_type);
    safe_writeln!(output, "| Total Parameters | {} |", format_param_count(report.metrics.total_parameters));
    safe_writeln!(output, "| Number of Layers | {} |", report.metrics.num_layers);
    safe_writeln!(output, "| Generated | {} |", report.metadata.generated_at.format("%Y-%m-%d %H:%M:%S UTC"));
    safe_writeln!(output);
    
    // Structure Metrics
    safe_writeln!(output, "## Structure Metrics");
    safe_writeln!(output, "| Metric | Value |");
    safe_writeln!(output, "|--------|-------|");
    safe_writeln!(output, "| Total Parameters | {} |", format_param_count(report.metrics.total_parameters));
    safe_writeln!(output, "| Number of Layers | {} |", report.metrics.num_layers);
    safe_writeln!(output, "| Graph Depth | {} |", report.metrics.graph_depth);
    safe_writeln!(output, "| Total Operations | {} |", report.metrics.total_operations);
    safe_writeln!(output);
    
    // Compute Metrics
    safe_writeln!(output, "## Compute Metrics");
    safe_writeln!(output, "| Metric | Value |");
    safe_writeln!(output, "|--------|-------|");
    safe_writeln!(output, "| Total FLOPs | {} |", format_flops(report.metrics.total_flops));
    safe_writeln!(output, "| Forward FLOPs | {} |", format_flops(report.metrics.forward_flops));
    safe_writeln!(output, "| Backward FLOPs | {} |", format_flops(report.metrics.backward_flops));
    safe_writeln!(output, "| FLOPs per Token | {:.2} |", report.metrics.flops_per_token);
    safe_writeln!(output, "| Arithmetic Intensity | {:.2} FLOPs/byte |", report.metrics.arithmetic_intensity);
    safe_writeln!(output);
    
    // Memory Metrics
    safe_writeln!(output, "## Memory Metrics");
    safe_writeln!(output, "| Metric | Value |");
    safe_writeln!(output, "|--------|-------|");
    safe_writeln!(output, "| Peak VRAM | {} |", format_bytes(report.metrics.peak_vram_bytes));
    safe_writeln!(output, "| Parameter Memory | {} |", format_bytes(report.metrics.parameter_memory_bytes));
    safe_writeln!(output, "| Activation Memory | {} |", format_bytes(report.metrics.activation_memory_bytes));
    safe_writeln!(output, "| Gradient Memory | {} |", format_bytes(report.metrics.gradient_memory_bytes));
    safe_writeln!(output, "| Optimizer State | {} |", format_bytes(report.metrics.optimizer_state_bytes));
    safe_writeln!(output, "| Max Batch Size | {} |", report.metrics.max_batch_size_fit);
    safe_writeln!(output);
    
    // Performance Metrics
    safe_writeln!(output, "## Performance Metrics");
    safe_writeln!(output, "| Metric | Value |");
    safe_writeln!(output, "|--------|-------|");
    safe_writeln!(output, "| Latency per Step | {} |", format_latency(report.metrics.latency_ms));
    safe_writeln!(output, "| Throughput | {} |", format_throughput(report.metrics.throughput_tokens_per_s));
    safe_writeln!(output, "| GPU Utilization | {:.1}% |", report.metrics.gpu_utilization * 100.0);
    safe_writeln!(output, "| Bottleneck | {} |", report.metrics.bottleneck);
    safe_writeln!(output);
    
    // Parallelism Metrics
    safe_writeln!(output, "## Parallelism Metrics");
    safe_writeln!(output, "| Metric | Value |");
    safe_writeln!(output, "|--------|-------|");
    safe_writeln!(output, "| Data Parallel Efficiency | {:.1}% |", report.metrics.data_parallel_efficiency * 100.0);
    safe_writeln!(output, "| Communication Overhead | {:.1}% |", report.metrics.communication_overhead * 100.0);
    safe_writeln!(output, "| Optimal GPU Count | {} |", report.metrics.optimal_gpu_count);
    safe_writeln!(output);
    
    // Cost Metrics
    safe_writeln!(output, "## Cost Metrics");
    safe_writeln!(output, "| Metric | Value |");
    safe_writeln!(output, "|--------|-------|");
    safe_writeln!(output, "| Training Cost | {} |", format_cost(report.metrics.training_cost_usd));
    safe_writeln!(output, "| Training Time | {:.1} hours |", report.metrics.training_time_hours);
    safe_writeln!(output, "| Energy Consumption | {} |", format_energy(report.metrics.energy_kwh));
    safe_writeln!(output, "| CO₂ Emissions | {} |", format_co2(report.metrics.co2_kg));
    safe_writeln!(output, "| Cost per Million Tokens | ${:.2} |", report.metrics.cost_per_million_tokens_usd);
    safe_writeln!(output);
    
    // Diagnostics
    if !report.diagnostics.is_empty() {
        safe_writeln!(output, "## Diagnostics");
        safe_writeln!(output);
        for diag in &report.diagnostics {
            let severity = match diag.severity {
                Severity::Critical => "🔴 **CRITICAL**",
                Severity::Warning => "🟡 **WARNING**",
                Severity::Info => "🔵 **INFO**",
                Severity::Hint => "💡 **HINT**",
            };
            safe_writeln!(output, "### {} - {}", severity, diag.message);
            if let Some(ref suggestion) = diag.suggestion {
                safe_writeln!(output, "**Suggestion:** {}", suggestion);
            }
            safe_writeln!(output);
        }
    }
    
    // Recommendations
    if !report.recommendations.is_empty() {
        safe_writeln!(output, "## Recommendations");
        safe_writeln!(output);
        for rec in &report.recommendations {
            let priority = match rec.priority {
                Priority::High => "🔥 High",
                Priority::Medium => "⚡ Medium",
                Priority::Low => "💡 Low",
            };
            safe_writeln!(output, "### {} - {}", rec.title, priority);
            safe_writeln!(output, "{}", rec.description);
            safe_writeln!(output, "**Impact:** {}", rec.impact);
            safe_writeln!(output);
        }
    }
    
    // Warnings
    if !report.warnings.is_empty() {
        safe_writeln!(output, "## Warnings");
        safe_writeln!(output);
        for warning in &report.warnings {
            safe_writeln!(output, "- ⚠️ {}", warning);
        }
        safe_writeln!(output);
    }
    
    output
}

/// Format report as JSON (no panics, returns empty string on error)
pub fn format_json(report: &ReportIR) -> String {
    serde_json::to_string_pretty(&report.metrics).unwrap_or_else(|e| {
        tracing::error!("Failed to serialize JSON: {}", e);
        String::from("{\"error\": \"Failed to serialize metrics\"}")
    })
}

/// Format report as complete JSON output with all 35 metrics
pub fn format_json_output(report: &ReportIR, input_file: &str, analysis_time_ms: u64) -> String {
    let json_output = super::JsonOutput::from_report(report, input_file, analysis_time_ms);
    json_output.to_json().unwrap_or_else(|e| {
        tracing::error!("Failed to serialize JSON output: {}", e);
        String::from("{\"error\": \"Failed to serialize output\"}")
    })
}

// Helper formatting functions
fn format_param_count(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.2}B", count as f64 / 1e9)
    } else if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1e6)
    } else if count >= 1_000 {
        format!("{:.2}K", count as f64 / 1e3)
    } else {
        count.to_string()
    }
}

fn format_flops(flops: f64) -> String {
    if flops >= 1e15 {
        format!("{:.2} PFLOPs", flops / 1e15)
    } else if flops >= 1e12 {
        format!("{:.2} TFLOPs", flops / 1e12)
    } else if flops >= 1e9 {
        format!("{:.2} GFLOPs", flops / 1e9)
    } else if flops >= 1e6 {
        format!("{:.2} MFLOPs", flops / 1e6)
    } else {
        format!("{:.0} FLOPs", flops)
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn format_latency(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2} s", ms / 1000.0)
    } else if ms >= 1.0 {
        format!("{:.2} ms", ms)
    } else {
        format!("{:.2} µs", ms * 1000.0)
    }
}

fn format_throughput(tokens_per_s: f64) -> String {
    if tokens_per_s >= 1e6 {
        format!("{:.2} M tokens/s", tokens_per_s / 1e6)
    } else if tokens_per_s >= 1e3 {
        format!("{:.2} K tokens/s", tokens_per_s / 1e3)
    } else {
        format!("{:.0} tokens/s", tokens_per_s)
    }
}

fn format_cost(usd: f64) -> String {
    if usd >= 1_000_000.0 {
        format!("${:.2}M", usd / 1_000_000.0)
    } else if usd >= 1_000.0 {
        format!("${:.2}K", usd / 1_000.0)
    } else {
        format!("${:.2}", usd)
    }
}

fn format_energy(kwh: f64) -> String {
    if kwh >= 1_000.0 {
        format!("{:.2} MWh", kwh / 1_000.0)
    } else {
        format!("{:.2} kWh", kwh)
    }
}

fn format_co2(kg: f64) -> String {
    if kg >= 1_000.0 {
        format!("{:.2} tonnes CO₂", kg / 1_000.0)
    } else {
        format!("{:.2} kg CO₂", kg)
    }
}
