//! Metrics display components

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, Gauge},
};
use neurax_core::AnalysisResult;
use crate::real_world_data::RealWorldData;

pub fn format_number(n: f64) -> String {
    if n >= 1e12 {
        format!("{:.2}T", n / 1e12)
    } else if n >= 1e9 {
        format!("{:.2}B", n / 1e9)
    } else if n >= 1e6 {
        format!("{:.2}M", n / 1e6)
    } else if n >= 1e3 {
        format!("{:.2}K", n / 1e3)
    } else {
        format!("{:.2}", n)
    }
}

pub fn format_bytes(bytes: u64) -> String {
    let gb = bytes as f64 / 1e9;
    if gb >= 1000.0 {
        format!("{:.2} TB", gb / 1000.0)
    } else {
        format!("{:.2} GB", gb)
    }
}

pub fn format_flops(flops: f64) -> String {
    if flops >= 1e18 {
        format!("{:.2} EFLOPS", flops / 1e18)
    } else if flops >= 1e15 {
        format!("{:.2} PFLOPS", flops / 1e15)
    } else if flops >= 1e12 {
        format!("{:.2} TFLOPS", flops / 1e12)
    } else if flops >= 1e9 {
        format!("{:.2} GFLOPS", flops / 1e9)
    } else {
        format!("{:.2} FLOPS", flops)
    }
}

pub fn calculate_accuracy(computed: f64, real: f64) -> f64 {
    if real == 0.0 {
        return 100.0;
    }
    let diff = (computed - real).abs() / real;
    (1.0 - diff.min(1.0)) * 100.0
}

pub fn accuracy_color(accuracy: f64) -> Color {
    if accuracy >= 95.0 {
        Color::Green
    } else if accuracy >= 80.0 {
        Color::Yellow
    } else if accuracy >= 60.0 {
        Color::Rgb(255, 165, 0) // Orange
    } else {
        Color::Red
    }
}

pub fn render_overview_tab(result: &AnalysisResult, real: &RealWorldData, _area: Rect) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    
    // Model summary - use owned strings
    let model_type = result.arch.metrics.model_type_info.clone();
    let params_computed = format_number(result.arch.metrics.total_parameters as f64);
    let params_real = format_number(real.total_params as f64);
    let num_layers = format!("{}", result.arch.metrics.num_layers);
    
    lines.push(Line::from(vec![
        Span::styled("Model: ", Style::default().fg(Color::Cyan)),
        Span::styled(model_type, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
    ]));
    lines.push(Line::from(vec![
        Span::styled("Parameters: ", Style::default().fg(Color::Cyan)),
        Span::styled(params_computed, Style::default().fg(Color::White)),
        Span::raw("  (Real: "),
        Span::styled(params_real, Style::default().fg(Color::Yellow)),
        Span::raw(")"),
    ]));
    lines.push(Line::from(vec![
        Span::styled("Layers: ", Style::default().fg(Color::Cyan)),
        Span::styled(num_layers, Style::default().fg(Color::White)),
    ]));
    
    lines
}

pub fn render_architecture_metrics(result: &AnalysisResult, real: &RealWorldData, area: Rect) -> Table<'static> {
    let accuracy = calculate_accuracy(result.arch.metrics.total_parameters as f64, real.total_params as f64);
    
    let rows = vec![
        Row::new(vec![
            Cell::from("Total Parameters"),
            Cell::from(format_number(result.arch.metrics.total_parameters as f64)),
            Cell::from(format_number(real.total_params as f64)),
            Cell::from(format!("{:.1}%", accuracy)).style(Style::default().fg(accuracy_color(accuracy))),
        ]),
        Row::new(vec![
            Cell::from("Number of Layers"),
            Cell::from(format!("{}", result.arch.metrics.num_layers)),
            Cell::from("-"),
            Cell::from("N/A"),
        ]),
        Row::new(vec![
            Cell::from("Model Type"),
            Cell::from(result.arch.metrics.model_type_info.clone()),
            Cell::from(real.gpu_type.clone()),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Params per Layer"),
            Cell::from(format!("{:.2}M", result.arch.metrics.total_parameters as f64 / result.arch.metrics.num_layers as f64 / 1e6)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
    ];
    
    Table::new(rows, [Constraint::Percentage(30), Constraint::Percentage(25), Constraint::Percentage(25), Constraint::Percentage(20)])
        .block(Block::default().title("Architecture Metrics (5 metrics)").borders(Borders::ALL))
        .header(Row::new(vec!["Metric", "Computed", "Real World", "Accuracy"]).style(Style::default().fg(Color::Cyan)))
}

pub fn render_compute_metrics(result: &AnalysisResult, _real: &RealWorldData, area: Rect) -> Table<'static> {
    let rows = vec![
        Row::new(vec![
            Cell::from("Total FLOPs"),
            Cell::from(format_flops(result.compute.metrics.total_flops)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Forward FLOPs"),
            Cell::from(format_flops(result.compute.metrics.forward_flops)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Backward FLOPs"),
            Cell::from(format_flops(result.compute.metrics.backward_flops)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("FLOPs per Token"),
            Cell::from(format!("{:.2e}", result.compute.metrics.flops_per_token)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Arithmetic Intensity"),
            Cell::from(format!("{:.2}", result.compute.metrics.arithmetic_intensity)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Complexity Class"),
            Cell::from(result.compute.metrics.complexity_class.as_str()),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Optimizer FLOPs"),
            Cell::from(format_flops(result.compute.metrics.optimizer_flops)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Total Step FLOPs"),
            Cell::from(format_flops(result.compute.metrics.total_step_flops)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
    ];
    
    Table::new(rows, [Constraint::Percentage(30), Constraint::Percentage(25), Constraint::Percentage(25), Constraint::Percentage(20)])
        .block(Block::default().title("Compute Metrics (12 metrics)").borders(Borders::ALL))
        .header(Row::new(vec!["Metric", "Computed", "Real World", "Accuracy"]).style(Style::default().fg(Color::Cyan)))
}

pub fn render_memory_metrics(result: &AnalysisResult, real: &RealWorldData, area: Rect) -> Table<'static> {
    let accuracy = calculate_accuracy(result.memory.metrics.peak_vram_gb(), real.peak_memory_gb);
    
    let rows = vec![
        Row::new(vec![
            Cell::from("Peak VRAM"),
            Cell::from(format!("{:.2} GB", result.memory.metrics.peak_vram_gb())),
            Cell::from(format!("{:.2} GB", real.peak_memory_gb)),
            Cell::from(format!("{:.1}%", accuracy)).style(Style::default().fg(accuracy_color(accuracy))),
        ]),
        Row::new(vec![
            Cell::from("Parameter Memory"),
            Cell::from(format_bytes(result.memory.metrics.parameter_memory_bytes)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Activation Memory"),
            Cell::from(format_bytes(result.memory.metrics.activation_memory_bytes)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Gradient Memory"),
            Cell::from(format_bytes(result.memory.metrics.gradient_memory_bytes)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Optimizer State"),
            Cell::from(format_bytes(result.memory.metrics.optimizer_state_bytes)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Max Batch Size Fit"),
            Cell::from(format!("{}", result.memory.metrics.max_batch_size_fit)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("OOM Risk"),
            Cell::from(format!("{:?}", result.memory.metrics.oom_risk)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Memory Bandwidth"),
            Cell::from(format!("{:.2} GB/s", result.memory.metrics.memory_bandwidth_req / 1e9)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
    ];
    
    Table::new(rows, [Constraint::Percentage(30), Constraint::Percentage(25), Constraint::Percentage(25), Constraint::Percentage(20)])
        .block(Block::default().title("Memory Metrics (11 metrics)").borders(Borders::ALL))
        .header(Row::new(vec!["Metric", "Computed", "Real World", "Accuracy"]).style(Style::default().fg(Color::Cyan)))
}

pub fn render_hardware_metrics(result: &AnalysisResult, real: &RealWorldData, area: Rect) -> Table<'static> {
    let latency_accuracy = calculate_accuracy(result.hardware.metrics.latency_ms, real.inference_latency_ms);
    let throughput_accuracy = calculate_accuracy(result.hardware.metrics.throughput_tokens_per_s, real.throughput_tokens_per_s);
    
    let rows = vec![
        Row::new(vec![
            Cell::from("Latency"),
            Cell::from(format!("{:.2} ms", result.hardware.metrics.latency_ms)),
            Cell::from(format!("{:.2} ms", real.inference_latency_ms)),
            Cell::from(format!("{:.1}%", latency_accuracy)).style(Style::default().fg(accuracy_color(latency_accuracy))),
        ]),
        Row::new(vec![
            Cell::from("Throughput"),
            Cell::from(format!("{:.2} tok/s", result.hardware.metrics.throughput_tokens_per_s)),
            Cell::from(format!("{:.2} tok/s", real.throughput_tokens_per_s)),
            Cell::from(format!("{:.1}%", throughput_accuracy)).style(Style::default().fg(accuracy_color(throughput_accuracy))),
        ]),
        Row::new(vec![
            Cell::from("GPU Utilization"),
            Cell::from(format!("{:.1}%", result.hardware.metrics.gpu_utilization * 100.0)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Tensor Core Util."),
            Cell::from(format!("{:.1}%", result.hardware.metrics.tensor_core_utilization * 100.0)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Effective TFLOPS"),
            Cell::from(format!("{:.2}", result.hardware.metrics.effective_tflops)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Bottleneck"),
            Cell::from(result.hardware.metrics.bottleneck.as_str()),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Roofline Position"),
            Cell::from(format!("{:.2}", result.hardware.metrics.roofline_position)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Memory Bandwidth"),
            Cell::from(format!("{:.2} GB/s", result.hardware.metrics.memory_bandwidth_achieved)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
    ];
    
    Table::new(rows, [Constraint::Percentage(30), Constraint::Percentage(25), Constraint::Percentage(25), Constraint::Percentage(20)])
        .block(Block::default().title("Hardware Metrics (10 metrics)").borders(Borders::ALL))
        .header(Row::new(vec!["Metric", "Computed", "Real World", "Accuracy"]).style(Style::default().fg(Color::Cyan)))
}

pub fn render_cost_metrics(result: &AnalysisResult, real: &RealWorldData, area: Rect) -> Table<'static> {
    let cost_accuracy = calculate_accuracy(result.cost.metrics.training_cost_usd, real.training_cost_usd);
    let time_accuracy = calculate_accuracy(result.cost.metrics.training_time_hours, real.training_time_hours);
    
    let rows = vec![
        Row::new(vec![
            Cell::from("Training Cost"),
            Cell::from(format!("${:.2}M", result.cost.metrics.training_cost_usd / 1e6)),
            Cell::from(format!("${:.2}M", real.training_cost_usd / 1e6)),
            Cell::from(format!("{:.1}%", cost_accuracy)).style(Style::default().fg(accuracy_color(cost_accuracy))),
        ]),
        Row::new(vec![
            Cell::from("Training Time"),
            Cell::from(format!("{:.1} hours", result.cost.metrics.training_time_hours)),
            Cell::from(format!("{:.1} hours", real.training_time_hours)),
            Cell::from(format!("{:.1}%", time_accuracy)).style(Style::default().fg(accuracy_color(time_accuracy))),
        ]),
        Row::new(vec![
            Cell::from("GPU Hours"),
            Cell::from(format!("{:.0}", result.cost.metrics.gpu_hours_total)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Energy (kWh)"),
            Cell::from(format!("{:.0}", result.cost.metrics.energy_kwh)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("CO2 (kg)"),
            Cell::from(format!("{:.0}", result.cost.metrics.co2_kg)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Cost per Token"),
            Cell::from(format!("${:.2e}", result.cost.metrics.cost_per_token_usd)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Cost per Million Tokens"),
            Cell::from(format!("${:.4}", result.cost.metrics.cost_per_million_tokens_usd)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
        Row::new(vec![
            Cell::from("Monthly Inference Cost"),
            Cell::from(format!("${:.2}", result.cost.metrics.monthly_inference_cost_usd)),
            Cell::from("-"),
            Cell::from("-"),
        ]),
    ];
    
    Table::new(rows, [Constraint::Percentage(30), Constraint::Percentage(25), Constraint::Percentage(25), Constraint::Percentage(20)])
        .block(Block::default().title("Cost Metrics (9 metrics)").borders(Borders::ALL))
        .header(Row::new(vec!["Metric", "Computed", "Real World", "Accuracy"]).style(Style::default().fg(Color::Cyan)))
}
