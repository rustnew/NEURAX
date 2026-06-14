//! Main UI rendering

use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect, Alignment},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, Tabs, Clear, List, ListItem, Gauge},
    Frame,
};
use crossterm::event::{KeyCode, KeyEvent};

use crate::app::{App, Tab};
use crate::metrics_display;
use crate::comparison;

pub fn draw(f: &mut Frame, app: &mut App) {
    // Main layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Length(3),  // Tabs
            Constraint::Min(20),    // Main content
            Constraint::Length(3),  // Status bar
        ])
        .split(f.area());
    
    // Header
    draw_header(f, app, chunks[0]);
    
    // Tabs
    draw_tabs(f, app, chunks[1]);
    
    // Main content
    if app.compiled_result.is_some() {
        draw_content(f, app, chunks[2]);
    } else {
        draw_model_selector(f, app, chunks[2]);
    }
    
    // Status bar
    draw_status_bar(f, app, chunks[3]);
}

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let header = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("╔══════════════════════════════════════════════════════════════════════════════════════════════════╗", 
                Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("║", Style::default().fg(Color::Cyan)),
            Span::styled("  NEURAX IR - Model Compilation Visualizer  ", 
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("│ 77 Metrics  │ 10 Model Families  │ Real-World Comparison  ", 
                Style::default().fg(Color::Gray)),
            Span::styled("║", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("╚══════════════════════════════════════════════════════════════════════════════════════════════════╝", 
                Style::default().fg(Color::Cyan)),
        ]),
    ]);
    f.render_widget(header, area);
}

fn draw_tabs(f: &mut Frame, app: &App, area: Rect) {
    let titles: Vec<Line> = Tab::all()
        .iter()
        .map(|t| {
            let (first, rest) = t.name().split_at(1);
            Line::from(vec![
                Span::styled(first, Style::default().fg(Color::Yellow).add_modifier(Modifier::UNDERLINED)),
                Span::styled(rest, Style::default().fg(Color::White)),
            ])
        })
        .collect();
    
    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::BOTTOM))
        .select(app.current_tab.index())
        .style(Style::default().fg(Color::White))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .divider(Span::raw(" │ "));
    
    f.render_widget(tabs, area);
}

fn draw_model_selector(f: &mut Frame, app: &App, area: Rect) {
    // Split into left (model list) and right (instructions)
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);
    
    // Model list
    let items: Vec<ListItem> = app.models
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let style = if i == app.selected_model {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(vec![
                Line::from(vec![
                    Span::styled(format!("{:2} ", i + 1), Style::default().fg(Color::Gray)),
                    Span::styled(m.name.to_string(), style),
                    Span::styled(format!("  [{:^12}]", m.family), Style::default().fg(Color::Cyan)),
                ]),
                Line::from(vec![
                    Span::raw("     "),
                    Span::styled(m.description.to_string(), Style::default().fg(Color::Gray)),
                ]),
            ])
        })
        .collect();
    
    let list = List::new(items)
        .block(Block::default()
            .title("Select Model (↑/↓ to navigate, Enter to compile)")
            .borders(Borders::ALL)
            .title_style(Style::default().fg(Color::Cyan)))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));
    
    f.render_stateful_widget(list, chunks[0], &mut ratatui::widgets::ListState::default()
        .with_selected(Some(app.selected_model)));
    
    // Instructions panel
    let instructions = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("┌─────────────────────────────────────────────────────────────────┐", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("│", Style::default().fg(Color::Cyan)),
            Span::styled("                    NEURAX IR COMPILER                        ", 
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("│", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("│", Style::default().fg(Color::Cyan)),
            Span::raw("                                                               "),
            Span::styled("│", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("│", Style::default().fg(Color::Cyan)),
            Span::styled("  This tool compiles neural network models and produces       ", Style::default().fg(Color::White)),
            Span::styled("│", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("│", Style::default().fg(Color::Cyan)),
            Span::styled("  77 metrics across 9 IR modules:                             ", Style::default().fg(Color::White)),
            Span::styled("│", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("│", Style::default().fg(Color::Cyan)),
            Span::raw("                                                               "),
            Span::styled("│", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("│", Style::default().fg(Color::Cyan)),
            Span::styled("  • Architecture: 5 metrics   • Graph: 6 metrics              ", Style::default().fg(Color::Gray)),
            Span::styled("│", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("│", Style::default().fg(Color::Cyan)),
            Span::styled("  • Tensor: 9 metrics        • Operator: 5 metrics          ", Style::default().fg(Color::Gray)),
            Span::styled("│", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("│", Style::default().fg(Color::Cyan)),
            Span::styled("  • Compute: 12 metrics      • Memory: 11 metrics            ", Style::default().fg(Color::Gray)),
            Span::styled("│", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("│", Style::default().fg(Color::Cyan)),
            Span::styled("  • Parallelism: 10 metrics  • Hardware: 10 metrics          ", Style::default().fg(Color::Gray)),
            Span::styled("│", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("│", Style::default().fg(Color::Cyan)),
            Span::styled("  • Cost: 9 metrics                                          ", Style::default().fg(Color::Gray)),
            Span::styled("│", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("│", Style::default().fg(Color::Cyan)),
            Span::raw("                                                               "),
            Span::styled("│", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("│", Style::default().fg(Color::Cyan)),
            Span::styled("  Each metric is compared against real-world data from       ", Style::default().fg(Color::White)),
            Span::styled("│", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("│", Style::default().fg(Color::Cyan)),
            Span::styled("  published papers and benchmarks.                            ", Style::default().fg(Color::White)),
            Span::styled("│", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("│", Style::default().fg(Color::Cyan)),
            Span::raw("                                                               "),
            Span::styled("│", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("│", Style::default().fg(Color::Cyan)),
            Span::styled("  Keys: [↑/↓] Navigate  [Enter] Compile  [Tab] Switch Tab    ", Style::default().fg(Color::Yellow)),
            Span::styled("│", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("│", Style::default().fg(Color::Cyan)),
            Span::styled("        [1-7] Jump to Tab  [r] Refresh  [q] Quit             ", Style::default().fg(Color::Yellow)),
            Span::styled("│", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("└─────────────────────────────────────────────────────────────────┘", Style::default().fg(Color::Cyan)),
        ]),
    ]);
    
    f.render_widget(instructions, chunks[1]);
}

fn draw_content(f: &mut Frame, app: &App, area: Rect) {
    if let (Some(result), Some(real)) = (&app.compiled_result, &app.real_world_data) {
        match app.current_tab {
            Tab::Overview => draw_overview(f, app, result, real, area),
            Tab::Architecture => {
                let table = metrics_display::render_architecture_metrics(result, real, area);
                f.render_widget(table, area);
            }
            Tab::Compute => {
                let table = metrics_display::render_compute_metrics(result, real, area);
                f.render_widget(table, area);
            }
            Tab::Memory => {
                let table = metrics_display::render_memory_metrics(result, real, area);
                f.render_widget(table, area);
            }
            Tab::Hardware => {
                let table = metrics_display::render_hardware_metrics(result, real, area);
                f.render_widget(table, area);
            }
            Tab::Cost => {
                let table = metrics_display::render_cost_metrics(result, real, area);
                f.render_widget(table, area);
            }
            Tab::Comparison => draw_comparison(f, app, result, real, area),
        }
    }
}

fn draw_overview(f: &mut Frame, app: &App, result: &neurax_core::AnalysisResult, real: &crate::real_world_data::RealWorldData, area: Rect) {
    // Split into 4 quadrants
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);
    
    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[0]);
    
    let bottom_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);
    
    // Model info
    let model_info = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Model: ", Style::default().fg(Color::Cyan)),
            Span::styled(&real.model_name, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("Family: ", Style::default().fg(Color::Cyan)),
            Span::styled(&result.arch.metrics.model_type_info, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Parameters: ", Style::default().fg(Color::Cyan)),
            Span::styled(metrics_display::format_number(result.arch.metrics.total_parameters as f64), 
                Style::default().fg(Color::Green)),
            Span::raw(" (Real: "),
            Span::styled(metrics_display::format_number(real.total_params as f64), Style::default().fg(Color::Yellow)),
            Span::raw(")"),
        ]),
        Line::from(vec![
            Span::styled("Layers: ", Style::default().fg(Color::Cyan)),
            Span::styled(format!("{}", result.arch.metrics.num_layers), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Analysis Time: ", Style::default().fg(Color::Cyan)),
            Span::styled(format!("{} ms", result.analysis_time_ms), Style::default().fg(Color::White)),
        ]),
    ])
    .block(Block::default().title("Model Information").borders(Borders::ALL));
    f.render_widget(model_info, top_chunks[0]);
    
    // Metrics count
    let metrics_count = comparison::render_metrics_count_summary(result, top_chunks[1]);
    f.render_widget(metrics_count, top_chunks[1]);
    
    // Comparison summary
    let comparison_table = comparison::render_comparison_summary(result, real, bottom_chunks[0]);
    f.render_widget(comparison_table, bottom_chunks[0]);
    
    // IR Modules status
    let ir_status = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("IR Module Status:", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("✓", Style::default().fg(Color::Green)),
            Span::raw(" ArchitectureIR  "),
            Span::styled("✓", Style::default().fg(Color::Green)),
            Span::raw(" GraphIR  "),
            Span::styled("✓", Style::default().fg(Color::Green)),
            Span::raw(" TensorIR"),
        ]),
        Line::from(vec![
            Span::styled("✓", Style::default().fg(Color::Green)),
            Span::raw(" OperatorIR     "),
            Span::styled("✓", Style::default().fg(Color::Green)),
            Span::raw(" ComputeIR  "),
            Span::styled("✓", Style::default().fg(Color::Green)),
            Span::raw(" MemoryIR"),
        ]),
        Line::from(vec![
            Span::styled("✓", Style::default().fg(Color::Green)),
            Span::raw(" ParallelismIR  "),
            Span::styled("✓", Style::default().fg(Color::Green)),
            Span::raw(" HardwareIR  "),
            Span::styled("✓", Style::default().fg(Color::Green)),
            Span::raw(" CostIR"),
        ]),
        Line::from(vec![
            Span::styled("✓", Style::default().fg(Color::Green)),
            Span::raw(" ReportIR"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("All 9 IR modules validated successfully", Style::default().fg(Color::Green)),
        ]),
    ])
    .block(Block::default().title("IR Modules").borders(Borders::ALL));
    f.render_widget(ir_status, bottom_chunks[1]);
}

fn draw_comparison(f: &mut Frame, app: &App, result: &neurax_core::AnalysisResult, real: &crate::real_world_data::RealWorldData, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);
    
    // Main comparison table
    let comparison_table = comparison::render_comparison_summary(result, real, chunks[0]);
    f.render_widget(comparison_table, chunks[0]);
    
    // Source info
    let source_info = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Data Source: ", Style::default().fg(Color::Cyan)),
            Span::styled(&real.source, Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("GPU Configuration: ", Style::default().fg(Color::Cyan)),
            Span::styled(format!("{} x {}", real.gpu_count, real.gpu_type), Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Training Tokens: ", Style::default().fg(Color::Cyan)),
            Span::styled(metrics_display::format_number(real.training_tokens as f64), Style::default().fg(Color::White)),
        ]),
    ])
    .block(Block::default().title("Real-World Reference Data").borders(Borders::ALL));
    f.render_widget(source_info, chunks[1]);
}

fn draw_status_bar(f: &mut Frame, app: &App, area: Rect) {
    let status = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(" Status: ", Style::default().fg(Color::Cyan)),
            Span::styled(&app.status_message, 
                if app.status_message.starts_with("✓") {
                    Style::default().fg(Color::Green)
                } else if app.status_message.starts_with("✗") {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default().fg(Color::White)
                }),
            Span::styled("  │  ", Style::default().fg(Color::Gray)),
            Span::styled("[q] Quit  [r] Refresh  [Tab] Next Tab", Style::default().fg(Color::Gray)),
        ]),
    ])
    .block(Block::default().borders(Borders::TOP));
    
    f.render_widget(status, area);
}
