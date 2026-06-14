//! Neurax TUI - Terminal User Interface for model compilation visualization
//!
//! A clean, organized interface to visualize all 35+ metrics and compare with real-world data

mod app;
mod ui;
mod model_selector;
mod metrics_display;
mod comparison;
mod real_world_data;

use color_eyre::Result;
use ratatui::{
    backend::CrosstermBackend,
    Terminal,
};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::io;

use app::App;

fn main() -> Result<()> {
    // Setup terminal
    color_eyre::install()?;
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and run
    let app = App::new();
    let res = run_app(&mut terminal, app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        eprintln!("Error: {:?}", err);
    }

    Ok(())
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    mut app: App,
) -> Result<()> {
    loop {
        terminal.draw(|f| ui::draw(f, &mut app))?;

        if let Event::Key(key) = event::read()? {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => {
                    return Ok(());
                }
                KeyCode::Char('c') if key.modifiers == KeyModifiers::CONTROL => {
                    return Ok(());
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    app.previous_model();
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    app.next_model();
                }
                KeyCode::Enter => {
                    app.compile_selected_model();
                }
                KeyCode::Tab => {
                    app.next_tab();
                }
                KeyCode::BackTab => {
                    app.previous_tab();
                }
                KeyCode::Char('1') => {
                    app.select_tab(0);
                }
                KeyCode::Char('2') => {
                    app.select_tab(1);
                }
                KeyCode::Char('3') => {
                    app.select_tab(2);
                }
                KeyCode::Char('4') => {
                    app.select_tab(3);
                }
                KeyCode::Char('r') => {
                    app.refresh();
                }
                _ => {}
            }
        }
    }
}
