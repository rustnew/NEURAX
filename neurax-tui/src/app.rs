//! Main application state and logic

use neurax_core::AnalysisResult;

use crate::model_selector::{Model, MODEL_LIST};
use crate::real_world_data::RealWorldData;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Overview,
    Architecture,
    Compute,
    Memory,
    Hardware,
    Cost,
    Comparison,
}

impl Tab {
    pub fn all() -> &'static [Tab] {
        &[Tab::Overview, Tab::Architecture, Tab::Compute, Tab::Memory, Tab::Hardware, Tab::Cost, Tab::Comparison]
    }
    
    pub fn name(&self) -> &'static str {
        match self {
            Tab::Overview => "Overview",
            Tab::Architecture => "Architecture",
            Tab::Compute => "Compute",
            Tab::Memory => "Memory",
            Tab::Hardware => "Hardware",
            Tab::Cost => "Cost",
            Tab::Comparison => "Comparison",
        }
    }
    
    pub fn index(&self) -> usize {
        match self {
            Tab::Overview => 0,
            Tab::Architecture => 1,
            Tab::Compute => 2,
            Tab::Memory => 3,
            Tab::Hardware => 4,
            Tab::Cost => 5,
            Tab::Comparison => 6,
        }
    }
}

pub struct App {
    pub running: bool,
    pub current_tab: Tab,
    pub models: Vec<Model>,
    pub selected_model: usize,
    pub compiled_result: Option<AnalysisResult>,
    pub real_world_data: Option<RealWorldData>,
    pub status_message: String,
    pub scroll_offset: u16,
}

impl App {
    pub fn new() -> Self {
        Self {
            running: true,
            current_tab: Tab::Overview,
            models: MODEL_LIST.to_vec(),
            selected_model: 0,
            compiled_result: None,
            real_world_data: None,
            status_message: "Select a model and press Enter to compile".to_string(),
            scroll_offset: 0,
        }
    }
    
    pub fn next_model(&mut self) {
        if self.selected_model < self.models.len() - 1 {
            self.selected_model += 1;
        }
    }
    
    pub fn previous_model(&mut self) {
        if self.selected_model > 0 {
            self.selected_model -= 1;
        }
    }
    
    pub fn next_tab(&mut self) {
        let tabs = Tab::all();
        let current_idx = self.current_tab.index();
        self.current_tab = tabs[(current_idx + 1) % tabs.len()];
    }
    
    pub fn previous_tab(&mut self) {
        let tabs = Tab::all();
        let current_idx = self.current_tab.index();
        self.current_tab = tabs[(current_idx + tabs.len() - 1) % tabs.len()];
    }
    
    pub fn select_tab(&mut self, idx: usize) {
        let tabs = Tab::all();
        if idx < tabs.len() {
            self.current_tab = tabs[idx];
        }
    }
    
    pub fn compile_selected_model(&mut self) {
        let model = &self.models[self.selected_model];
        self.status_message = format!("Compiling {}...", model.name);
        
        match neurax_core::analyze_json(model.json_content) {
            Ok(result) => {
                self.compiled_result = Some(result);
                self.real_world_data = Some(RealWorldData::for_model(&model.name));
                self.status_message = format!("✓ {} compiled successfully - {} metrics", 
                    model.name, 77);
            }
            Err(e) => {
                self.status_message = format!("✗ Compilation failed: {}", e);
            }
        }
    }
    
    pub fn refresh(&mut self) {
        self.compiled_result = None;
        self.real_world_data = None;
        self.status_message = "Select a model and press Enter to compile".to_string();
    }
}
