//! IREE Compiler Driver
//!
//! Provides a unified interface for compiling NEURAX MLIR to IREE bytecode (.vmfb)
//! for deployment across multiple hardware backends.

use super::IreeDevice;
use crate::targets::TargetBackend;
use std::path::{Path, PathBuf};
use std::process::Command;

/// IREE compilation target configuration
#[derive(Debug, Clone)]
pub struct IreeTarget {
    pub device: IreeDevice,
    pub input_type: IreeInputType,
    pub optimization_level: OptimizationLevel,
}

impl IreeTarget {
    pub fn new(device: IreeDevice) -> Self {
        Self {
            device,
            input_type: IreeInputType::Linalg,
            optimization_level: OptimizationLevel::O3,
        }
    }
    
    pub fn from_backend(backend: TargetBackend) -> Self {
        Self::new(IreeDevice::from_backend(backend))
    }
    
    pub fn with_input_type(mut self, input_type: IreeInputType) -> Self {
        self.input_type = input_type;
        self
    }
    
    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }
}

/// IREE input dialect types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IreeInputType {
    /// Linalg dialect (default)
    Linalg,
    /// StableHLO dialect
    StableHLO,
    /// TOSA dialect
    Tosa,
    /// TensorFlow dialect
    TensorFlow,
    /// PyTorch dialect
    Torch,
}

impl IreeInputType {
    pub fn flag(&self) -> &'static str {
        match self {
            Self::Linalg => "", // Default, no flag needed
            Self::StableHLO => "--iree-input-type=stablehlo",
            Self::Tosa => "--iree-input-type=tosa",
            Self::TensorFlow => "--iree-input-type=tensorflow",
            Self::Torch => "--iree-input-type=torch",
        }
    }
}

/// Optimization levels for IREE compilation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    O0,
    O1,
    O2,
    O3,
}

impl OptimizationLevel {
    pub fn flag(&self) -> &'static str {
        match self {
            Self::O0 => "-O0",
            Self::O1 => "-O1",
            Self::O2 => "-O2",
            Self::O3 => "-O3",
        }
    }
}

/// IREE compiler driver
pub struct IreeCompiler {
    /// Path to iree-compile executable
    iree_compile_path: PathBuf,
    /// Path to iree-run-module executable
    iree_run_path: PathBuf,
    /// Additional compiler flags
    extra_flags: Vec<String>,
}

impl IreeCompiler {
    /// Create a new IREE compiler with default paths
    pub fn new() -> Self {
        Self {
            iree_compile_path: PathBuf::from("iree-compile"),
            iree_run_path: PathBuf::from("iree-run-module"),
            extra_flags: Vec::new(),
        }
    }
    
    /// Create with custom paths
    pub fn with_paths(compile_path: &Path, run_path: &Path) -> Self {
        Self {
            iree_compile_path: compile_path.to_path_buf(),
            iree_run_path: run_path.to_path_buf(),
            extra_flags: Vec::new(),
        }
    }
    
    /// Add extra compiler flag
    pub fn add_flag(mut self, flag: &str) -> Self {
        self.extra_flags.push(flag.to_string());
        self
    }
    
    /// Compile MLIR to IREE bytecode (.vmfb)
    pub fn compile(
        &self,
        input_mlir: &Path,
        output_vmfb: &Path,
        target: &IreeTarget,
    ) -> Result<(), String> {
        let mut cmd = Command::new(&self.iree_compile_path);
        
        // Add input file
        cmd.arg(input_mlir);
        
        // Add target device and backend flags
        cmd.arg(target.device.backend_flag());
        
        // Add input type flag
        if !target.input_type.flag().is_empty() {
            cmd.arg(target.input_type.flag());
        }
        
        // Add optimization level
        cmd.arg(target.optimization_level.flag());
        
        // Add output file
        cmd.arg("-o").arg(output_vmfb);
        
        // Add extra flags
        for flag in &self.extra_flags {
            cmd.arg(flag);
        }
        
        // Execute compilation
        let output = cmd.output()
            .map_err(|e| format!("Failed to execute iree-compile: {}", e))?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("IREE compilation failed:\n{}", stderr));
        }
        
        Ok(())
    }
    
    /// Generate the compile command string (for display or manual execution)
    pub fn compile_command(
        &self,
        input_mlir: &Path,
        output_vmfb: &Path,
        target: &IreeTarget,
    ) -> String {
        let mut args = vec![
            input_mlir.display().to_string(),
            target.device.backend_flag().to_string(),
        ];
        
        if !target.input_type.flag().is_empty() {
            args.push(target.input_type.flag().to_string());
        }
        
        args.push(target.optimization_level.flag().to_string());
        args.push("-o".to_string());
        args.push(output_vmfb.display().to_string());
        
        for flag in &self.extra_flags {
            args.push(flag.clone());
        }
        
        format!("{} {}", self.iree_compile_path.display(), args.join(" "))
    }
    
    /// Run a compiled IREE module
    pub fn run(
        &self,
        vmfb_path: &Path,
        function: &str,
        inputs: &[String],
    ) -> Result<String, String> {
        let mut cmd = Command::new(&self.iree_run_path);
        
        cmd.arg("--module").arg(vmfb_path);
        cmd.arg("--function").arg(function);
        
        for input in inputs {
            cmd.arg("--input").arg(input);
        }
        
        let output = cmd.output()
            .map_err(|e| format!("Failed to execute iree-run-module: {}", e))?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("IREE execution failed:\n{}", stderr));
        }
        
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
    
    /// Check if IREE tools are available
    pub fn check_available(&self) -> Result<(), String> {
        let output = Command::new(&self.iree_compile_path)
            .arg("--version")
            .output()
            .map_err(|_| "iree-compile not found. Install IREE tools first.")?;
        
        if !output.status.success() {
            return Err("iree-compile failed to run. Check IREE installation.".to_string());
        }
        
        Ok(())
    }
}

impl Default for IreeCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate IREE compilation script for all targets
pub fn generate_multi_target_script(
    input_mlir: &str,
    model_name: &str,
    targets: &[IreeDevice],
    output_dir: &str,
) -> String {
    let mut script = String::new();
    
    script.push_str("#!/bin/bash\n\n");
    script.push_str(&format!("# IREE compilation script for {}\n", model_name));
    script.push_str(&format!("# Input: {}\n\n", input_mlir));
    
    for target in targets {
        let output_file = format!("{}/{}_{}.vmfb", output_dir, model_name, target);
        
        script.push_str(&format!("# Compile for {}\n", target));
        script.push_str(&format!(
            "echo 'Compiling for {}...'\n",
            target
        ));
        script.push_str(&format!(
            "iree-compile {} {} -o {}\n",
            input_mlir,
            target.backend_flag(),
            output_file
        ));
        script.push_str(&format!(
            "echo 'Output: {}'\n\n",
            output_file
        ));
    }
    
    script.push_str("echo 'All targets compiled successfully!'\n");
    
    script
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_iree_target() {
        let target = IreeTarget::new(IreeDevice::Cuda);
        assert_eq!(target.device, IreeDevice::Cuda);
    }
    
    #[test]
    fn test_iree_target_from_backend() {
        let target = IreeTarget::from_backend(TargetBackend::Vulkan);
        assert_eq!(target.device, IreeDevice::Vulkan);
    }
    
    #[test]
    fn test_iree_compiler_command() {
        let compiler = IreeCompiler::new();
        let target = IreeTarget::new(IreeDevice::Cpu);
        let cmd = compiler.compile_command(
            Path::new("model.mlir"),
            Path::new("model.vmfb"),
            &target,
        );
        assert!(cmd.contains("iree-compile"));
        assert!(cmd.contains("llvm-cpu"));
    }
    
    #[test]
    fn test_multi_target_script() {
        let script = generate_multi_target_script(
            "model.mlir",
            "test_model",
            &[IreeDevice::Cpu, IreeDevice::Cuda],
            "output",
        );
        assert!(script.contains("llvm-cpu"));
        assert!(script.contains("cuda"));
    }
    
    #[test]
    fn test_optimization_flags() {
        assert_eq!(OptimizationLevel::O3.flag(), "-O3");
        assert_eq!(OptimizationLevel::O0.flag(), "-O0");
    }
    
    #[test]
    fn test_input_type_flags() {
        assert_eq!(IreeInputType::StableHLO.flag(), "--iree-input-type=stablehlo");
        assert_eq!(IreeInputType::Tosa.flag(), "--iree-input-type=tosa");
    }
}
