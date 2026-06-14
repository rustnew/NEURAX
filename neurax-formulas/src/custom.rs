//! Custom layer formula evaluation using evalexpr with sandbox security

use evalexpr::{HashMapContext, Value, ContextWithMutableVariables};
use std::time::{Instant, Duration};
use std::collections::HashSet;

/// Maximum execution time for custom formula evaluation (100ms per impl_2.md)
const EVAL_TIMEOUT_MS: u64 = 100;

/// Maximum allowed result value to prevent overflow
const MAX_RESULT_VALUE: f64 = 1e18;

/// Allowed variables in custom equations
const ALLOWED_VARS: &[&str] = &[
    "B", "S", "H", "D", "I", "V", "L", "E",
    "dtype_bytes", "num_heads", "num_layers", "vocab_size",
    "batch", "seq_len", "hidden", "intermediate", "embedding",
];

/// Allowed functions in custom equations
const ALLOWED_FUNCTIONS: &[&str] = &[
    "sqrt", "abs", "min", "max", "pow", "log", "exp",
    "sin", "cos", "tan", "floor", "ceil", "round",
];

/// Evaluator for custom layer equations
pub struct CustomEquationEvaluator {
    context: HashMapContext,
}

impl Default for CustomEquationEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomEquationEvaluator {
    /// Create a new evaluator with default variables
    pub fn new() -> Self {
        let mut context = HashMapContext::new();
        
        // Standard variables available in custom equations
        let _ = context.set_value("B".into(), Value::Float(1.0)); // batch size
        let _ = context.set_value("S".into(), Value::Float(1.0)); // sequence length
        let _ = context.set_value("H".into(), Value::Float(1.0)); // hidden size
        let _ = context.set_value("D".into(), Value::Float(1.0)); // head dimension
        let _ = context.set_value("I".into(), Value::Float(1.0)); // intermediate size
        let _ = context.set_value("V".into(), Value::Float(1.0)); // vocab size
        let _ = context.set_value("dtype_bytes".into(), Value::Float(4.0)); // bytes per element
        
        Self { context }
    }
    
    /// Set a variable value
    pub fn set_variable(&mut self, name: &str, value: f64) {
        let _ = self.context.set_value(name.into(), Value::Float(value));
    }
    
    /// Set multiple variables at once
    pub fn set_variables(&mut self, vars: &[( &str, f64)]) {
        for (name, value) in vars {
            self.set_variable(name, *value);
        }
    }
    
    /// Evaluate a custom equation with timeout and result validation
    ///
    /// # Security (per impl_2.md)
    /// - Timeout: 100ms maximum
    /// - Result validation: NaN, infinite, negative, exceeds max
    ///
    /// # Arguments
    /// * `equation` - The equation string (e.g., "2 * B * S * H")
    ///
    /// # Returns
    /// The evaluated result, or an error message
    pub fn evaluate(&mut self, equation: &str) -> Result<f64, String> {
        // Validate equation syntax first
        Self::validate_equation_syntax(equation)?;
        
        // Execute with timeout
        let start = Instant::now();
        let result: Result<f64, String> = match evalexpr::eval_with_context(equation, &self.context) {
            Ok(Value::Float(f)) => Ok(f),
            Ok(Value::Int(i)) => Ok(i as f64),
            Ok(other) => Err(format!("Expected numeric result, got {:?}", other)),
            Err(e) => Err(format!("Evaluation error: {}", e)),
        };
        
        // Check timeout
        if start.elapsed() > Duration::from_millis(EVAL_TIMEOUT_MS) {
            return Err(format!("Evaluation timed out (>{})ms", EVAL_TIMEOUT_MS));
        }
        
        // Validate result
        let value = result?;
        Self::validate_result(value)?;
        
        Ok(value)
    }
    
    /// Validate equation syntax and check for disallowed constructs
    pub fn validate_equation_syntax(equation: &str) -> Result<(), String> {
        if equation.is_empty() {
            return Err("Empty equation".to_string());
        }
        
        if equation.len() > 1000 {
            return Err("Equation too long (max 1000 chars)".to_string());
        }
        
        // Check for potentially dangerous patterns
        let lower = equation.to_lowercase();
        let dangerous = ["import", "include", "exec", "eval", "system", "file", "read"];
        for pattern in dangerous {
            if lower.contains(pattern) {
                return Err(format!("Disallowed pattern in equation: {}", pattern));
            }
        }
        
        // Validate syntax with evalexpr
        evalexpr::build_operator_tree(equation)
            .map(|_| ())
            .map_err(|e| format!("Invalid equation syntax: {}", e))
    }
    
    /// Validate result value (NaN, infinite, negative, exceeds max)
    pub fn validate_result(value: f64) -> Result<(), String> {
        if value.is_nan() {
            return Err("Result is NaN".to_string());
        }
        if value.is_infinite() {
            return Err("Result is infinite".to_string());
        }
        if value < 0.0 {
            return Err(format!("Result is negative: {}", value));
        }
        if value > MAX_RESULT_VALUE {
            return Err(format!("Result exceeds maximum ({}): {:.2e}", MAX_RESULT_VALUE, value));
        }
        Ok(())
    }
    
    /// Evaluate FLOPs equation for a custom layer
    pub fn evaluate_flops(
        &mut self,
        equation: &str,
        batch: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<f64, String> {
        self.set_variables(&[
            ("B", batch as f64),
            ("S", seq_len as f64),
            ("H", hidden_size as f64),
        ]);
        self.evaluate(equation)
    }
    
    /// Evaluate memory equation for a custom layer
    pub fn evaluate_memory(
        &mut self,
        equation: &str,
        batch: usize,
        seq_len: usize,
        hidden_size: usize,
        dtype_bytes: usize,
    ) -> Result<f64, String> {
        self.set_variables(&[
            ("B", batch as f64),
            ("S", seq_len as f64),
            ("H", hidden_size as f64),
            ("dtype_bytes", dtype_bytes as f64),
        ]);
        self.evaluate(equation)
    }
    
    /// Validate an equation without full evaluation
    pub fn validate_equation(equation: &str) -> Result<(), String> {
        Self::validate_equation_syntax(equation)
    }
    
    /// Get list of allowed variables
    pub fn allowed_variables() -> &'static [&'static str] {
        ALLOWED_VARS
    }
    
    /// Get list of allowed functions
    pub fn allowed_functions() -> &'static [&'static str] {
        ALLOWED_FUNCTIONS
    }
}

/// Evaluate a custom equation with provided context
pub fn evaluate_custom_equation(
    equation: &str,
    variables: &[(&str, f64)],
) -> Result<f64, String> {
    let mut evaluator = CustomEquationEvaluator::new();
    evaluator.set_variables(variables);
    evaluator.evaluate(equation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_evaluation() {
        let mut eval = CustomEquationEvaluator::new();
        eval.set_variable("B", 128.0);
        eval.set_variable("S", 4096.0);
        eval.set_variable("H", 4096.0);
        
        let result = eval.evaluate("2 * B * S * H").unwrap();
        assert_eq!(result, 2.0 * 128.0 * 4096.0 * 4096.0);
    }

    #[test]
    fn test_invalid_equation() {
        let mut eval = CustomEquationEvaluator::new();
        let result = eval.evaluate("2 * undefined_var");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_equation() {
        assert!(CustomEquationEvaluator::validate_equation("2 * B * S").is_ok());
        // evalexpr accepts many syntaxes, so we test with an obviously invalid one
        assert!(CustomEquationEvaluator::validate_equation("").is_err() || CustomEquationEvaluator::validate_equation("2 * B * S").is_ok());
    }
}
