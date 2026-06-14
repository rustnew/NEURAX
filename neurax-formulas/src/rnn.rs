//! RNN, LSTM, GRU formulas
//!
//! Hot path — all functions are #[inline(always)] for zero-cost abstraction.

/// Compute FLOPs for LSTM cell
///
/// Standard LSTM equations:
/// f = sigmoid(Wf * [h, x] + bf)
/// i = sigmoid(Wi * [h, x] + bi)
/// o = sigmoid(Wo * [h, x] + bo)
/// g = tanh(Wg * [h, x] + bg)
/// c' = f * c + i * g
/// h' = o * tanh(c')
#[inline(always)]
pub fn lstm_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    input_size: usize,
) -> f64 {
    // 4 gates: forget, input, output, cell
    // Each gate: [h; x] × W + b where [h; x] has dimension (hidden + input)
    let combined_dim = hidden_size + input_size;
    
    // 4 matrix multiplications: 4 × (combined_dim × hidden)
    let gate_flops = 4.0 * 2.0 * batch as f64 * seq_len as f64 * combined_dim as f64 * hidden_size as f64;
    
    // 4 sigmoid + 1 tanh activations
    let activation_flops = 5.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * 10.0;
    
    // Cell update: c' = f * c + i * g (element-wise)
    let cell_update = 3.0 * batch as f64 * seq_len as f64 * hidden_size as f64;
    
    // Output: h' = o * tanh(c')
    let output_flops = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64;
    
    gate_flops + activation_flops + cell_update + output_flops
}

/// Compute FLOPs for GRU cell
///
/// Standard GRU equations:
/// z = sigmoid(Wz * [h, x] + bz)
/// r = sigmoid(Wr * [h, x] + br)
/// h' = (1-z) * h + z * tanh(Wh * [r*h, x] + bh)
#[inline(always)]
pub fn gru_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    input_size: usize,
) -> f64 {
    let combined_dim = hidden_size + input_size;
    
    // 2 gates: update (z), reset (r)
    let gate_flops = 2.0 * 2.0 * batch as f64 * seq_len as f64 * combined_dim as f64 * hidden_size as f64;
    
    // Candidate hidden state: tanh(Wh * [r*h, x])
    let candidate_dim = hidden_size + input_size;
    let candidate_flops = 2.0 * batch as f64 * seq_len as f64 * candidate_dim as f64 * hidden_size as f64;
    
    // Activations
    let activation_flops = 3.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * 10.0;
    
    // Hidden state update
    let update_flops = 3.0 * batch as f64 * seq_len as f64 * hidden_size as f64;
    
    gate_flops + candidate_flops + activation_flops + update_flops
}

/// Compute FLOPs for simple RNN cell
#[inline(always)]
pub fn rnn_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    input_size: usize,
) -> f64 {
    // h' = tanh(Wh * h + Wx * x + b)
    let h_proj = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * hidden_size as f64;
    let x_proj = 2.0 * batch as f64 * seq_len as f64 * input_size as f64 * hidden_size as f64;
    let activation = batch as f64 * seq_len as f64 * hidden_size as f64 * 10.0;
    
    h_proj + x_proj + activation
}

/// Compute parameters for LSTM
#[inline(always)]
pub fn lstm_params(hidden_size: usize, input_size: usize, bias: bool) -> u64 {
    let combined_dim = hidden_size + input_size;
    
    // 4 gates, each with combined_dim × hidden weights
    let weights = 4 * combined_dim * hidden_size;
    
    let biases = if bias { 4 * hidden_size } else { 0 };
    
    (weights + biases) as u64
}

/// Compute parameters for GRU
#[inline(always)]
pub fn gru_params(hidden_size: usize, input_size: usize, bias: bool) -> u64 {
    let combined_dim = hidden_size + input_size;
    
    // 2 gates (z, r) + 1 candidate
    let weights = 2 * combined_dim * hidden_size + (hidden_size + input_size) * hidden_size;
    
    let biases = if bias { 3 * hidden_size } else { 0 };
    
    (weights + biases) as u64
}

/// Compute parameters for simple RNN
#[inline(always)]
pub fn rnn_params(hidden_size: usize, input_size: usize, bias: bool) -> u64 {
    let weights = hidden_size * hidden_size + input_size * hidden_size;
    let biases = if bias { hidden_size } else { 0 };
    
    (weights + biases) as u64
}

/// Compute FLOPs for bidirectional RNN (forward + backward)
#[inline(always)]
pub fn bidirectional_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    input_size: usize,
    rnn_type: &str,
) -> f64 {
    let single_direction = match rnn_type {
        "lstm" => lstm_flops(batch, seq_len, hidden_size, input_size),
        "gru" => gru_flops(batch, seq_len, hidden_size, input_size),
        _ => rnn_flops(batch, seq_len, hidden_size, input_size),
    };
    
    2.0 * single_direction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_flops() {
        let flops = lstm_flops(32, 128, 256, 256);
        assert!(flops > 0.0);
    }

    #[test]
    fn test_gru_flops() {
        let flops = gru_flops(32, 128, 256, 256);
        // GRU should be more efficient than LSTM
        let lstm_flops = lstm_flops(32, 128, 256, 256);
        assert!(flops < lstm_flops);
    }

    #[test]
    fn test_lstm_params() {
        let params = lstm_params(256, 256, true);
        // 4 gates × (512 × 256) weights + 4 × 256 biases
        assert!(params > 0);
    }
}
