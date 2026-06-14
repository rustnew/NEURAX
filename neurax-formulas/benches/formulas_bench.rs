//! Benchmarks for neurax-formulas
//!
//! Run with: cargo bench --package neurax-formulas

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

// ═══════════════════════════════════════════════════════════════════════════
// CONVOLUTION BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_conv2d_flops(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_flops");
    
    // ResNet-style convolutions
    for (name, params) in [
        ("7x7_stride2", (1, 3, 64, 224, 224, 7, 7, 2, 3, 1)),
        ("3x3_stride1", (1, 64, 64, 56, 56, 3, 3, 1, 1, 1)),
        ("1x1_stride1", (1, 256, 512, 14, 14, 1, 1, 1, 0, 1)),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(name), &params, |b, &(batch, ic, oc, h, w, kh, kw, s, p, g)| {
            b.iter(|| neurax_formulas::conv::conv2d_flops(
                black_box(batch), black_box(ic), black_box(oc),
                black_box(h), black_box(w), black_box(kh), black_box(kw),
                black_box(s), black_box(p), black_box(g),
            ))
        });
    }
    group.finish();
}

fn bench_conv2d_params(c: &mut Criterion) {
    c.bench_function("conv2d_params_resnet_first", |b| {
        b.iter(|| neurax_formulas::conv::conv2d_params(
            black_box(3), black_box(64), black_box(7), black_box(7), black_box(1), black_box(true),
        ))
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// ATTENTION BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_attention_flops(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_flops");
    
    // GPT-2 style attention
    for (name, params) in [
        ("gpt2_small", (1, 1024, 768, 12)),
        ("gpt2_medium", (1, 1024, 1024, 16)),
        ("gpt2_large", (1, 1024, 1280, 20)),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(name), &params, |b, &(batch, seq, hidden, heads)| {
            b.iter(|| neurax_formulas::attention::attention_flops(
                black_box(batch), black_box(seq), black_box(hidden), black_box(heads), black_box(true),
            ))
        });
    }
    group.finish();
}

fn bench_gqa_flops(c: &mut Criterion) {
    let mut group = c.benchmark_group("gqa_flops");
    
    // LLaMA style GQA
    for (name, params) in [
        ("llama_7b", (1, 2048, 4096, 32, 32)),
        ("llama_70b", (1, 2048, 8192, 64, 8)),  // 8 KV heads for 70B
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(name), &params, |b, &(batch, seq, hidden, heads, kv_heads)| {
            b.iter(|| neurax_formulas::attention::gqa_flops(
                black_box(batch), black_box(seq), black_box(hidden),
                black_box(heads), black_box(kv_heads), black_box(true),
            ))
        });
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// MLP BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_mlp_flops(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp_flops");
    
    for (name, params) in [
        ("gpt2_small", (1, 1024, 768, 3072)),
        ("llama_7b", (1, 2048, 4096, 11008)),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(name), &params, |b, &(batch, seq, hidden, intermediate)| {
            b.iter(|| neurax_formulas::mlp::mlp_flops(
                black_box(batch), black_box(seq), black_box(hidden), black_box(intermediate), black_box("gelu"),
            ))
        });
    }
    group.finish();
}

fn bench_gated_mlp_flops(c: &mut Criterion) {
    c.bench_function("gated_mlp_llama", |b| {
        b.iter(|| neurax_formulas::mlp::gated_mlp_flops(
            black_box(1), black_box(2048), black_box(4096), black_box(11008), black_box("silu"),
        ))
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// SSM BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_mamba_flops(c: &mut Criterion) {
    let mut group = c.benchmark_group("mamba_flops");
    
    for (name, params) in [
        ("mamba_130m", (1, 2048, 768, 16, 2)),
        ("mamba_1.4b", (1, 2048, 2048, 16, 2)),
        ("mamba_2.8b", (1, 2048, 2560, 16, 2)),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(name), &params, |b, &(batch, seq, hidden, state, expand)| {
            b.iter(|| neurax_formulas::ssm::mamba_flops(
                black_box(batch), black_box(seq), black_box(hidden), black_box(state), black_box(expand),
            ))
        });
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// RNN/LSTM BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════════

fn bench_lstm_flops(c: &mut Criterion) {
    c.bench_function("lstm_flops", |b| {
        b.iter(|| neurax_formulas::rnn::lstm_flops(
            black_box(32), black_box(128), black_box(256), black_box(256),
        ))
    });
}

fn bench_gru_flops(c: &mut Criterion) {
    c.bench_function("gru_flops", |b| {
        b.iter(|| neurax_formulas::rnn::gru_flops(
            black_box(32), black_box(128), black_box(256), black_box(256),
        ))
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// CRITERION CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(5));
    targets = 
        bench_conv2d_flops,
        bench_conv2d_params,
        bench_attention_flops,
        bench_gqa_flops,
        bench_mlp_flops,
        bench_gated_mlp_flops,
        bench_mamba_flops,
        bench_lstm_flops,
        bench_gru_flops,
}

criterion_main!(benches);
