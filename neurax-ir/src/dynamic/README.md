# NEURAX Dynamic Predictive System

## Overview

The Dynamic Predictive System extends the static Neurax compiler with three optional analysis passes that provide predictive capabilities:

1. **VirtualMemoryPass** - Models memory fragmentation and predicts savings from virtualization
2. **StabilityAnalysisPass** - Predicts training stability via Lyapunov exponents
3. **BehavioralSynthesisPass** - Infers runtime behaviors (MoE imbalance, cache locality)

## Metrics Produced

### Virtual Memory (M36-M42)
| Metric | Description | Range |
|--------|-------------|-------|
| M36 | fragmentation_overhead_gb | 0+ |
| M37 | fragmentation_pct | 0-50% |
| M38 | defrag_savings_gb | 0+ |
| M39 | virtual_savings_gb | 0+ |
| M40 | virtual_savings_pct | 0-75% |
| M41 | peak_vram_with_defrag_gb | ≤ peak |
| M42 | peak_vram_with_virtual_gb | ≤ defrag |

### Stability (M43-M49)
| Metric | Description | Range |
|--------|-------------|-------|
| M43 | stability_margin_by_layer | 0-1 per layer |
| M44 | lyapunov_exponent_mean | >0 = chaos |
| M45 | chaos_index | 0-1 |
| M46 | high_risk_layers | list |
| M47 | fp32_required_pct | 0-100% |
| M48 | global_robustness_score | 0-1 |
| M49 | fp32_fallback_memory_overhead_gb | 0+ |

### Behavioral (M50-M55)
| Metric | Description | Range |
|--------|-------------|-------|
| M50 | expert_load_imbalance | 0-1 |
| M51 | memory_contention_score | 0-1 |
| M52 | cache_locality_score | 0-1 |
| M53 | numerical_sensitivity | 0-1 |
| M54 | load_balance_efficiency | 0-100% |
| M55 | memory_bank_conflict_rate | 0-15% |

## Usage

```rust
use neurax_ir::dynamic::{
    VirtualMemoryPass, StabilityAnalysisPass, BehavioralSynthesisPass,
    DynamicConfig, DynamicResults, apply_dynamic_feedback,
};

// Create passes
let vm_pass = VirtualMemoryPass::new();
let sta_pass = StabilityAnalysisPass::new();
let bps_pass = BehavioralSynthesisPass::new();

// Run analysis
let vm_metrics = vm_pass.run(&memory_metrics);
let sta_metrics = sta_pass.run(&graph_ir, &memory_metrics);
let bps_metrics = bps_pass.run(&compute_ir, &config);

// Apply feedback
let dynamic_results = DynamicResults {
    virtual_memory: Some(vm_metrics),
    stability: Some(sta_metrics),
    behavioral: Some(bps_metrics),
};

let diagnostics = apply_dynamic_feedback(&mut memory_metrics, &dynamic_results);
```

## Cross-Pass Amplification

The system implements feedback loops between passes:

1. **Stability → Memory**: Unstable layers trigger fp32 fallback
2. **MoE → Liveness**: Load imbalance extends activation memory
3. **Fragmentation → Flash Attention**: High fragmentation triggers recommendation

## Allocation Strategies

| Fragmentation | Strategy |
|--------------|----------|
| 0-4% | NoAction |
| 5-14% | EnableCompaction |
| 15-29% | EnableFlashAttention |
| 30%+ | EnableVirtualMemory |

## Evaluation

Run the evaluation suite:

```rust
use neurax_ir::dynamic::run_full_evaluation;

let report = run_full_evaluation();
report.print_summary();

// Check if production ready (85% pass rate)
if report.is_production_ready() {
    println!("System is production ready!");
}
```

## Performance Overhead

| Pass | Overhead |
|------|----------|
| VirtualMemoryPass | ~5ms |
| StabilityAnalysisPass | ~10ms |
| BehavioralSynthesisPass (V1) | ~5ms |
| BehavioralSynthesisPass (V2) | ~50ms |

## Implementation Status

- [x] VirtualMemoryPass (M36-M42)
- [x] StabilityAnalysisPass (M43-M49)
- [x] BehavioralSynthesisPass (M50-M55)
- [x] Cross-pass feedback
- [x] Evaluation suite (14 tests)
- [x] Documentation

## Future Work

- [ ] V2 BehavioralSynthesisPass with GNN model
- [ ] Calibration data integration
- [ ] MLIR dialect `virt` implementation
- [ ] Additional stability metrics per layer type
