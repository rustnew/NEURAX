//! Tests for NEURAX MLIR dialects

#[cfg(test)]
mod tests {
    use melior::ir::Location;
    use crate::{
        NeuraxContext, NeuraxModule,
        dialects::{
            ArchitectureDialect, GraphDialect, TensorDialect, OperatorDialect,
            ComputeDialect, MemoryDialect, HardwareDialect, ParallelismDialect,
            CostDialect, ReportDialect,
        },
    };

    #[test]
    fn test_context_creation() {
        let ctx = NeuraxContext::new();
        // Context should be created successfully
        let _ = ctx.as_context();
    }

    #[test]
    fn test_module_creation() {
        let ctx = NeuraxContext::new();
        let module = NeuraxModule::new(ctx.as_context());
        assert!(module.as_module().as_operation().to_string().contains("module"));
    }

    // Architecture dialect tests
    mod architecture {
        use super::*;

        #[test]
        fn test_arch_model_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = ArchitectureDialect::model(
                ctx.as_context(),
                "test_model",
                "transformer",
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("arch.model"));
            assert!(op_str.contains("test_model"));
            assert!(op_str.contains("transformer"));
        }

        #[test]
        fn test_arch_layer_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = ArchitectureDialect::layer(
                ctx.as_context(),
                "layer_0",
                "attention",
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("arch.layer"));
            assert!(op_str.contains("layer_0"));
        }

        #[test]
        fn test_arch_metrics_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = ArchitectureDialect::metrics(
                ctx.as_context(),
                70000000000, // 70B parameters
                80,          // 80 layers
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("arch.metrics"));
        }
    }

    // Graph dialect tests
    mod graph {
        use super::*;

        #[test]
        fn test_graph_node_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = GraphDialect::node(
                ctx.as_context(),
                "layer_0_attn",
                "attention",
                1.4e12, // FLOPs
                8388608, // params
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("graph.node"));
        }

        #[test]
        fn test_graph_edge_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = GraphDialect::edge(
                ctx.as_context(),
                &[1, 8192, 8192],
                "f32",
                268435456, // bytes
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("graph.edge"));
        }
    }

    // Tensor dialect tests
    mod tensor {
        use super::*;

        #[test]
        fn test_tensor_info_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = TensorDialect::tensor_info(
                ctx.as_context(),
                "hidden_state_0",
                &[1, 8192, 8192],
                "f32",
                268435456,
                "layer_0_attn",
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("tensor.info"));
        }
    }

    // Operator dialect tests
    mod operator {
        use super::*;

        #[test]
        fn test_matmul_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = OperatorDialect::matmul(
                ctx.as_context(),
                8388608,  // param_count
                1.4e12,   // flops
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("op.matmul"));
        }

        #[test]
        fn test_attention_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = OperatorDialect::attention(
                ctx.as_context(),
                8192,     // hidden_size
                64,       // num_heads
                8388608,  // param_count
                1.4e12,   // flops
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("op.attention"));
        }

        #[test]
        fn test_moe_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = OperatorDialect::moe(
                ctx.as_context(),
                8192,     // hidden_size
                8,        // num_experts
                2,        // top_k
                16777216, // param_count
                2.8e12,   // flops
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("op.moe"));
        }
    }

    // Compute dialect tests
    mod compute {
        use super::*;

        #[test]
        fn test_flops_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = ComputeDialect::flops(
                ctx.as_context(),
                1.4e12,  // forward_flops
                2.8e12,  // backward_flops
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("compute.flops"));
        }

        #[test]
        fn test_intensity_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = ComputeDialect::intensity(
                ctx.as_context(),
                1.4e12,       // flops
                268435456,    // bytes_accessed
                5222.0,       // intensity
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("compute.intensity"));
        }
    }

    // Memory dialect tests
    mod memory {
        use super::*;

        #[test]
        fn test_liveness_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = MemoryDialect::liveness(
                ctx.as_context(),
                "tensor_0",
                0,           // start_step
                10,          // end_step
                268435456,   // size_bytes
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("mem.liveness"));
        }

        #[test]
        fn test_peak_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = MemoryDialect::peak(
                ctx.as_context(),
                80000000000, // peak_bytes (80GB)
                Some(42),    // peak_step
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("mem.peak"));
        }

        #[test]
        fn test_oom_risk_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = MemoryDialect::oom_risk(
                ctx.as_context(),
                "high",
                0.95,  // utilization_ratio
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("mem.oom_risk"));
        }
    }

    // Hardware dialect tests
    mod hardware {
        use super::*;

        #[test]
        fn test_gpu_profile_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = HardwareDialect::gpu(
                ctx.as_context(),
                "H100_SXM5",
                80,       // vram_gb
                989.0,    // peak_tflops
                3352.0,   // memory_bandwidth GB/s
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("hw.gpu"));
        }

        #[test]
        fn test_roofline_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = HardwareDialect::roofline(
                ctx.as_context(),
                989.0,    // compute_roof
                3352.0,   // memory_roof
                0.295,    // ridge_point
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("hw.roofline"));
        }
    }

    // Parallelism dialect tests
    mod parallelism {
        use super::*;

        #[test]
        fn test_data_parallel_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = ParallelismDialect::data_parallel(
                ctx.as_context(),
                8,      // num_gpus
                0.85,   // efficiency
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("par.data_parallel"));
        }

        #[test]
        fn test_tensor_parallel_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = ParallelismDialect::tensor_parallel(
                ctx.as_context(),
                8,  // tp_degree
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("par.tensor_parallel"));
        }

        #[test]
        fn test_zero_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = ParallelismDialect::zero(
                ctx.as_context(),
                3,              // stage
                10000000000,    // memory_per_gpu_bytes
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("par.zero"));
        }
    }

    // Cost dialect tests
    mod cost {
        use super::*;

        #[test]
        fn test_pricing_model_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = CostDialect::pricing_model(
                ctx.as_context(),
                2.5,    // gpu_hour_usd
                0.12,   // energy_kwh_usd
                1.4,    // pue_factor
                700.0,  // gpu_tdp_watts
                0.4,    // co2_per_kwh
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("cost.pricing"));
        }

        #[test]
        fn test_training_cost_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = CostDialect::training_cost(
                ctx.as_context(),
                720.0,      // training_time_hours
                144000.0,   // training_cost_usd
                57600.0,    // gpu_hours_total
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("cost.training"));
        }
    }

    // Report dialect tests
    mod report {
        use super::*;

        #[test]
        fn test_report_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = ReportDialect::report(
                ctx.as_context(),
                "Llama-3-70B",
                "transformer",
                "1.0",
                150,  // analysis_time_ms
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("report.report"));
        }

        #[test]
        fn test_all_metrics_operation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            let op = ReportDialect::all_metrics(
                ctx.as_context(),
                70000000000,   // total_parameters
                1.4e17,        // total_flops
                80000000000,   // peak_vram_bytes
                150.0,         // latency_ms
                144000.0,      // training_cost_usd
                location,
            ).unwrap();
            
            let op_str = op.to_string();
            assert!(op_str.contains("report.all_metrics"));
        }
    }

    // Integration test: Create a full model representation
    mod integration {
        use super::*;

        #[test]
        fn test_llama70b_model_representation() {
            let ctx = NeuraxContext::new();
            let location = Location::unknown(ctx.as_context());
            
            // Create architecture model
            let model_op = ArchitectureDialect::model(
                ctx.as_context(),
                "Llama-3-70B",
                "transformer",
                location,
            ).unwrap();
            assert!(model_op.to_string().contains("arch.model"));
            
            // Create layer operations
            let layer_op = ArchitectureDialect::layer(
                ctx.as_context(),
                "layer_0_attn",
                "attention",
                location,
            ).unwrap();
            assert!(layer_op.to_string().contains("arch.layer"));
            
            // Create attention operation
            let attn_op = OperatorDialect::attention(
                ctx.as_context(),
                8192,     // hidden_size
                64,       // num_heads
                8388608,  // param_count
                1.4e12,   // flops
                location,
            ).unwrap();
            assert!(attn_op.to_string().contains("op.attention"));
            
            // Create memory metrics
            let mem_op = MemoryDialect::metrics(
                ctx.as_context(),
                140000000000,  // parameter_memory_bytes
                80000000000,   // activation_memory_bytes
                140000000000,  // gradient_memory_bytes
                280000000000,  // optimizer_state_bytes
                160000000000,  // peak_vram_bytes
                4,             // max_batch_size_fit
                location,
            ).unwrap();
            assert!(mem_op.to_string().contains("mem.metrics"));
            
            // Create hardware profile
            let hw_op = HardwareDialect::gpu(
                ctx.as_context(),
                "H100_SXM5",
                80,         // vram_gb
                989.0,      // peak_tflops
                3352.0,     // memory_bandwidth
                location,
            ).unwrap();
            assert!(hw_op.to_string().contains("hw.gpu"));
            
            // Create parallelism config
            let par_op = ParallelismDialect::hybrid(
                ctx.as_context(),
                8,  // dp
                8,  // tp
                1,  // pp
                location,
            ).unwrap();
            assert!(par_op.to_string().contains("par.hybrid"));
            
            // Create cost metrics
            let cost_op = CostDialect::training_cost(
                ctx.as_context(),
                720.0,
                144000.0,
                57600.0,
                location,
            ).unwrap();
            assert!(cost_op.to_string().contains("cost.training"));
            
            // Create final report
            let report_op = ReportDialect::report(
                ctx.as_context(),
                "Llama-3-70B",
                "transformer",
                "1.0",
                150,
                location,
            ).unwrap();
            assert!(report_op.to_string().contains("report.report"));
        }
    }
}
