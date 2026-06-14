//===- ReportOps.td - Report dialect definitions -*- tablegen -*-===//
//
// Report dialect for NEURAX - models final consolidated reports
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Report Dialect
//===----------------------------------------------------------------------===//

def Report_Dialect : Dialect {
  let name = "report";
  let summary = "Final report dialect for NEURAX";
  let description = [{
    This dialect models the final consolidated report including
    all metrics, diagnostics, and recommendations.
  }];
  let cppNamespace = "::mlir::neurax::report";
}

//===----------------------------------------------------------------------===//
// Base class for Report operations
//===----------------------------------------------------------------------===//

class Report_Op<string mnemonic, list<Trait> traits = []> :
    Op<Report_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Report operation
//===----------------------------------------------------------------------===//

def ReportOp : Report_Op<"report"> {
  let summary = "Final analysis report";
  let description = [{
    Top-level operation that contains the complete analysis report.
  }];
  
  let arguments = (ins
    StrAttr:$model_name,
    StrAttr:$model_type,
    StrAttr:$schema_version,
    StrAttr:$neurax_version,
    I64Attr:$analysis_time_ms
  );
  
  let results = (outs);
  
  let regions = (region AnyRegion:$metrics, AnyRegion:$diagnostics, AnyRegion:$recommendations);
  
  let assemblyFormat = [{
    $model_name `:` $model_type attr-dict $metrics $diagnostics $recommendations
  }];
}

//===----------------------------------------------------------------------===//
// All metrics operation
//===----------------------------------------------------------------------===//

def AllMetricsOp : Report_Op<"all_metrics"> {
  let summary = "Consolidated metrics from all dialects";
  let description = [{
    Contains all computed metrics from the analysis pipeline.
  }];
  
  let arguments = (ins
    // Structure
    I64Attr:$total_parameters,
    I64Attr:$num_layers,
    // Compute
    F64Attr:$total_flops,
    F64Attr:$flops_per_token,
    F64Attr:$arithmetic_intensity,
    // Memory
    I64Attr:$peak_vram_bytes,
    I64Attr:$parameter_memory_bytes,
    I64Attr:$activation_memory_bytes,
    // Hardware
    F64Attr:$latency_ms,
    F64Attr:$throughput_tokens_per_s,
    F64Attr:$gpu_utilization,
    // Parallelism
    F64Attr:$data_parallel_efficiency,
    I64Attr:$optimal_gpu_count,
    // Cost
    F64Attr:$training_cost_usd,
    F64Attr:$energy_kwh,
    F64Attr:$co2_kg
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `params` $total_parameters `flops` $total_flops `peak` $peak_vram_bytes attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Diagnostic operation
//===----------------------------------------------------------------------===//

def DiagnosticOp : Report_Op<"diagnostic"> {
  let summary = "Analysis diagnostic";
  let description = [{
    Records a diagnostic finding from the analysis.
  }];
  
  let arguments = (ins
    StrAttr:$category,     // "memory_overflow", "performance_bottleneck", etc.
    StrAttr:$severity,     // "info", "warning", "critical"
    StrAttr:$message,
    OptionalAttr<StrAttr>:$layer_id,
    OptionalAttr<StrAttr>:$suggestion
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $severity `:` $category $message (`layer` $layer_id^)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Recommendation operation
//===----------------------------------------------------------------------===//

def RecommendationOp : Report_Op<"recommendation"> {
  let summary = "Analysis recommendation";
  let description = [{
    Records a recommendation for improving the model.
  }];
  
  let arguments = (ins
    StrAttr:$category,     // "memory_optimization", "compute_optimization", etc.
    StrAttr:$title,
    StrAttr:$description,
    StrAttr:$impact,
    StrAttr:$priority      // "high", "medium", "low"
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $priority `:` $title `-` $description `impact` $impact attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Hardware config operation
//===----------------------------------------------------------------------===//

def HardwareConfigOp : Report_Op<"hw_config"> {
  let summary = "Hardware configuration summary";
  let description = [{
    Records the hardware configuration used for analysis.
  }];
  
  let arguments = (ins
    StrAttr:$gpu_name,
    I64Attr:$gpu_count,
    F64Attr:$gpu_memory_gb,
    F64Attr:$gpu_tflops_fp16,
    F64Attr:$gpu_memory_bandwidth_gbs,
    StrAttr:$interconnect,
    F64Attr:$interconnect_bandwidth_gbs
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $gpu_count `x` $gpu_name `vram` $gpu_memory_gb `GB` attr-dict
  }];
}
