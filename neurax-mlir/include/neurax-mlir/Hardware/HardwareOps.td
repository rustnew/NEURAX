//===- HardwareOps.td - Hardware dialect definitions -*- tablegen -*-===//
//
// Hardware dialect for NEURAX - models GPU hardware simulation
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Hardware Dialect
//===----------------------------------------------------------------------===//

def Hardware_Dialect : Dialect {
  let name = "hw";
  let summary = "Hardware simulation dialect for NEURAX";
  let description = [{
    This dialect models GPU hardware profiles, roofline models,
    and timing analysis.
  }];
  let cppNamespace = "::mlir::neurax::hw";
}

//===----------------------------------------------------------------------===//
// Base class for Hardware operations
//===----------------------------------------------------------------------===//

class Hardware_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hardware_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// GPU profile operation
//===----------------------------------------------------------------------===//

def GpuProfileOp : Hardware_Op<"gpu"> {
  let summary = "GPU profile definition";
  let description = [{
    Defines a GPU profile with its compute and memory capabilities.
  }];
  
  let arguments = (ins
    StrAttr:$name,
    F64Attr:$peak_tflops,
    F64Attr:$memory_bandwidth,
    F64Attr:$tensor_core_tflops,
    F64Attr:$efficiency_factor,
    I64Attr:$vram_gb
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $name `tflops` $peak_tflops `bw` $memory_bandwidth `vram` $vram_gb attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Roofline operation
//===----------------------------------------------------------------------===//

def RooflineOp : Hardware_Op<"roofline"> {
  let summary = "Roofline model for a region";
  let description = [{
    Defines the roofline model constraints for a computational region.
  }];
  
  let arguments = (ins
    F64Attr:$compute_roof,
    F64Attr:$memory_roof,
    F64Attr:$ridge_point
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `compute` $compute_roof `memory` $memory_roof `ridge` $ridge_point attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Timing operation
//===----------------------------------------------------------------------===//

def TimingOp : Hardware_Op<"timing"> {
  let summary = "Layer timing information";
  let description = [{
    Records compute and memory timing for a layer.
  }];
  
  let arguments = (ins
    StrAttr:$layer_id,
    F64Attr:$compute_time_ms,
    F64Attr:$memory_time_ms,
    F64Attr:$total_time_ms
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $layer_id `compute` $compute_time_ms `ms` `memory` $memory_time_ms `ms` `total` $total_time_ms `ms` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Bottleneck operation
//===----------------------------------------------------------------------===//

def BottleneckOp : Hardware_Op<"bottleneck"> {
  let summary = "Bottleneck analysis";
  let description = [{
    Records the identified bottleneck type for a region.
  }];
  
  let arguments = (ins
    StrAttr:$bottleneck_type  // "compute-bound", "memory-bound", "balanced"
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $bottleneck_type attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Hardware metrics
//===----------------------------------------------------------------------===//

def HardwareMetricsOp : Hardware_Op<"metrics"> {
  let summary = "Store hardware analysis metrics";
  let description = [{
    Stores the computed hardware metrics including latency,
    throughput, and utilization.
  }];
  
  let arguments = (ins
    F64Attr:$latency_ms,
    F64Attr:$throughput_tokens_per_s,
    F64Attr:$gpu_utilization,
    F64Attr:$tensor_core_utilization,
    F64Attr:$effective_tflops,
    F64Attr:$memory_bandwidth_achieved
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `latency` `=` $latency_ms `ms` `,` `throughput` `=` $throughput_tokens_per_s `tok/s` `,` `util` `=` $gpu_utilization attr-dict
  }];
}
