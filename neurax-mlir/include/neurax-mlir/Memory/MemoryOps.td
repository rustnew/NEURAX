//===- MemoryOps.td - Memory dialect definitions -*- tablegen -*-===//
//
// Memory dialect for NEURAX - models memory simulation
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Memory Dialect
//===----------------------------------------------------------------------===//

def Memory_Dialect : Dialect {
  let name = "mem";
  let summary = "Memory simulation dialect for NEURAX";
  let description = [{
    This dialect models memory allocation, liveness analysis, and
    peak VRAM estimation for neural network training.
  }];
  let cppNamespace = "::mlir::neurax::mem";
}

//===----------------------------------------------------------------------===//
// Base class for Memory operations
//===----------------------------------------------------------------------===//

class Memory_Op<string mnemonic, list<Trait> traits = []> :
    Op<Memory_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Liveness operation
//===----------------------------------------------------------------------===//

def LivenessOp : Memory_Op<"liveness"> {
  let summary = "Annotate a tensor with its liveness interval";
  let description = [{
    Associates a tensor with the indices of the first and last
    step in which it is alive.
  }];
  
  let arguments = (ins
    AnyType:$tensor,
    I64Attr:$start_step,
    I64Attr:$end_step,
    I64Attr:$size_bytes
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $tensor `[` $start_step `,` $end_step `]` `size` $size_bytes attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Alloc operation
//===----------------------------------------------------------------------===//

def AllocOp : Memory_Op<"alloc"> {
  let summary = "Memory allocation operation";
  let description = [{
    Represents a memory allocation of a given size.
  }];
  
  let arguments = (ins
    I64Attr:$size_bytes,
    StrAttr:$tensor_id
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $size_bytes `for` $tensor_id attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Peak operation
//===----------------------------------------------------------------------===//

def PeakOp : Memory_Op<"peak"> {
  let summary = "Store the peak memory usage result";
  let description = [{
    Records the peak memory usage at a specific step.
  }];
  
  let arguments = (ins
    I64Attr:$peak_bytes,
    OptionalAttr<I64Attr>:$peak_step
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `peak` `(` $peak_bytes `)` (`at` $peak_step^)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Snapshot operation
//===----------------------------------------------------------------------===//

def SnapshotOp : Memory_Op<"snapshot"> {
  let summary = "Memory snapshot at a step";
  let description = [{
    Records the memory state at a specific step during execution.
  }];
  
  let arguments = (ins
    I64Attr:$step,
    I64Attr:$total_memory,
    StrArrayAttr:$live_tensors
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `step` $step `:` $total_memory `tensors` $live_tensors attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// OOM risk operation
//===----------------------------------------------------------------------===//

def OomRiskOp : Memory_Op<"oom_risk"> {
  let summary = "OOM risk assessment";
  let description = [{
    Records the OOM risk level for the model.
  }];
  
  let arguments = (ins
    StrAttr:$risk_level,  // "safe", "warning", "critical", "overflow"
    F64Attr:$utilization_ratio
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $risk_level `(` $utilization_ratio `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Memory metrics
//===----------------------------------------------------------------------===//

def MemoryMetricsOp : Memory_Op<"metrics"> {
  let summary = "Store memory analysis metrics";
  let description = [{
    Stores the computed memory metrics including parameter memory,
    activation memory, gradients, optimizer states, and peak VRAM.
  }];
  
  let arguments = (ins
    I64Attr:$parameter_memory_bytes,
    I64Attr:$activation_memory_bytes,
    I64Attr:$gradient_memory_bytes,
    I64Attr:$optimizer_state_bytes,
    I64Attr:$peak_vram_bytes,
    I64Attr:$max_batch_size_fit
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `params` `=` $parameter_memory_bytes `,` `activations` `=` $activation_memory_bytes `,` `gradients` `=` $gradient_memory_bytes `,` `optimizer` `=` $optimizer_state_bytes `,` `peak` `=` $peak_vram_bytes `,` `max_batch` `=` $max_batch_size_fit attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Zero Stage operation
//===----------------------------------------------------------------------===//

def ZeroStageOp : Memory_Op<"zero_stage"> {
  let summary = "ZeRO partitioning effect on per-GPU memory";
  let description = [{
    Models the effect of ZeRO (Zero Redundancy Optimizer) partitioning
    on per-GPU memory requirements.
  }];
  
  let arguments = (ins
    I32Attr:$stage,
    I32Attr:$num_gpus,
    F64Attr:$params_per_gpu_gb,
    F64Attr:$total_per_gpu_gb
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `stage` $stage `on` $num_gpus `gpus` `params_per_gpu` $params_per_gpu_gb `gb` `total` $total_per_gpu_gb `gb` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Memory breakdown operation
//===----------------------------------------------------------------------===//

def MemoryBreakdownOp : Memory_Op<"breakdown"> {
  let summary = "Detailed memory breakdown";
  let description = [{
    Provides a detailed breakdown of memory usage by component.
  }];
  
  let arguments = (ins
    I64Attr:$weights_memory,
    I64Attr:$optimizer_states_memory,
    I64Attr:$gradients_memory,
    I64Attr:$activations_memory,
    I64Attr:$temporary_memory
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `weights` $weights_memory `optimizer` $optimizer_states_memory `gradients` $gradients_memory `activations` $activations_memory `temp` $temporary_memory attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Memory hierarchy operation
//===----------------------------------------------------------------------===//

def MemoryHierarchyOp : Memory_Op<"hierarchy"> {
  let summary = "Memory hierarchy configuration";
  let description = [{
    Describes the memory hierarchy with HBM, DDR, and NVMe tiers.
  }];
  
  let arguments = (ins
    F64Attr:$hbm_size_gb,
    F64Attr:$ddr_size_gb,
    F64Attr:$nvme_size_gb
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `hbm` $hbm_size_gb `gb` `ddr` $ddr_size_gb `gb` `nvme` $nvme_size_gb `gb` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Memory offload operation
//===----------------------------------------------------------------------===//

def MemoryOffloadOp : Memory_Op<"offload"> {
  let summary = "Memory offload operation";
  let description = [{
    Describes memory offloading between memory tiers.
  }];
  
  let arguments = (ins
    StrAttr:$source,
    StrAttr:$destination,
    I64Attr:$size_bytes,
    F64Attr:$bandwidth_gbs
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $source `->` $destination `size` $size_bytes `bw` $bandwidth_gbs attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Memory efficiency operation
//===----------------------------------------------------------------------===//

def MemoryEfficiencyOp : Memory_Op<"efficiency"> {
  let summary = "Memory efficiency metrics";
  let description = [{
    Records memory efficiency metrics including utilization and fragmentation.
  }];
  
  let arguments = (ins
    F64Attr:$utilization,
    F64Attr:$fragmentation,
    F64Attr:$bandwidth_utilization
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `util` $utilization `frag` $fragmentation `bw_util` $bandwidth_utilization attr-dict
  }];
}
