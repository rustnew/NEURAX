//===- ParallelismOps.td - Parallelism dialect definitions -*- tablegen -*-===//
//
// Parallelism dialect for NEURAX - models parallelism strategies
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Parallelism Dialect
//===----------------------------------------------------------------------===//

def Parallelism_Dialect : Dialect {
  let name = "par";
  let summary = "Parallelism strategy dialect for NEURAX";
  let description = [{
    This dialect models parallelism strategies including data parallel,
    model parallel, pipeline parallel, tensor parallel, and ZeRO.
  }];
  let cppNamespace = "::mlir::neurax::par";
}

//===----------------------------------------------------------------------===//
// Base class for Parallelism operations
//===----------------------------------------------------------------------===//

class Parallelism_Op<string mnemonic, list<Trait> traits = []> :
    Op<Parallelism_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Data parallel operation
//===----------------------------------------------------------------------===//

def DataParallelOp : Parallelism_Op<"data_parallel"> {
  let summary = "Data parallel strategy";
  let description = [{
    Defines a data parallel strategy with number of GPUs and efficiency.
  }];
  
  let arguments = (ins
    I64Attr:$num_gpus,
    F64Attr:$efficiency
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $num_gpus `gpus` `efficiency` $efficiency attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Tensor parallel operation
//===----------------------------------------------------------------------===//

def TensorParallelOp : Parallelism_Op<"tensor_parallel"> {
  let summary = "Tensor parallel strategy";
  let description = [{
    Defines a tensor parallel strategy with degree.
  }];
  
  let arguments = (ins
    I64Attr:$tp_degree
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `degree` $tp_degree attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Pipeline parallel operation
//===----------------------------------------------------------------------===//

def PipelineParallelOp : Parallelism_Op<"pipeline_parallel"> {
  let summary = "Pipeline parallel strategy";
  let description = [{
    Defines a pipeline parallel strategy with stages and micro-batches.
  }];
  
  let arguments = (ins
    I64Attr:$stages,
    I64Attr:$micro_batches,
    F64Attr:$bubble_ratio
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $stages `stages` $micro_batches `micro_batches` `bubble` $bubble_ratio attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ZeRO operation
//===----------------------------------------------------------------------===//

def ZeroOp : Parallelism_Op<"zero"> {
  let summary = "ZeRO optimization strategy";
  let description = [{
    Defines a ZeRO stage with memory per GPU.
  }];
  
  let arguments = (ins
    I64Attr:$stage,
    I64Attr:$memory_per_gpu_bytes
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `stage` $stage `mem_per_gpu` $memory_per_gpu_bytes attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Hybrid parallel operation
//===----------------------------------------------------------------------===//

def HybridOp : Parallelism_Op<"hybrid"> {
  let summary = "Hybrid parallel strategy";
  let description = [{
    Defines a hybrid strategy combining DP, TP, and PP.
  }];
  
  let arguments = (ins
    I64Attr:$dp,
    I64Attr:$tp,
    I64Attr:$pp
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `dp` $dp `tp` $tp `pp` $pp attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Communication operation
//===----------------------------------------------------------------------===//

def CommunicationOp : Parallelism_Op<"comm"> {
  let summary = "Communication overhead";
  let description = [{
    Records communication overhead for a parallel strategy.
  }];
  
  let arguments = (ins
    F64Attr:$allreduce_time_ms,
    F64Attr:$overhead_fraction
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `allreduce` $allreduce_time_ms `ms` `overhead` $overhead_fraction attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Parallelism metrics
//===----------------------------------------------------------------------===//

def ParallelismMetricsOp : Parallelism_Op<"metrics"> {
  let summary = "Store parallelism analysis metrics";
  let description = [{
    Stores the computed parallelism metrics including efficiency
    and optimal GPU count.
  }];
  
  let arguments = (ins
    F64Attr:$data_parallel_efficiency,
    F64Attr:$communication_overhead,
    I64Attr:$optimal_gpu_count,
    I64Attr:$memory_per_gpu_bytes
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `efficiency` `=` $data_parallel_efficiency `,` `optimal_gpus` `=` $optimal_gpu_count attr-dict
  }];
}
