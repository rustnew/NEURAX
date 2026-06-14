//===- ComputeOps.td - Compute dialect definitions -*- tablegen -*-===//
//
// Compute dialect for NEURAX - models analytical computation
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Compute Dialect
//===----------------------------------------------------------------------===//

def Compute_Dialect : Dialect {
  let name = "compute";
  let summary = "Analytical computation dialect for NEURAX";
  let description = [{
    This dialect models FLOPs calculations, arithmetic intensity,
    and computational complexity analysis.
  }];
  let cppNamespace = "::mlir::neurax::compute";
}

//===----------------------------------------------------------------------===//
// Base class for Compute operations
//===----------------------------------------------------------------------===//

class Compute_Op<string mnemonic, list<Trait> traits = []> :
    Op<Compute_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// FLOPs operation
//===----------------------------------------------------------------------===//

def FlopsOp : Compute_Op<"flops"> {
  let summary = "FLOPs annotation for an operation";
  let description = [{
    Annotates an operation with its forward and backward FLOPs.
  }];
  
  let arguments = (ins
    AnyType:$operation,
    F64Attr:$forward_flops,
    F64Attr:$backward_flops
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $operation `fwd` $forward_flops `bwd` $backward_flops attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Intensity operation
//===----------------------------------------------------------------------===//

def IntensityOp : Compute_Op<"intensity"> {
  let summary = "Arithmetic intensity calculation";
  let description = [{
    Calculates and stores the arithmetic intensity (FLOPs/byte) for a region.
  }];
  
  let arguments = (ins
    F64Attr:$flops,
    I64Attr:$bytes_accessed,
    F64Attr:$intensity
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `flops` $flops `bytes` $bytes_accessed `=` $intensity attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Complexity operation
//===----------------------------------------------------------------------===//

def ComplexityOp : Compute_Op<"complexity"> {
  let summary = "Computational complexity class";
  let description = [{
    Annotates a region with its computational complexity class.
  }];
  
  let arguments = (ins
    StrAttr:$complexity_class  // "O(n)", "O(n^2)", "O(n^3)", etc.
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $complexity_class attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Compute metrics
//===----------------------------------------------------------------------===//

def ComputeMetricsOp : Compute_Op<"metrics"> {
  let summary = "Store compute analysis metrics";
  let description = [{
    Stores the computed compute metrics including total FLOPs,
    MACs, and intensity.
  }];
  
  let arguments = (ins
    F64Attr:$total_flops,
    F64Attr:$macs,
    F64Attr:$flops_per_token,
    F64Attr:$arithmetic_intensity,
    F64Attr:$forward_flops,
    F64Attr:$backward_flops,
    F64Attr:$optimizer_flops,
    F64Attr:$total_step_flops
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `total` `=` $total_flops `,` `fwd` `=` $forward_flops `,` `bwd` `=` $backward_flops `,` `intensity` `=` $arithmetic_intensity attr-dict
  }];
}
