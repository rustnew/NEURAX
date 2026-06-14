//===- OperatorOps.td - Operator dialect definitions -*- tablegen -*-===//
//
// Operator dialect for NEURAX - models ML operations
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Operator Dialect
//===----------------------------------------------------------------------===//

def Operator_Dialect : Dialect {
  let name = "op";
  let summary = "ML operator dialect for NEURAX";
  let description = [{
    This dialect models atomic ML operations like MatMul, Conv2D, Attention,
    normalization, and activation functions.
  }];
  let cppNamespace = "::mlir::neurax::op";
}

//===----------------------------------------------------------------------===//
// Base class for Operator operations
//===----------------------------------------------------------------------===//

class Operator_Op<string mnemonic, list<Trait> traits = []> :
    Op<Operator_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// MatMul operation
//===----------------------------------------------------------------------===//

def MatMulOp : Operator_Op<"matmul"> {
  let summary = "Matrix multiplication operation";
  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    I64Attr:$param_count,
    F64Attr:$flops
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $lhs `,` $rhs `:` type($lhs) `params` $param_count `flops` $flops attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Conv2D operation
//===----------------------------------------------------------------------===//

def Conv2DOp : Operator_Op<"conv2d"> {
  let summary = "2D convolution operation";
  let arguments = (ins
    AnyType:$input,
    AnyType:$kernel,
    I64Attr:$in_channels,
    I64Attr:$out_channels,
    I64ArrayAttr:$kernel_size,
    I64ArrayAttr:$stride,
    I64ArrayAttr:$padding,
    I64Attr:$param_count,
    F64Attr:$flops
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $input `,` $kernel `:` type($input) `channels` $in_channels `->` $out_channels 
    `kernel` $kernel_size `stride` $stride `pad` $padding attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Attention operation
//===----------------------------------------------------------------------===//

def AttentionOp : Operator_Op<"attention"> {
  let summary = "Multi-head attention operation";
  let arguments = (ins
    AnyType:$query,
    AnyType:$key,
    AnyType:$value,
    I64Attr:$hidden_size,
    I64Attr:$num_heads,
    I64Attr:$param_count,
    F64Attr:$flops
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $query `,` $key `,` $value `:` type($query) `hidden` $hidden_size `heads` $num_heads attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// MoE operation
//===----------------------------------------------------------------------===//

def MoEOp : Operator_Op<"moe"> {
  let summary = "Mixture of Experts operation";
  let arguments = (ins
    AnyType:$input,
    I64Attr:$hidden_size,
    I64Attr:$intermediate_size,
    I64Attr:$num_experts,
    I64Attr:$top_k,
    I64Attr:$param_count,
    F64Attr:$flops
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $input `:` type($input) `experts` $num_experts `top_k` $top_k `hidden` $hidden_size attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Normalization operations
//===----------------------------------------------------------------------===//

def LayerNormOp : Operator_Op<"layer_norm"> {
  let summary = "Layer normalization operation";
  let arguments = (ins
    AnyType:$input,
    I64Attr:$hidden_size,
    I64Attr:$param_count
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $input `:` type($input) `hidden` $hidden_size attr-dict `:` type($result)
  }];
}

def RMSNormOp : Operator_Op<"rms_norm"> {
  let summary = "RMS normalization operation";
  let arguments = (ins
    AnyType:$input,
    I64Attr:$hidden_size,
    I64Attr:$param_count
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $input `:` type($input) `hidden` $hidden_size attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Activation functions
//===----------------------------------------------------------------------===//

def GELUOp : Operator_Op<"gelu"> {
  let summary = "GELU activation function";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `:` type($input) attr-dict `:` type($result)";
}

def SiLUOp : Operator_Op<"silu"> {
  let summary = "SiLU/Swish activation function";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `:` type($input) attr-dict `:` type($result)";
}

def ReLUOp : Operator_Op<"relu"> {
  let summary = "ReLU activation function";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `:` type($input) attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Operator metrics
//===----------------------------------------------------------------------===//

def OperatorMetricsOp : Operator_Op<"metrics"> {
  let summary = "Store operator analysis metrics";
  let arguments = (ins
    F64Attr:$total_flops_approx,
    I64Attr:$total_op_count,
    I64Attr:$custom_op_count
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `flops` `=` $total_flops_approx `,` `ops` `=` $total_op_count attr-dict
  }];
}
