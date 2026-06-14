//===- TensorOps.td - Tensor dialect definitions -*- tablegen -*-===//
//
// Tensor dialect for NEURAX - models tensor shapes and propagation
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Tensor Dialect
//===----------------------------------------------------------------------===//

def Tensor_Dialect : Dialect {
  let name = "tensor";
  let summary = "Tensor shape propagation dialect for NEURAX";
  let description = [{
    This dialect models tensor shapes, dtypes, and their propagation
    through the neural network layers.
  }];
  let cppNamespace = "::mlir::neurax::tensor";
}

//===----------------------------------------------------------------------===//
// Base class for Tensor operations
//===----------------------------------------------------------------------===//

class Tensor_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tensor_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Tensor info operation
//===----------------------------------------------------------------------===//

def TensorInfoOp : Tensor_Op<"info"> {
  let summary = "Tensor information with shape and dtype";
  let description = [{
    Defines a tensor with its shape, dtype, and size in bytes.
  }];
  
  let arguments = (ins
    StrAttr:$tensor_id,
    I64ArrayAttr:$shape,
    StrAttr:$dtype,
    I64Attr:$size_bytes,
    StrAttr:$produced_by
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $tensor_id `:` $dtype `shape` $shape `size` $size_bytes `from` $produced_by attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Shape operation
//===----------------------------------------------------------------------===//

def ShapeOp : Tensor_Op<"shape"> {
  let summary = "Shape with symbolic dimensions";
  let description = [{
    Defines a shape that may contain symbolic or dynamic dimensions.
  }];
  
  let arguments = (ins
    I64ArrayAttr:$dims,
    StrArrayAttr:$dim_names
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $dims `names` $dim_names attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Tensor metrics
//===----------------------------------------------------------------------===//

def TensorMetricsOp : Tensor_Op<"metrics"> {
  let summary = "Store tensor analysis metrics";
  let description = [{
    Stores the computed tensor metrics including activation memory
    and tensor size distribution.
  }];
  
  let arguments = (ins
    I64Attr:$activation_memory_bytes,
    F64Attr:$memory_bandwidth_required,
    I64Attr:$total_tensor_count,
    I64Attr:$largest_tensor_bytes
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `activation` `=` $activation_memory_bytes `,` `bandwidth` `=` $memory_bandwidth_required attr-dict
  }];
}
