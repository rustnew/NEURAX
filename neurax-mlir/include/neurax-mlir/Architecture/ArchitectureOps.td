//===- ArchitectureOps.td - Architecture dialect definitions -*- tablegen -*-===//
//
// Architecture dialect for NEURAX - models neural network architecture structure
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Architecture Dialect
//===----------------------------------------------------------------------===//

def Architecture_Dialect : Dialect {
  let name = "arch";
  let summary = "Neural network architecture dialect for NEURAX";
  let description = [{
    This dialect models the high-level structure of neural network architectures,
    including layers, global parameters, and model type information.
  }];
  let cppNamespace = "::mlir::neurax::arch";
}

//===----------------------------------------------------------------------===//
// Base class for Architecture operations
//===----------------------------------------------------------------------===//

class Arch_Op<string mnemonic, list<Trait> traits = []> :
    Op<Architecture_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Model operation
//===----------------------------------------------------------------------===//

def ModelOp : Arch_Op<"model"> {
  let summary = "Define a neural network model";
  let description = [{
    Top-level operation that defines a neural network model with its type,
    name, and global parameters.
  }];
  
  let arguments = (ins
    StrAttr:$name,
    StrAttr:$model_type
  );
  
  let results = (outs);
  
  let regions = (region AnyRegion:$layers);
  
  let assemblyFormat = [{
    $name `:` $model_type attr-dict-with-type-keywords $layers
  }];
}

//===----------------------------------------------------------------------===//
// Layer operation
//===----------------------------------------------------------------------===//

def LayerOp : Arch_Op<"layer"> {
  let summary = "Define a single layer";
  let description = [{
    Represents a single layer in the neural network with its type,
    parameters, and input/output shapes.
  }];
  
  let arguments = (ins
    StrAttr:$id,
    StrAttr:$layer_type,
    OptionalAttr<I64ArrayAttr>:$input_shape,
    OptionalAttr<I64ArrayAttr>:$output_shape
  );
  
  let results = (outs);
  
  let regions = (region AnyRegion:$body);
  
  let assemblyFormat = [{
    $id `:` $layer_type (`in` $input_shape^)? (`out` $output_shape^)? attr-dict $body
  }];
}

//===----------------------------------------------------------------------===//
// Global parameters
//===----------------------------------------------------------------------===//

def GlobalParamsOp : Arch_Op<"global_params"> {
  let summary = "Define global model parameters";
  let description = [{
    Defines global parameters for the model such as hidden size,
    number of layers, sequence length, etc.
  }];
  
  let arguments = (ins
    OptionalAttr<I64Attr>:$sequence_length,
    OptionalAttr<I64Attr>:$vocab_size,
    OptionalAttr<I64Attr>:$embedding_dim,
    OptionalAttr<I64Attr>:$num_layers,
    OptionalAttr<I64Attr>:$num_dense_layers,
    OptionalAttr<I64Attr>:$num_attention_heads,
    OptionalAttr<I64Attr>:$num_experts,
    OptionalAttr<I64Attr>:$intermediate_size
  );
  
  let results = (outs);
  
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Architecture metrics
//===----------------------------------------------------------------------===//

def ArchMetricsOp : Arch_Op<"metrics"> {
  let summary = "Store architecture analysis metrics";
  let description = [{
    Stores the computed architecture metrics including total parameters
    and layer counts.
  }];
  
  let arguments = (ins
    I64Attr:$total_parameters,
    I64Attr:$num_layers
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `params` `=` $total_parameters `,` `layers` `=` $num_layers attr-dict
  }];
}
