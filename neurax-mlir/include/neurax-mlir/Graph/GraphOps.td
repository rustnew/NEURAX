//===- GraphOps.td - Graph dialect definitions -*- tablegen -*-===//
//
// Graph dialect for NEURAX - models computation graphs
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Graph Dialect
//===----------------------------------------------------------------------===//

def Graph_Dialect : Dialect {
  let name = "graph";
  let summary = "Computation graph dialect for NEURAX";
  let description = [{
    This dialect models computation graphs as DAGs, including nodes,
    edges, and topological ordering.
  }];
  let cppNamespace = "::mlir::neurax::graph";
}

//===----------------------------------------------------------------------===//
// Base class for Graph operations
//===----------------------------------------------------------------------===//

class Graph_Op<string mnemonic, list<Trait> traits = []> :
    Op<Graph_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Node operation
//===----------------------------------------------------------------------===//

def NodeOp : Graph_Op<"node"> {
  let summary = "A node in the computation graph";
  let description = [{
    Represents a node in the computation graph corresponding to a layer
    or operation.
  }];
  
  let arguments = (ins
    StrAttr:$layer_id,
    StrAttr:$layer_type,
    F64Attr:$flops_approx,
    I64Attr:$param_count
  );
  
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    $layer_id `:` $layer_type `flops` $flops_approx `params` $param_count attr-dict `:` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Edge operation
//===----------------------------------------------------------------------===//

def EdgeOp : Graph_Op<"edge"> {
  let summary = "An edge connecting two nodes";
  let description = [{
    Represents a data flow edge between two nodes in the computation graph.
  }];
  
  let arguments = (ins
    AnyType:$from,
    AnyType:$to,
    I64ArrayAttr:$tensor_shape,
    StrAttr:$dtype,
    I64Attr:$size_bytes
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $from `->` $to `:` type($from) `shape` $tensor_shape `dtype` $dtype `size` $size_bytes attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Graph metrics
//===----------------------------------------------------------------------===//

def GraphMetricsOp : Graph_Op<"metrics"> {
  let summary = "Store graph analysis metrics";
  let description = [{
    Stores the computed graph metrics including depth, operations count,
    and parallel paths.
  }];
  
  let arguments = (ins
    I64Attr:$graph_depth,
    I64Attr:$total_operations,
    I64Attr:$total_intermediate_tensors,
    I64Attr:$critical_path_length,
    I64Attr:$edge_count
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `depth` `=` $graph_depth `,` `ops` `=` $total_operations `,` `tensors` `=` $total_intermediate_tensors attr-dict
  }];
}
