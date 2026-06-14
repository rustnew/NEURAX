//===- CostOps.td - Cost dialect definitions -*- tablegen -*-===//
//
// Cost dialect for NEURAX - models economic analysis
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Cost Dialect
//===----------------------------------------------------------------------===//

def Cost_Dialect : Dialect {
  let name = "cost";
  let summary = "Economic analysis dialect for NEURAX";
  let description = [{
    This dialect models training costs, energy consumption,
    and CO2 emissions.
  }];
  let cppNamespace = "::mlir::neurax::cost";
}

//===----------------------------------------------------------------------===//
// Base class for Cost operations
//===----------------------------------------------------------------------===//

class Cost_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cost_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Pricing model operation
//===----------------------------------------------------------------------===//

def PricingModelOp : Cost_Op<"pricing"> {
  let summary = "Pricing model definition";
  let description = [{
    Defines the pricing model for cost calculations.
  }];
  
  let arguments = (ins
    F64Attr:$gpu_hour_usd,
    F64Attr:$energy_kwh_usd,
    F64Attr:$pue_factor,
    F64Attr:$gpu_tdp_watts,
    F64Attr:$co2_per_kwh
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `gpu` $gpu_hour_usd `$/h` `energy` $energy_kwh_usd `$/kWh` `pue` $pue_factor attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Training cost operation
//===----------------------------------------------------------------------===//

def TrainingCostOp : Cost_Op<"training"> {
  let summary = "Training cost calculation";
  let description = [{
    Records the training cost and time.
  }];
  
  let arguments = (ins
    F64Attr:$training_time_hours,
    F64Attr:$training_cost_usd,
    F64Attr:$gpu_hours_total
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $training_time_hours `h` `$` $training_cost_usd `gpu_hours` $gpu_hours_total attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Energy operation
//===----------------------------------------------------------------------===//

def EnergyOp : Cost_Op<"energy"> {
  let summary = "Energy consumption";
  let description = [{
    Records energy consumption and CO2 emissions.
  }];
  
  let arguments = (ins
    F64Attr:$energy_kwh,
    F64Attr:$co2_kg
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $energy_kwh `kWh` `co2` $co2_kg `kg` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Token cost operation
//===----------------------------------------------------------------------===//

def TokenCostOp : Cost_Op<"token_cost"> {
  let summary = "Cost per token";
  let description = [{
    Records the cost per token for inference.
  }];
  
  let arguments = (ins
    F64Attr:$cost_per_token_usd,
    F64Attr:$cost_per_million_tokens_usd
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    $cost_per_million_tokens_usd `$/M` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Cost metrics
//===----------------------------------------------------------------------===//

def CostMetricsOp : Cost_Op<"metrics"> {
  let summary = "Store cost analysis metrics";
  let description = [{
    Stores the computed cost metrics including training cost,
    energy, and CO2.
  }];
  
  let arguments = (ins
    F64Attr:$training_cost_usd,
    F64Attr:$training_time_hours,
    F64Attr:$energy_kwh,
    F64Attr:$co2_kg,
    F64Attr:$cost_per_million_tokens_usd
  );
  
  let results = (outs);
  
  let assemblyFormat = [{
    `$` $training_cost_usd `,` $training_time_hours `h` `,` $energy_kwh `kWh` `,` $co2_kg `kg CO2` attr-dict
  }];
}
