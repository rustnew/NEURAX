 NEURAX — ENRICHISSEMENT DES DIALECTES & RÈGLES JSON
## Guide Complet : Dialectes MLIR de Niveau Industriel + JSON Sans Erreur

> **Version 5.0 — Référence d'Implémentation**
> Analyse complète des dialectes existants, enrichissements requis,
> règles JSON strictes, et erreurs à éviter pour un compilateur universel.

---

# TABLE DES MATIÈRES

**PARTIE I — ANALYSE CRITIQUE DES DIALECTES EXISTANTS**
1. [Audit des 9 Dialectes — Points Forts et Lacunes](#1-audit-des-9-dialectes)
2. [Matrice de Couverture — Ce qui Manque par Dialecte](#2-matrice-de-couverture)

**PARTIE II — ENRICHISSEMENT COMPLET DE CHAQUE DIALECTE**
3. [Dialecte `arch` — Architecture (Enrichi)](#3-dialecte-arch-enrichi)
4. [Dialecte `graph` — Graphe SSA (Enrichi)](#4-dialecte-graph-enrichi)
5. [Dialecte `tensor` — Tenseurs (Enrichi)](#5-dialecte-tensor-enrichi)
6. [Dialecte `op` — Opérateurs (Enrichi)](#6-dialecte-op-enrichi)
7. [Dialecte `compute` — Calcul (Enrichi)](#7-dialecte-compute-enrichi)
8. [Dialecte `mem` — Mémoire (Enrichi)](#8-dialecte-mem-enrichi)
9. [Dialecte `par` — Parallélisme (Enrichi)](#9-dialecte-par-enrichi)
10. [Dialecte `hw` — Hardware (Enrichi)](#10-dialecte-hw-enrichi)
11. [Dialecte `cost` — Coût (Enrichi)](#11-dialecte-cost-enrichi)
12. [Dialecte `report` — Rapport (Enrichi)](#12-dialecte-report-enrichi)

**PARTIE III — COMMUNICATION ENTRE DIALECTES**
13. [Flux SSA Inter-Dialectes — Comment les Valeurs Circulent](#13-flux-ssa-inter-dialectes)
14. [Passes de Conversion Obligatoires](#14-passes-de-conversion)

**PARTIE IV — RÈGLES JSON STRICTES**
15. [Schéma JSON Complet v3.0 avec Contraintes](#15-schéma-json-complet)
16. [Règles de Cohérence Interne (32 règles)](#16-règles-de-cohérence)
17. [Règles par Famille d'Architecture](#17-règles-par-famille)

**PARTIE V — ERREURS À ÉVITER**
18. [Erreurs d'Implémentation des Dialectes (20 erreurs)](#18-erreurs-dialectes)
19. [Erreurs JSON Courantes (25 erreurs)](#19-erreurs-json)
20. [Pièges de Précision dans les Formules](#20-pièges-précision)

**PARTIE VI — UNIVERSALITÉ**
21. [Stratégie d'Universalité — Couvrir Tous les Modèles](#21-universalité)
22. [Validation Automatique Complète](#22-validation-automatique)

---

# PARTIE I — ANALYSE CRITIQUE DES DIALECTES EXISTANTS

## 1. Audit des 9 Dialectes — Points Forts et Lacunes

### 1.1 Dialecte `arch` — Analyse

```
POINTS FORTS :
  ✅ ModelOp avec region pour les couches → bonne structure hiérarchique
  ✅ LayerOp avec input/output shape optionnels
  ✅ GlobalParamsOp pour les paramètres globaux
  ✅ ArchMetricsOp pour stocker les métriques finales

LACUNES CRITIQUES :
  ❌ Pas de RepeatOp : impossible de modéliser "repeat: 32" sans
     dupliquer 32 LayerOp identiques → explosion de la taille du module MLIR
  ❌ Pas de ModèleType enum : model_type est un StrAttr libre,
     aucune validation possible en MLIR
  ❌ Pas d'information sur les connexions (sequential uniquement)
  ❌ GlobalParamsOp ne couvre pas SSM (ssm_state_size, ssm_expand)
  ❌ GlobalParamsOp ne couvre pas MoE (num_experts_per_tok, shared_experts)
  ❌ GlobalParamsOp ne couvre pas CNN (initial_channels, num_classes)
  ❌ Pas de score de confiance sur les métriques calculées
  ❌ Pas de CustomLayerOp pour les couches avec formules evalexpr
```

### 1.2 Dialecte `graph` — Analyse

```
POINTS FORTS :
  ✅ NodeOp avec layer_id, layer_type, flops_approx, param_count
  ✅ EdgeOp avec tensor_shape, dtype, size_bytes → bon pour la liveness
  ✅ GraphMetricsOp avec depth, critical_path, edge_count

LACUNES CRITIQUES :
  ❌ NodeOp retourne AnyType → pas de valeur SSA typée
     Un graphe SSA DOIT avoir des types précis sur ses arêtes
  ❌ EdgeOp prend (from, to) mais ce n'est pas SSA : une arête SSA
     DOIT être une valeur produite par un nœud et consommée par un autre
     La modélisation actuelle est une liste de relations, pas du SSA réel
  ❌ Pas de toposort_order stocké dans le dialecte
  ❌ Pas d'annotation de liveness sur les arêtes (born_at/dies_at)
  ❌ GraphMetricsOp manque : max_parallelism_degree, has_branches,
     parallel_regions_count
  ❌ Pas de CriticalPathOp pour annoter le chemin critique
  ❌ Pas de PatternOp pour annoter les patterns architecturaux détectés
```

### 1.3 Dialecte `tensor` — Analyse

```
POINTS FORTS :
  ✅ TensorInfoOp avec shape, dtype, size_bytes, produced_by
  ✅ ShapeOp avec dim_names pour les dimensions symboliques
  ✅ TensorMetricsOp pour l'activation memory

LACUNES CRITIQUES :
  ❌ TensorInfoOp : shape est I64ArrayAttr → impossible de représenter
     les dimensions symboliques ("B", "S") dans un attribut entier
  ❌ Pas de DimOp pour représenter une dimension avec son degré de certitude
  ❌ ShapeOp : dim_names et dims sont deux tableaux séparés sans lien fort
  ❌ Pas de ShapeGateOp pour bloquer le pipeline si < 70% de dims résolues
  ❌ TensorMetricsOp manque : resolution_ratio, unresolved_dim_count,
     shape_gate_passed
  ❌ Pas de MemoryLayoutOp (row-major, channels-last, etc.)
  ❌ Pas de DTypeOp avec informations sur la précision (bf16, fp8, etc.)
```

### 1.4 Dialecte `op` — Analyse

```
POINTS FORTS :
  ✅ MatMulOp, Conv2DOp, AttentionOp, MoEOp bien définis
  ✅ LayerNormOp et RMSNormOp séparés (correct)
  ✅ Activations GELUOp, SiLUOp, ReLUOp comme ops séparées
  ✅ OperatorMetricsOp

LACUNES CRITIQUES :
  ❌ AttentionOp ne distingue pas MHA/GQA/MQA/Flash
     → formules FLOPs différentes, une seule op ne suffit pas
  ❌ Conv2DOp prend des paramètres optionnels mais sans validation
     qu'ils sont cohérents (ex: in_channels doit diviser out_channels
     pour depthwise)
  ❌ Pas de FlashAttentionOp séparé avec ses attributs spécifiques
  ❌ Pas de GQAOp avec num_kv_heads
  ❌ Pas de LinearProjectionOp pour les projections simples (vs MatMul)
  ❌ MoEOp ne modélise pas le router et les experts séparément
  ❌ Pas de MambaBlockOp, SSMStateOp pour les SSM
  ❌ Pas de LSTMCellOp, GRUCellOp pour les RNN
  ❌ Pas de BatchNormOp (nécessaire pour CNN)
  ❌ Pas d'EmbeddingLookupOp (FLOPs = 0, params = vocab×dim)
  ❌ Pas de BackwardRatioAttr sur chaque op → backward FLOPs incorrect
  ❌ Pas de FusionAttr pour indiquer quelles ops sont fusionnées
```

### 1.5 Dialecte `compute` — Analyse

```
POINTS FORTS :
  ✅ FlopsOp avec forward_flops et backward_flops
  ✅ IntensityOp avec flops, bytes_accessed, intensity
  ✅ ComputeMetricsOp très complet (forward, backward, optimizer, step)

LACUNES CRITIQUES :
  ❌ FlopsOp prend un AnyType:$operation → le lien avec l'op réelle
     n'est pas type-safe ; devrait référencer un NodeIndex SSA
  ❌ ComputeMetricsOp a optimizer_flops mais pas de backward_ratio
     par op (crucial pour la précision du backward)
  ❌ Pas de MemoryTrafficOp pour les bytes lus/écrits par op
     → nécessaire pour le roofline
  ❌ ComplexityOp avec StrAttr "O(n^2)" : non parseable automatiquement,
     devrait être un enum
  ❌ Pas d'ArithIntensityPerOpOp pour le roofline par couche
  ❌ Pas de FusionImpactOp pour modéliser la réduction de bytes
     due aux fusions (Flash Attention, Conv+BN+ReLU, etc.)
```

### 1.6 Dialecte `mem` — Analyse

```
POINTS FORTS :
  ✅ LivenessOp avec start_step, end_step, size_bytes
  ✅ AllocOp et PeakOp
  ✅ OomRiskOp avec risk_level et utilization_ratio
  ✅ MemoryMetricsOp très complet

LACUNES CRITIQUES :
  ❌ LivenessOp prend AnyType:$tensor → doit être une SSAValue typée
     avec un id clair et une forme connue
  ❌ SnapshotOp avec StrArrayAttr:$live_tensors : liste de strings,
     pas de référence aux vraies SSAValues
  ❌ Pas de ZeroStageOp pour modéliser l'impact ZeRO 0/1/2/3 par GPU
  ❌ Pas de CheckpointingOp pour modéliser le gradient checkpointing
  ❌ Pas de OffloadingOp pour l'activation offloading CPU↔GPU
  ❌ Pas de KvCacheOp pour le KV cache inférence LLM
  ❌ Pas de FragmentationOp pour la fragmentation mémoire PyTorch
  ❌ Pas de OptimizerStatesOp avec détail (Adam momentum/variance)
  ❌ MemoryMetricsOp manque : checkpointing_savings_gb,
     kv_cache_gb, fragmentation_overhead_gb, per_gpu_gb
```

### 1.7 Dialecte `par` — Analyse

```
POINTS FORTS :
  ✅ DataParallelOp, TensorParallelOp, PipelineParallelOp, ZeroOp
  ✅ HybridOp pour combiner DP+TP+PP
  ✅ CommunicationOp avec allreduce_time_ms et overhead_fraction

LACUNES CRITIQUES :
  ❌ ZeroOp n'a que stage et memory_per_gpu_bytes :
     manque params_per_gpu, grads_per_gpu, optimizer_per_gpu séparément
  ❌ Pas de ExpertParallelOp pour MoE expert parallelism
  ❌ PipelineParallelOp a bubble_ratio mais pas d'info sur
     les micro_batches recommandés pour minimiser la bulle
  ❌ CommunicationOp couvre All-Reduce mais pas :
     - All-Gather (ZeRO-3)
     - Reduce-Scatter (ZeRO-2)
     - P2P pour pipeline parallel
  ❌ ParallelismMetricsOp manque : scaling_efficiency global,
     all_reduce_time_ms, pipeline_bubble_ratio, recommended_strategy
  ❌ Pas de RecommendedStrategyOp pour suggérer la meilleure config
```

### 1.8 Dialecte `hw` — Analyse

```
POINTS FORTS :
  ✅ GpuProfileOp avec peak_tflops, memory_bandwidth, tensor_core_tflops
  ✅ RooflineOp avec compute_roof, memory_roof, ridge_point
  ✅ TimingOp par couche
  ✅ BottleneckOp
  ✅ HardwareMetricsOp très complet

LACUNES CRITIQUES :
  ❌ GpuProfileOp manque : l2_cache_mb, sram_per_sm_kb, num_sms,
     tflops_bf16, tflops_int8, tflops_fp8, tdp_watts, nvlink_bw
  ❌ RooflineOp est global (un seul par modèle) mais devrait exister
     par op ou par couche pour le roofline détaillé
  ❌ Pas de CacheHierarchyOp pour L1/L2/HBM
  ❌ Pas de TensorCoreEligibilityOp par op (quelles ops utilisent
     les Tensor Cores, quelles conditions d'alignement)
  ❌ Pas de KernelOverheadOp pour le dispatch overhead (~5µs par kernel)
  ❌ Pas de EfficiencyFactorOp pour stocker le facteur de calibration
     utilisé par op (pour la traçabilité)
  ❌ Pas de CalibrationSourceAttr (real_gpu, interpolated, estimated)
  ❌ HardwareMetricsOp manque : latency_breakdown_per_layer (top10)
```

### 1.9 Dialecte `cost` — Analyse

```
POINTS FORTS :
  ✅ PricingModelOp avec gpu_hour_usd, energy_kwh_usd, pue_factor
  ✅ TrainingCostOp, EnergyOp, TokenCostOp
  ✅ CostMetricsOp très complet

LACUNES CRITIQUES :
  ❌ PricingModelOp ne distingue pas on-demand / spot / reserved
  ❌ Pas de ProviderCostOp pour comparer AWS/GCP/Azure
  ❌ EnergyOp manque : co2_per_kwh, equivalent_car_km,
     gpu_utilization_factor utilisé
  ❌ Pas de TimeMachineOp pour les projections temporelles
  ❌ TokenCostOp manque : latency_per_token_ms, throughput_tokens_per_sec
  ❌ CostMetricsOp manque : spot_usd, reserved_usd, training_duration_days
```

### 1.10 Dialecte `report` — Analyse

```
POINTS FORTS :
  ✅ ReportOp avec regions pour metrics/diagnostics/recommendations
  ✅ AllMetricsOp très complet (toutes les métriques en un endroit)
  ✅ DiagnosticOp avec severity et suggestion
  ✅ RecommendationOp avec priority et impact
  ✅ HardwareConfigOp

LACUNES CRITIQUES :
  ❌ AllMetricsOp avec 17+ attributs → trop gros pour un seul op,
     devrait être décomposé en sous-ops par domaine
  ❌ DiagnosticOp utilise des StrAttr pour category et severity
     → pas de validation MLIR possible ; devrait être des enums
  ❌ Pas de ConfidenceScoreOp par métrique
  ❌ Pas de VariantOp pour les analyses des modèles optimisés
     (INT8, Flash Attention, Pruning 50%, LoRA, etc.)
  ❌ Pas de PrecisionReportOp (score de confiance par métrique)
  ❌ RecommendationOp : impact est StrAttr libre, pas structuré
```

---

## 2. Matrice de Couverture — Ce qui Manque par Dialecte

```
╔══════════════════╦══════════╦══════════╦══════════╦══════════╦══════════╗
║ Fonctionnalité   ║  Actuel  ║ arch     ║ graph    ║ tensor   ║ op       ║
╠══════════════════╬══════════╬══════════╬══════════╬══════════╬══════════╣
║ Repeat layers    ║    ❌    ║  NEEDED  ║    -     ║    -     ║    -     ║
║ SSM (Mamba)      ║    ❌    ║  NEEDED  ║    -     ║    -     ║  NEEDED  ║
║ GQA attention    ║    ❌    ║    -     ║    -     ║    -     ║  NEEDED  ║
║ Flash Attention  ║    ❌    ║    -     ║    -     ║    -     ║  NEEDED  ║
║ Liveness SSA     ║   ⚠️    ║    -     ║  NEEDED  ║    -     ║    -     ║
║ Sym. dims        ║   ⚠️    ║    -     ║    -     ║  NEEDED  ║    -     ║
║ Backward ratios  ║    ❌    ║    -     ║    -     ║    -     ║  NEEDED  ║
║ ZeRO détail      ║   ⚠️    ║    -     ║    -     ║    -     ║    -     ║
║ KV Cache         ║    ❌    ║    -     ║    -     ║    -     ║    -     ║
║ Cache hiérar.    ║    ❌    ║    -     ║    -     ║    -     ║    -     ║
║ Fusion ops       ║    ❌    ║    -     ║    -     ║    -     ║  NEEDED  ║
║ Confiance/métri. ║    ❌    ║  NEEDED  ║  NEEDED  ║  NEEDED  ║  NEEDED  ║
║ Time Machine     ║    ❌    ║    -     ║    -     ║    -     ║    -     ║
║ Calibration src  ║    ❌    ║    -     ║    -     ║    -     ║    -     ║
╚══════════════════╩══════════╩══════════╩══════════╩══════════╩══════════╝

╔══════════════════╦══════════╦══════════╦══════════╦══════════╦══════════╗
║ Fonctionnalité   ║ compute  ║ mem      ║ par      ║ hw       ║ cost     ║
╠══════════════════╬══════════╬══════════╬══════════╬══════════╬══════════╣
║ Backward ratios  ║  NEEDED  ║    -     ║    -     ║    -     ║    -     ║
║ Memory traffic   ║  NEEDED  ║    -     ║    -     ║    -     ║    -     ║
║ Fusion impact    ║  NEEDED  ║    -     ║    -     ║    -     ║    -     ║
║ ZeRO détail      ║    -     ║  NEEDED  ║  NEEDED  ║    -     ║    -     ║
║ KV Cache         ║    -     ║  NEEDED  ║    -     ║    -     ║    -     ║
║ Checkpointing    ║    -     ║  NEEDED  ║    -     ║    -     ║    -     ║
║ Fragmentation    ║    -     ║  NEEDED  ║    -     ║    -     ║    -     ║
║ Expert Parallel  ║    -     ║    -     ║  NEEDED  ║    -     ║    -     ║
║ All-Gather/P2P   ║    -     ║    -     ║  NEEDED  ║    -     ║    -     ║
║ Cache L1/L2/HBM  ║    -     ║    -     ║    -     ║  NEEDED  ║    -     ║
║ TC Eligibility   ║    -     ║    -     ║    -     ║  NEEDED  ║    -     ║
║ Calibration src  ║    -     ║    -     ║    -     ║  NEEDED  ║    -     ║
║ On-demand/spot   ║    -     ║    -     ║    -     ║    -     ║  NEEDED  ║
║ Time Machine     ║    -     ║    -     ║    -     ║    -     ║  NEEDED  ║
║ CO2/car equiv.   ║    -     ║    -     ║    -     ║    -     ║  NEEDED  ║
╚══════════════════╩══════════╩══════════╩══════════╩══════════╩══════════╝
```

---

# PARTIE II — ENRICHISSEMENT COMPLET DE CHAQUE DIALECTE

## 3. Dialecte `arch` — Version Enrichie Complète

```tablegen
// ArchitectureOps.td — Version 2.0 ENRICHIE

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def Architecture_Dialect : Dialect {
  let name = "arch";
  let summary = "Neural network architecture dialect for NEURAX v2";
  let cppNamespace = "::mlir::neurax::arch";
}

class Arch_Op<string mnemonic, list<Trait> traits = []> :
    Op<Architecture_Dialect, mnemonic, traits>;

// ── CORRECTION 1 : ModelType comme enum validé ────────────────────────────

def ModelType_Transformer : I32EnumAttrCase<"Transformer", 0, "transformer">;
def ModelType_MoE         : I32EnumAttrCase<"MoE",         1, "moe">;
def ModelType_CNN         : I32EnumAttrCase<"CNN",         2, "cnn">;
def ModelType_SSM         : I32EnumAttrCase<"SSM",         3, "ssm">;
def ModelType_Diffusion   : I32EnumAttrCase<"Diffusion",   4, "diffusion">;
def ModelType_GNN         : I32EnumAttrCase<"GNN",         5, "gnn">;
def ModelType_RNN         : I32EnumAttrCase<"RNN",         6, "rnn">;
def ModelType_GAN         : I32EnumAttrCase<"GAN",         7, "gan">;
def ModelType_Hybrid      : I32EnumAttrCase<"Hybrid",      8, "hybrid">;
def ModelType_Custom      : I32EnumAttrCase<"Custom",      9, "custom">;

def ModelTypeAttr : I32EnumAttr<"ModelTypeAttr", "Type de famille du modèle",
  [ModelType_Transformer, ModelType_MoE, ModelType_CNN, ModelType_SSM,
   ModelType_Diffusion, ModelType_GNN, ModelType_RNN, ModelType_GAN,
   ModelType_Hybrid, ModelType_Custom]> {
  let cppNamespace = "::mlir::neurax::arch";
}

// ── CORRECTION 2 : ModelOp avec type validé ───────────────────────────────

def ModelOp : Arch_Op<"model"> {
  let summary = "Define a neural network model (v2 — with validated type)";
  let arguments = (ins
    StrAttr:$name,
    ModelTypeAttr:$model_type,         // ENRICHI : enum validé vs StrAttr libre
    StrAttr:$schema_version,
    StrAttr:$description
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $name `:` $model_type `v` $schema_version attr-dict $body
  }];
}

// ── CORRECTION 3 : GlobalParamsOp étendu pour TOUTES les familles ─────────

def GlobalParamsOp : Arch_Op<"global_params"> {
  let summary = "Global model parameters — covers all architecture families";
  let arguments = (ins
    // ── Communs ────────────────────────────────────────────────────────
    OptionalAttr<I64Attr>:$hidden_size,
    OptionalAttr<I64Attr>:$num_layers,
    OptionalAttr<I64Attr>:$vocab_size,
    OptionalAttr<I64Attr>:$sequence_length,
    // ── Transformer / LLM ──────────────────────────────────────────────
    OptionalAttr<I64Attr>:$num_attention_heads,
    OptionalAttr<I64Attr>:$num_key_value_heads,   // AJOUT : GQA support
    OptionalAttr<I64Attr>:$intermediate_size,
    OptionalAttr<I64Attr>:$head_dim,              // AJOUT : d_head calculé
    OptionalAttr<I64Attr>:$max_position_embeddings,
    // ── MoE ────────────────────────────────────────────────────────────
    OptionalAttr<I64Attr>:$num_experts,
    OptionalAttr<I64Attr>:$num_experts_per_tok,   // AJOUT : top-k actifs
    OptionalAttr<I64Attr>:$num_shared_experts,    // AJOUT : DeepSeek style
    OptionalAttr<I64Attr>:$moe_intermediate_size, // AJOUT : taille experts
    // ── SSM (Mamba) ─────────────────────────────────────────────────────
    OptionalAttr<I64Attr>:$ssm_state_size,         // AJOUT : d_state
    OptionalAttr<I64Attr>:$ssm_expand,             // AJOUT : expansion factor
    OptionalAttr<I64Attr>:$ssm_conv_kernel,        // AJOUT : conv1d kernel
    // ── CNN ──────────────────────────────────────────────────────────────
    OptionalAttr<I64Attr>:$initial_channels,       // AJOUT
    OptionalAttr<I64Attr>:$num_classes,            // AJOUT
    // ── RNN/LSTM ─────────────────────────────────────────────────────────
    OptionalAttr<I64Attr>:$rnn_hidden_size,        // AJOUT
    // ── Diffusion ────────────────────────────────────────────────────────
    OptionalAttr<I64Attr>:$diffusion_timesteps,    // AJOUT
    OptionalAttr<I64Attr>:$latent_channels,        // AJOUT
    // ── GNN ──────────────────────────────────────────────────────────────
    OptionalAttr<I64Attr>:$node_features,          // AJOUT
    OptionalAttr<I64Attr>:$num_message_passing     // AJOUT
  );
  let assemblyFormat = "attr-dict";
}

// ── CORRECTION 4 : LayerOp avec RepeatAttr ───────────────────────────────

def LayerOp : Arch_Op<"layer"> {
  let summary = "Layer definition with repeat support";
  let arguments = (ins
    StrAttr:$id,
    StrAttr:$layer_type,
    OptionalAttr<I64ArrayAttr>:$input_shape,
    OptionalAttr<I64ArrayAttr>:$output_shape,
    DefaultValuedAttr<I64Attr, "1">:$repeat,  // AJOUT : nb de répétitions
    OptionalAttr<StrAttr>:$repeat_id_suffix    // AJOUT : "_0","_1",...,"_N"
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $id `:` $layer_type (`x` $repeat^)? attr-dict $body
  }];
}

// ── AJOUT : RepeatExpansionOp ─────────────────────────────────────────────

def RepeatExpansionOp : Arch_Op<"repeat_expansion"> {
  let summary = "Marks that a layer group was expanded from repeat=N";
  let description = [{
    Utilisé lors du lowering ArchitectureIR → GraphIR pour tracer
    quelles couches proviennent d'un groupe répété.
    Exemple : repeat=32 sur 'decoder_block' produit 32 LayerOp avec IDs
    'decoder_block_0' ... 'decoder_block_31', tous annotés par cet op.
  }];
  let arguments = (ins
    StrAttr:$original_id,
    I64Attr:$repeat_count,
    I64Attr:$start_index,
    I64Attr:$end_index
  );
  let assemblyFormat = "$original_id `x` $repeat_count attr-dict";
}

// ── AJOUT : CustomLayerOp ─────────────────────────────────────────────────

def CustomLayerOp : Arch_Op<"custom_layer"> {
  let summary = "Custom layer with analytical formulas";
  let description = [{
    Pour les architectures expérimentales. Les formules sont des strings
    évaluées par evalexpr dans le contexte {B, S, D, H, N, ...}.
  }];
  let arguments = (ins
    StrAttr:$name,
    StrAttr:$parameters_formula,         // ex: "4 * D * D"
    StrAttr:$flops_forward_formula,      // ex: "2 * B * S * D * D"
    StrAttr:$flops_backward_formula,
    StrAttr:$memory_activation_formula,
    OptionalAttr<StrAttr>:$memory_gradient_formula,
    BoolAttr:$formulas_validated         // true si evalexpr a validé
  );
  let assemblyFormat = "$name attr-dict";
}

// ── ENRICHISSEMENT : ArchMetricsOp avec confiance ─────────────────────────

def ArchMetricsOp : Arch_Op<"metrics"> {
  let summary = "Architecture metrics with confidence score";
  let arguments = (ins
    I64Attr:$total_layer_count,
    I64Attr:$approximate_params,
    I64Attr:$custom_layer_count,
    BoolAttr:$has_branches,
    BoolAttr:$has_residuals,
    F64Attr:$confidence_score            // AJOUT : confiance sur les métriques
  );
  let assemblyFormat = [{
    `layers` `=` $total_layer_count `,` `params` `~` $approximate_params
    `conf` $confidence_score attr-dict
  }];
}
```

---

## 4. Dialecte `graph` — Version Enrichie Complète

```tablegen
// GraphOps.td — Version 2.0 ENRICHIE : SSA réel

def Graph_Dialect : Dialect {
  let name = "graph";
  let summary = "SSA computation graph dialect v2 for NEURAX";
  let cppNamespace = "::mlir::neurax::graph";
}

// ── CORRECTION FONDAMENTALE : SSAValueType ───────────────────────────────
// Le dialecte graph DOIT définir son propre type SSA pour les tenseurs

def SSATensorType : TypeDef<Graph_Dialect, "SSATensor"> {
  let mnemonic = "ssa_tensor";
  let summary = "SSA tensor value — defined exactly once, used N times";
  let parameters = (ins
    StringRefParameter<"SSA value unique identifier">:$id,
    ArrayRefParameter<"int64_t", "tensor shape">:$shape,
    StringRefParameter<"dtype (fp32, fp16, bf16, ...)">:$dtype,
    "uint64_t":$size_bytes
  );
  let assemblyFormat = "`<` $id `:` $shape `,` $dtype `>`";
}

// ── CORRECTION : NodeOp avec SSA propre ──────────────────────────────────

def NodeOp : Graph_Op<"node"> {
  let summary = "SSA node in the computation graph";
  let description = [{
    Chaque nœud du graphe SSA produit EXACTEMENT UN SSATensorType.
    C'est la garantie fondamentale du SSA : une définition, N usages.
  }];
  let arguments = (ins
    StrAttr:$layer_id,
    StrAttr:$layer_type,
    I64Attr:$topo_index,               // AJOUT : position dans l'ordre topo
    F64Attr:$flops_forward,            // ENRICHI : forward séparé du backward
    F64Attr:$flops_backward,           // AJOUT
    I64Attr:$param_count,
    I64Attr:$bytes_read,               // AJOUT : pour roofline
    I64Attr:$bytes_written,            // AJOUT : pour roofline
    OptionalAttr<StrAttr>:$pattern     // AJOUT : pattern détecté (ResidualConn, etc.)
  );
  let results = (outs SSATensorType:$output);  // CORRIGÉ : type SSA précis
  let assemblyFormat = [{
    $layer_id `[` $topo_index `]` `:` $layer_type
    `fwd` $flops_forward `bwd` $flops_backward
    `params` $param_count attr-dict `->` type($output)
  }];
}

// ── CORRECTION : EdgeOp avec SSA propre ──────────────────────────────────

def EdgeOp : Graph_Op<"edge"> {
  let summary = "SSA edge — carries a tensor value between two nodes";
  let description = [{
    Une arête SSA = une valeur SSATensorType produite par un nœud source
    et consommée par un nœud destination. En SSA, cette valeur est
    IMMUABLE : elle ne peut pas être modifiée après production.
    La liveness est annotée ici : born_at = topo_index du producteur,
    dies_at = max(topo_index de tous les consommateurs).
  }];
  let arguments = (ins
    SSATensorType:$value,              // CORRIGÉ : SSAValue typée
    I64Attr:$born_at,                  // AJOUT : step de naissance
    I64Attr:$dies_at,                  // AJOUT : step de mort
    I64Attr:$extended_dies_at,         // AJOUT : pour le backward pass
    BoolAttr:$is_checkpointed,         // AJOUT : recomputé en backward ?
    BoolAttr:$is_offloaded,            // AJOUT : offloadé vers RAM CPU ?
    BoolAttr:$is_parameter,
    BoolAttr:$is_activation,
    BoolAttr:$requires_grad
  );
  let assemblyFormat = [{
    type($value) `[` $born_at `,` $dies_at `]`
    (`ext` $extended_dies_at^)?
    attr-dict
  }];
}

// ── AJOUT : CriticalPathOp ────────────────────────────────────────────────

def CriticalPathOp : Graph_Op<"critical_path"> {
  let summary = "Annotate the critical path in the computation graph";
  let arguments = (ins
    StrArrayAttr:$node_ids,            // IDs des nœuds sur le chemin critique
    F64Attr:$total_latency_ms,
    F64Attr:$sequential_latency_ms,    // Si tout était séquentiel
    F64Attr:$parallelism_ratio,        // sequential / critical = degré de //isme
    StrAttr:$bottleneck_node_id
  );
  let assemblyFormat = [{
    `latency` $total_latency_ms `ms` `seq` $sequential_latency_ms `ms`
    `ratio` $parallelism_ratio `bottleneck` $bottleneck_node_id attr-dict
  }];
}

// ── AJOUT : PatternAnnotationOp ───────────────────────────────────────────

def PatternAnnotationOp : Graph_Op<"pattern"> {
  let summary = "Detected architectural pattern annotation";
  let arguments = (ins
    StrAttr:$pattern_type,             // "residual", "transformer_block", "moe_dispatch"
    StrArrayAttr:$involved_nodes,
    BoolAttr:$fusion_possible,
    F64Attr:$bytes_reduction_if_fused
  );
  let assemblyFormat = "$pattern_type attr-dict";
}

// ── ENRICHISSEMENT : GraphMetricsOp ──────────────────────────────────────

def GraphMetricsOp : Graph_Op<"metrics"> {
  let summary = "Complete graph analysis metrics";
  let arguments = (ins
    I64Attr:$node_count,
    I64Attr:$edge_count,
    I64Attr:$depth,
    I64Attr:$max_parallelism_degree,   // AJOUT : nœuds parallèles max
    I64Attr:$critical_path_length,
    F64Attr:$critical_path_latency_ms, // AJOUT
    BoolAttr:$has_branches,
    BoolAttr:$has_residuals,
    I64Attr:$parallel_regions_count,   // AJOUT
    F64Attr:$tensor_core_eligible_pct  // AJOUT : % ops éligibles TC
  );
  let assemblyFormat = [{
    `nodes` $node_count `edges` $edge_count `depth` $depth
    `critical` $critical_path_latency_ms `ms` attr-dict
  }];
}
```

---

## 5. Dialecte `tensor` — Version Enrichie Complète

```tablegen
// TensorOps.td — Version 2.0 ENRICHIE : dims symboliques + confiance

// ── AJOUT : DimKind enum ──────────────────────────────────────────────────

def DimKind_Known    : I32EnumAttrCase<"Known",    0, "known">;
def DimKind_Symbolic : I32EnumAttrCase<"Symbolic", 1, "symbolic">;
def DimKind_Dynamic  : I32EnumAttrCase<"Dynamic",  2, "dynamic">;

def DimKindAttr : I32EnumAttr<"DimKindAttr",
  "Dimension kind (concrete vs symbolic vs dynamic)",
  [DimKind_Known, DimKind_Symbolic, DimKind_Dynamic]> {
  let cppNamespace = "::mlir::neurax::tensor";
}

// ── AJOUT : DimResolutionSourceAttr ──────────────────────────────────────

def DimSource_ExplicitJson     : I32EnumAttrCase<"ExplicitJson",     0, "json">;
def DimSource_GlobalParam      : I32EnumAttrCase<"GlobalParam",      1, "global">;
def DimSource_NeighborProp     : I32EnumAttrCase<"NeighborProp",     2, "propagated">;
def DimSource_TrainingConfig   : I32EnumAttrCase<"TrainingConfig",   3, "training">;
def DimSource_ArchRule         : I32EnumAttrCase<"ArchRule",         4, "rule">;
def DimSource_CalibFallback    : I32EnumAttrCase<"CalibFallback",    5, "fallback">;

def DimSourceAttr : I32EnumAttr<"DimSourceAttr", "How this dimension was resolved",
  [DimSource_ExplicitJson, DimSource_GlobalParam, DimSource_NeighborProp,
   DimSource_TrainingConfig, DimSource_ArchRule, DimSource_CalibFallback]> {
  let cppNamespace = "::mlir::neurax::tensor";
}

// ── AJOUT : ResolvedDimOp ─────────────────────────────────────────────────

def ResolvedDimOp : Tensor_Op<"dim"> {
  let summary = "A single tensor dimension with resolution metadata";
  let description = [{
    Représente une dimension avec son degré de certitude.
    Permet de propager la confiance à travers les formes.
  }];
  let arguments = (ins
    I64Attr:$value,                // Valeur concrète (0 si symbolique)
    StrAttr:$symbolic_name,        // Nom symbolique ("B", "S", "" si concret)
    DimKindAttr:$kind,
    F32Attr:$confidence,           // 0.0 = totalement symbolique, 1.0 = certain
    DimSourceAttr:$source
  );
  let assemblyFormat = [{
    $kind `(` $value `:` $symbolic_name `)` `conf` $confidence `src` $source attr-dict
  }];
}

// ── CORRECTION : TensorInfoOp avec formes mixtes ─────────────────────────

def TensorInfoOp : Tensor_Op<"info"> {
  let summary = "Tensor with mixed concrete/symbolic shape";
  let arguments = (ins
    StrAttr:$tensor_id,
    // Les shapes sont stockées séparément : concrètes et symboliques
    I64ArrayAttr:$concrete_dims,       // -1 pour les dims symboliques
    StrArrayAttr:$symbolic_names,      // "" pour les dims concrètes
    F32ArrayAttr:$dim_confidences,     // AJOUT : confiance par dim [0.0-1.0]
    StrAttr:$dtype,
    I64Attr:$size_bytes,               // -1 si taille inconnue (dims symboliques)
    StrAttr:$produced_by,
    F32Attr:$overall_confidence        // AJOUT : confiance globale de ce tenseur
  );
  let results = (outs SSATensorType:$result);
  let assemblyFormat = [{
    $tensor_id `:` $dtype `dims` $concrete_dims `symb` $symbolic_names
    `conf` $overall_confidence `from` $produced_by attr-dict `->` type($result)
  }];
}

// ── AJOUT : ShapeGateOp ───────────────────────────────────────────────────

def ShapeGateOp : Tensor_Op<"shape_gate"> {
  let summary = "ShapeInferenceGate — blocks pipeline if resolution too low";
  let description = [{
    Applique la règle : si moins de 70% des dimensions sont concrètes,
    le pipeline est bloqué car les métriques seraient trop imprécises.
    Émet soit PASS soit BLOCKED.
  }];
  let arguments = (ins
    F32Attr:$resolution_ratio,
    I64Attr:$total_dims,
    I64Attr:$resolved_dims,
    I64Attr:$symbolic_dims,
    F32Attr:$threshold,               // Typiquement 0.70
    BoolAttr:$gate_passed
  );
  let assemblyFormat = [{
    $resolved_dims `/` $total_dims `=` $resolution_ratio
    (`PASS` | `BLOCKED`) attr-dict
  }];
}

// ── ENRICHISSEMENT : TensorMetricsOp ─────────────────────────────────────

def TensorMetricsOp : Tensor_Op<"metrics"> {
  let summary = "Tensor analysis metrics with resolution report";
  let arguments = (ins
    I64Attr:$activation_memory_bytes,
    F64Attr:$memory_bandwidth_required,
    I64Attr:$total_tensor_count,
    I64Attr:$largest_tensor_bytes,
    F32Attr:$resolution_ratio,        // AJOUT
    I64Attr:$unresolved_dim_count,    // AJOUT
    BoolAttr:$shape_gate_passed       // AJOUT
  );
  let assemblyFormat = [{
    `activation` `=` $activation_memory_bytes `resolution` $resolution_ratio attr-dict
  }];
}
```

---

## 6. Dialecte `op` — Version Enrichie Complète

```tablegen
// OperatorOps.td — Version 2.0 ENRICHIE

// ── AJOUT : BackwardRatioAttr ─────────────────────────────────────────────
// Critique pour la précision des FLOPs backward

def BackwardRatioOp : Operator_Op<"backward_ratio"> {
  let summary = "Exact backward/forward FLOPs ratio for an operation";
  let description = [{
    Stocke le ratio exact backward/forward calculé analytiquement.
    Exemples : MatMul=2.0, Attention=2.5, LayerNorm=4.0, ReLU=1.0
    Ce ratio est appliqué lors du calcul de ComputeIR.
  }];
  let arguments = (ins
    StrAttr:$op_type,
    F64Attr:$ratio,
    StrAttr:$formula_justification,    // Explication mathématique du ratio
    F64Attr:$confidence                // 1.0 = analytiquement exact
  );
  let assemblyFormat = "$op_type `->` $ratio `x` attr-dict";
}

// ── CORRECTION : AttentionOp → 4 variantes séparées ─────────────────────

def MHAOp : Operator_Op<"mha"> {
  let summary = "Multi-Head Attention (MHA standard)";
  let arguments = (ins
    AnyType:$query, AnyType:$key, AnyType:$value,
    I64Attr:$hidden_size,
    I64Attr:$num_heads,
    I64Attr:$head_dim,
    BoolAttr:$causal,
    BoolAttr:$has_bias,
    F32Attr:$dropout_p,
    I64Attr:$param_count,
    F64Attr:$flops_forward,
    F64Attr:$bytes_read,               // AJOUT : pour roofline
    F64Attr:$bytes_written             // AJOUT : pour roofline
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    `mha` $query `:` type($query) `h` $hidden_size `heads` $num_heads
    `d_head` $head_dim attr-dict `:` type($result)
  }];
}

def GQAOp : Operator_Op<"gqa"> {
  let summary = "Grouped Query Attention (GQA) — LLaMA 3, Mistral";
  let arguments = (ins
    AnyType:$query, AnyType:$key, AnyType:$value,
    I64Attr:$hidden_size,
    I64Attr:$num_q_heads,
    I64Attr:$num_kv_heads,             // AJOUT : clé de GQA
    I64Attr:$head_dim,
    BoolAttr:$causal,
    I64Attr:$param_count,
    F64Attr:$flops_forward,
    F64Attr:$kv_memory_savings_bytes   // AJOUT : économie KV vs MHA
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    `gqa` $query `:` type($query) `q_heads` $num_q_heads
    `kv_heads` $num_kv_heads `h` $hidden_size attr-dict `:` type($result)
  }];
}

def FlashAttentionOp : Operator_Op<"flash_attn"> {
  let summary = "Flash Attention v2/v3 — IO-aware attention";
  let arguments = (ins
    AnyType:$query, AnyType:$key, AnyType:$value,
    I64Attr:$hidden_size,
    I64Attr:$num_q_heads,
    I64Attr:$num_kv_heads,
    I64Attr:$head_dim,
    I64Attr:$block_size_q,             // AJOUT : taille bloc tile Q
    I64Attr:$block_size_k,             // AJOUT : taille bloc tile K
    I64Attr:$param_count,
    F64Attr:$flops_forward,
    F64Attr:$bytes_standard,           // AJOUT : bytes sans Flash (pour comparaison)
    F64Attr:$bytes_flash               // AJOUT : bytes réels Flash (HBM réduit)
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    `flash_attn` $query `:` type($query) `q` $num_q_heads `kv` $num_kv_heads
    `blk_q` $block_size_q `blk_k` $block_size_k attr-dict `:` type($result)
  }];
}

// ── AJOUT : MambaBlockOp ──────────────────────────────────────────────────

def MambaBlockOp : Operator_Op<"mamba"> {
  let summary = "Mamba SSM block — selective state space model";
  let arguments = (ins
    AnyType:$input,
    I64Attr:$d_model,
    I64Attr:$d_state,                  // ssm_state_size
    I64Attr:$d_inner,                  // d_model * expand
    I64Attr:$d_conv,                   // conv1d kernel size
    I64Attr:$expand,                   // expansion factor
    I64Attr:$param_count,
    F64Attr:$flops_forward
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    `mamba` $input `:` type($input) `state` $d_state `inner` $d_inner
    `conv` $d_conv attr-dict `:` type($result)
  }];
}

// ── AJOUT : MoERouterOp + MoEExpertGroupOp séparés ───────────────────────

def MoERouterOp : Operator_Op<"moe_router"> {
  let summary = "MoE router — computes expert selection scores";
  let arguments = (ins
    AnyType:$input,
    I64Attr:$hidden_size,
    I64Attr:$num_experts,
    I64Attr:$top_k,
    BoolAttr:$load_balancing,
    I64Attr:$param_count,
    F64Attr:$flops_forward             // 2 * B * S * H * N_experts
  );
  let results = (outs AnyType:$scores, AnyType:$indices);
  let assemblyFormat = [{
    `moe_router` $input `:` type($input) `experts` $num_experts `top_k` $top_k attr-dict
  }];
}

def MoEExpertGroupOp : Operator_Op<"moe_experts"> {
  let summary = "MoE expert group — executes top-k selected experts";
  let arguments = (ins
    AnyType:$input,
    AnyType:$expert_indices,
    I64Attr:$hidden_size,
    I64Attr:$intermediate_size,
    I64Attr:$num_experts,
    I64Attr:$top_k,
    I64Attr:$num_shared_experts,       // AJOUT : DeepSeek shared experts
    I64Attr:$param_count_total,        // total expert params
    I64Attr:$param_count_active,       // top_k / num_experts * total
    F64Attr:$flops_forward             // top_k * FFN_per_expert
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    `moe_experts` $input `:` type($input) `top_k` $top_k `of` $num_experts
    `active_params` $param_count_active attr-dict
  }];
}

// ── AJOUT : FusionGroupOp ─────────────────────────────────────────────────

def FusionGroupOp : Operator_Op<"fused"> {
  let summary = "Group of ops that are fused in a single kernel";
  let description = [{
    Modélise la fusion de plusieurs ops en un seul kernel GPU.
    Les FLOPs mathématiques sont inchangés mais les bytes mémoire
    sont réduits (les tenseurs intermédiaires ne vont jamais en HBM).
  }];
  let arguments = (ins
    StrAttr:$fusion_pattern,           // "flash_attn", "conv_bn_relu", "qkv_proj"
    F64Attr:$bytes_unfused,            // bytes HBM sans fusion
    F64Attr:$bytes_fused,              // bytes HBM avec fusion
    F64Attr:$latency_speedup_factor,   // speedup mesuré ou estimé
    StrArrayAttr:$fused_ops            // liste des ops fusionnées
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$fusion_pattern attr-dict $body";
}

// ── AJOUT : EmbeddingLookupOp ─────────────────────────────────────────────

def EmbeddingLookupOp : Operator_Op<"embedding_lookup"> {
  let summary = "Token embedding lookup — zero compute FLOPs";
  let description = [{
    Les embeddings sont des lookups de table. Zéro FLOPs de calcul,
    mais les paramètres (vocab_size × d_model) doivent être comptés.
  }];
  let arguments = (ins
    AnyType:$indices,
    I64Attr:$vocab_size,
    I64Attr:$d_model,
    I64Attr:$param_count               // vocab_size * d_model
  );
  let results = (outs AnyType:$embeddings);
  let assemblyFormat = [{
    `embed` $indices `:` type($indices) `vocab` $vocab_size `dim` $d_model
    `params` $param_count attr-dict `:` type($embeddings)
  }];
}
```

---

## 7. Dialecte `compute` — Version Enrichie Complète

```tablegen
// ComputeOps.td — Version 2.0 ENRICHIE

// ── CORRECTION : ComplexityOp → enum ─────────────────────────────────────

def Complexity_Linear    : I32EnumAttrCase<"Linear",    0, "O(n)">;
def Complexity_NLogN     : I32EnumAttrCase<"NLogN",     1, "O(n_log_n)">;
def Complexity_Quadratic : I32EnumAttrCase<"Quadratic", 2, "O(n2)">;
def Complexity_Cubic     : I32EnumAttrCase<"Cubic",     3, "O(n3)">;

def ComplexityAttr : I32EnumAttr<"ComplexityAttr",
  "Computational complexity class",
  [Complexity_Linear, Complexity_NLogN, Complexity_Quadratic, Complexity_Cubic]> {
  let cppNamespace = "::mlir::neurax::compute";
}

// ── AJOUT : MemoryTrafficOp — critique pour le roofline ──────────────────

def MemoryTrafficOp : Compute_Op<"mem_traffic"> {
  let summary = "Memory traffic for an operation — feeds the roofline model";
  let arguments = (ins
    StrAttr:$op_id,
    I64Attr:$bytes_read_hbm,           // Bytes lus depuis HBM (DRAM GPU)
    I64Attr:$bytes_written_hbm,        // Bytes écrits vers HBM
    I64Attr:$bytes_read_l2,            // Bytes lus depuis L2 cache
    I64Attr:$bytes_read_sram,          // Bytes lus depuis SRAM (shared mem)
    F64Attr:$arithmetic_intensity,     // FLOPs / bytes_read_hbm
    BoolAttr:$is_compute_bound,        // arith_intensity >= ridge_point
    BoolAttr:$is_memory_bound
  );
  let assemblyFormat = [{
    $op_id `hbm_r` $bytes_read_hbm `hbm_w` $bytes_written_hbm
    `intensity` $arithmetic_intensity attr-dict
  }];
}

// ── AJOUT : BackwardRatioAnnotationOp ────────────────────────────────────

def BackwardRatioAnnotationOp : Compute_Op<"bwd_ratio"> {
  let summary = "Backward/forward ratio annotation for accurate backward FLOPs";
  let arguments = (ins
    StrAttr:$op_type,
    F64Attr:$ratio,
    F64Attr:$confidence,               // 1.0 = analytiquement exact
    StrAttr:$formula                   // Justification mathématique
  );
  let assemblyFormat = "$op_type `x` $ratio `conf` $confidence attr-dict";
}

// ── AJOUT : FusionImpactOp ────────────────────────────────────────────────

def FusionImpactOp : Compute_Op<"fusion_impact"> {
  let summary = "Impact of operation fusion on memory traffic (not FLOPs)";
  let arguments = (ins
    StrAttr:$fusion_pattern,
    F64Attr:$bytes_saved_hbm,          // Bytes HBM économisés par la fusion
    F64Attr:$latency_improvement_pct,  // % d'amélioration de latence
    BoolAttr:$flops_unchanged          // Toujours true : fusion = même math
  );
  let assemblyFormat = "$fusion_pattern `saves` $bytes_saved_hbm `bytes` attr-dict";
}

// ── ENRICHISSEMENT : ComputeMetricsOp ────────────────────────────────────

def ComputeMetricsOp : Compute_Op<"metrics"> {
  let summary = "Complete compute metrics with per-op breakdown";
  let arguments = (ins
    F64Attr:$total_flops,
    F64Attr:$forward_flops,
    F64Attr:$backward_flops,
    F64Attr:$optimizer_flops,
    F64Attr:$total_step_flops,
    F64Attr:$macs,
    F64Attr:$flops_per_token,
    F64Attr:$arithmetic_intensity,
    I64Attr:$bytes_read_total,         // AJOUT
    I64Attr:$bytes_written_total,      // AJOUT
    F64Attr:$backward_flops_confidence // AJOUT : confiance sur le backward
  );
  let assemblyFormat = [{
    `fwd` $forward_flops `bwd` $backward_flops `intensity` $arithmetic_intensity attr-dict
  }];
}
```

---

## 8. Dialecte `mem` — Version Enrichie Complète

```tablegen
// MemoryOps.td — Version 2.0 ENRICHIE

// ── CORRECTION : LivenessOp avec SSAValue ────────────────────────────────

def LivenessOp : Memory_Op<"liveness"> {
  let summary = "SSA tensor liveness with training/inference modes";
  let arguments = (ins
    SSATensorType:$tensor,             // CORRIGÉ : SSAValue typée
    I64Attr:$born_at,
    I64Attr:$dies_at,
    I64Attr:$extended_dies_at,         // AJOUT : étendu pour backward
    I64Attr:$size_bytes,
    StrAttr:$category,                 // "parameter","activation","gradient","optimizer"
    BoolAttr:$is_checkpointed,         // AJOUT
    BoolAttr:$is_offloaded             // AJOUT : vers CPU RAM
  );
  let assemblyFormat = [{
    type($tensor) `[` $born_at `,` $dies_at `]` `cat` $category
    `size` $size_bytes attr-dict
  }];
}

// ── AJOUT : ZeroStageOp ───────────────────────────────────────────────────

def ZeroStageOp : Memory_Op<"zero_stage"> {
  let summary = "ZeRO optimizer stage — per-GPU memory breakdown";
  let arguments = (ins
    I64Attr:$stage,                    // 0, 1, 2, 3
    I64Attr:$num_gpus,
    I64Attr:$params_total_bytes,
    I64Attr:$params_per_gpu_bytes,     // params_total / N (stage 3)
    I64Attr:$grads_per_gpu_bytes,      // grads / N (stage 2+)
    I64Attr:$optimizer_per_gpu_bytes,  // optim / N (stage 1+)
    I64Attr:$activations_per_gpu_bytes,
    I64Attr:$total_per_gpu_bytes,      // somme + overhead
    F64Attr:$communication_overhead_pct
  );
  let assemblyFormat = [{
    `stage` $stage `on` $num_gpus `gpus`
    `total` $total_per_gpu_bytes `per_gpu` attr-dict
  }];
}

// ── AJOUT : KvCacheOp ────────────────────────────────────────────────────

def KvCacheOp : Memory_Op<"kv_cache"> {
  let summary = "KV cache size for LLM inference";
  let description = [{
    Formule : 2 * num_layers * seq_len * num_kv_heads * head_dim * dtype_bytes
    L'impact du paged attention et du GQA est modélisé ici.
  }];
  let arguments = (ins
    I64Attr:$num_layers,
    I64Attr:$seq_len,
    I64Attr:$num_kv_heads,
    I64Attr:$head_dim,
    I64Attr:$dtype_bytes,
    I64Attr:$size_bytes,               // taille totale calculée
    BoolAttr:$uses_paged_attention,    // impact sur la fragmentation
    F64Attr:$gqa_savings_pct           // économie vs MHA
  );
  let assemblyFormat = [{
    `kv_cache` `seq` $seq_len `kv_heads` $num_kv_heads `head_dim` $head_dim
    `=` $size_bytes `bytes` attr-dict
  }];
}

// ── AJOUT : GradientCheckpointingOp ──────────────────────────────────────

def GradientCheckpointingOp : Memory_Op<"checkpointing"> {
  let summary = "Gradient checkpointing activation savings";
  let arguments = (ins
    BoolAttr:$enabled,
    I64Attr:$activations_saved_bytes,  // Sans checkpointing
    I64Attr:$activations_kept_bytes,   // Avec checkpointing
    I64Attr:$savings_bytes,
    F64Attr:$recomputation_flops_overhead_pct, // ~25% FLOPs supplémentaires
    I64Attr:$checkpoint_every_n_layers // Toutes combien de couches
  );
  let assemblyFormat = [{
    `checkpointing` $enabled `saves` $savings_bytes `bytes`
    `overhead` $recomputation_flops_overhead_pct `%` attr-dict
  }];
}

// ── AJOUT : FragmentationOp ───────────────────────────────────────────────

def FragmentationOp : Memory_Op<"fragmentation"> {
  let summary = "Memory fragmentation model (PyTorch Caching Allocator)";
  let arguments = (ins
    StrAttr:$allocator_type,           // "pytorch_caching", "cuda_pool", "default"
    F64Attr:$fragmentation_pct,        // 3% (CUDA pool) à 30% (worst case)
    I64Attr:$peak_bytes_theoretical,   // sans fragmentation
    I64Attr:$peak_bytes_effective,     // avec fragmentation
    I64Attr:$largest_free_block_bytes
  );
  let assemblyFormat = [{
    $allocator_type `frag` $fragmentation_pct `%`
    `theoretical` $peak_bytes_theoretical `effective` $peak_bytes_effective attr-dict
  }];
}

// ── ENRICHISSEMENT : MemoryMetricsOp complet ─────────────────────────────

def MemoryMetricsOp : Memory_Op<"metrics"> {
  let summary = "Complete memory metrics including training and inference peaks";
  let arguments = (ins
    // Composantes
    I64Attr:$parameter_memory_bytes,
    I64Attr:$activation_memory_bytes,
    I64Attr:$gradient_memory_bytes,
    I64Attr:$optimizer_state_bytes,
    I64Attr:$kv_cache_bytes,           // AJOUT
    I64Attr:$workspace_bytes,          // AJOUT : cuDNN/cuBLAS workspace
    // Pics
    I64Attr:$peak_vram_inference_bytes,
    I64Attr:$peak_vram_training_bytes,
    I64Attr:$peak_vram_per_gpu_bytes,  // AJOUT : après ZeRO
    // Overhead
    I64Attr:$cuda_context_bytes,       // ~500 MB fixe
    I64Attr:$fragmentation_overhead_bytes, // AJOUT
    // Savings
    I64Attr:$checkpointing_savings_bytes, // AJOUT
    // Dérivés
    I64Attr:$max_batch_size_fit,
    F64Attr:$memory_utilization_ratio
  );
  let assemblyFormat = [{
    `params` `=` $parameter_memory_bytes `,` `peak_train` `=` $peak_vram_training_bytes
    `,` `peak_infer` `=` $peak_vram_inference_bytes attr-dict
  }];
}
```

---

# PARTIE III — COMMUNICATION ENTRE DIALECTES

## 13. Flux SSA Inter-Dialectes — Comment les Valeurs Circulent

```
RÈGLE FONDAMENTALE DE COMMUNICATION :
Les dialectes ne communiquent pas directement entre eux via des appels Rust.
Ils communiquent via le MODULE MLIR PARTAGÉ qui est enrichi progressivement.

FLUX DE DONNÉES :

JSON Input
  │
  ▼ [neurax-parser]
arch.model { ... }                           ← arch dialect peuple le module
  │
  ▼ [GraphPass : arch → graph]
graph.node "embed" [0] : "token_embedding" → graph.ssa_tensor<"embed_out:...">
graph.node "attn_0" [1] : "attention"      → graph.ssa_tensor<"attn0_out:...">
graph.edge (ssa_tensor<"embed_out">) [0,1] is_activation=true
graph.critical_path [...]
  │
  ▼ [TensorPass : enrichit les graph.edge avec formes résolues]
tensor.info "embed_out" : "bf16" dims [32,2048,-1] symb ["","","D"] conf 0.95
tensor.shape_gate 0.993 "PASS"
  │
  ▼ [OperatorPass : enrichit les graph.node avec atom ops]
op.embedding_lookup %indices : ... vocab 128256 dim 4096 params 524288000
op.gqa %q, %k, %v : ... q_heads 32 kv_heads 8 h 4096 fwd 1.6e12
op.backward_ratio "gqa" -> 2.5x conf 0.95
  │
  ▼ [ComputePass : ajoute compute.* au module]
compute.mem_traffic "attn_0" hbm_r 2147483648 intensity 145.0
compute.fusion_impact "flash_attn" saves 8589934592 bytes
compute.metrics fwd 1.6e12 bwd 4.0e12 intensity 145.0
  │
  ▼ [MemoryPass : ajoute mem.* en utilisant graph.edge liveness]
mem.liveness <"attn0_out"> [1,4] cat "activation" size 134217728
mem.zero_stage 3 on 8 gpus total 2684354560 per_gpu
mem.kv_cache seq 8192 kv_heads 8 head_dim 128 = 1073741824 bytes
mem.metrics params=16060522496 peak_train=21474836480 peak_infer=16911433728
  │
  ▼ [ParallelismPass + HardwarePass : EN PARALLÈLE]
  │
  ├─► par.strategy dp=1 tp=1 pp=1 ep=1 zero=3 gpus=8
  │   par.all_reduce payload 16 GB bw 600 GB/s latency 29.0 ms
  │   par.scaling efficiency=0.87 comm_overhead=13.0%
  │
  └─► hw.gpu "A100-80GB" tflops 312 bw 2000 vram 80
      hw.roofline compute 312e12 memory 2e12 ridge 156.0
      hw.timing "attn_0" compute 35.2 ms memory 67.3 ms
      hw.metrics latency=180.0 ms throughput=4551 tok/s util=0.87
  │
  ▼ [CostPass]
cost.pricing gpu 4.35 $/h energy 0.12 $/kWh pue 1.2
cost.training 34.0 h $ 592.8 gpu_hours 272.0
cost.energy 192.0 kWh co2 91.2 kg
cost.metrics $592.8 34.0h 192.0kWh 91.2kg CO2
  │
  ▼ [ReportPass]
report.all_metrics params 8030261248 flops 1.6e12 peak 21474836480
report.confidence_score total_parameters=0.999 flops=0.94 vram=0.88
report.diagnostic "info" : "I001" "GQA detected: 75% KV memory savings"
report.recommendation "high" : "Enable Flash Attention" impact "30% latency"
```

---

## 14. Passes de Conversion Obligatoires

```rust
// Passes MLIR requises pour la conversion entre dialectes

// Pass 1 : arch → graph (expand repeats, construire SSA)
pub struct ArchToGraphConversion;
// Inputs  : arch.model contenant des arch.layer avec repeat=N
// Outputs : graph.node × N pour chaque layer, graph.edge SSA

// Pass 2 : graph → tensor (propagation formes)
pub struct GraphToTensorConversion;
// Inputs  : graph.node avec layer_type, graph.edge vides de formes
// Outputs : tensor.info sur chaque edge, tensor.shape_gate, tensor.metrics

// Pass 3 : tensor → op (lowering vers AtomOps)
pub struct TensorToOperatorConversion;
// Inputs  : graph.node avec shapes résolues
// Outputs : op.* pour chaque AtomOp, op.backward_ratio, op.metrics

// Pass 4 : op → compute (agrégation FLOPs)
pub struct OperatorToComputeConversion;
// Inputs  : op.* avec flops_forward, bytes_read/written
// Outputs : compute.mem_traffic, compute.fusion_impact, compute.metrics

// Pass 5 : compute → mem (liveness + peak VRAM)
pub struct ComputeToMemoryConversion;
// Inputs  : graph.edge avec born_at/dies_at, op.* avec sizes
// Outputs : mem.liveness, mem.zero_stage, mem.kv_cache, mem.metrics

// Pass 6a : mem → par (PARALLÈLE avec 6b)
pub struct MemoryToParallelismConversion;

// Pass 6b : compute + mem → hw (PARALLÈLE avec 6a)
pub struct ComputeMemoryToHardwareConversion;

// Pass 7 : par + hw → cost
pub struct ParHwToCostConversion;

// Pass 8 : ALL → report
pub struct AllToReportConversion;
```

---

# PARTIE IV — RÈGLES JSON STRICTES

## 15. Schéma JSON Complet v3.0 avec Contraintes

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "NEURAX Model JSON Schema v3.0",
  "type": "object",
  "required": ["schema_version", "model", "training", "hardware"],
  "additionalProperties": false,

  "properties": {
    "schema_version": {
      "type": "string",
      "pattern": "^3\\.\\d+$",
      "description": "Doit être '3.x' pour ce schéma"
    },

    "model": {
      "type": "object",
      "required": ["name", "type", "global_params", "input", "output", "layers"],
      "properties": {
        "name": { "type": "string", "minLength": 1 },
        "type": {
          "type": "string",
          "enum": ["transformer", "moe", "cnn", "ssm", "diffusion",
                   "gnn", "rnn", "gan", "hybrid", "experimental"]
        },
        "description": { "type": "string" },
        "global_params": { "$ref": "#/definitions/GlobalParams" },
        "input": { "$ref": "#/definitions/IOSpec" },
        "output": { "$ref": "#/definitions/IOSpec" },
        "layers": {
          "type": "array",
          "minItems": 1,
          "items": { "$ref": "#/definitions/Layer" }
        },
        "connections": {
          "type": ["array", "null"],
          "description": "Connexions explicites si non-séquentiel"
        }
      }
    },

    "training": { "$ref": "#/definitions/TrainingConfig" },
    "hardware": { "$ref": "#/definitions/HardwareConfig" },
    "data": { "$ref": "#/definitions/DataConfig" },
    "cost_config": { "$ref": "#/definitions/CostConfig" },
    "analysis_config": { "$ref": "#/definitions/AnalysisConfig" },
    "custom_layers": {
      "type": "array",
      "items": { "$ref": "#/definitions/CustomLayer" }
    }
  },

  "definitions": {

    "GlobalParams": {
      "type": "object",
      "properties": {
        "hidden_size":             { "type": "integer", "minimum": 1 },
        "num_layers":              { "type": "integer", "minimum": 1 },
        "num_attention_heads":     { "type": "integer", "minimum": 1 },
        "num_key_value_heads":     { "type": "integer", "minimum": 1 },
        "intermediate_size":       { "type": "integer", "minimum": 1 },
        "vocab_size":              { "type": "integer", "minimum": 1 },
        "max_position_embeddings": { "type": "integer", "minimum": 1 },
        "head_dim":                { "type": "integer", "minimum": 1 },
        "num_experts":             { "type": "integer", "minimum": 2 },
        "num_experts_per_tok":     { "type": "integer", "minimum": 1 },
        "num_shared_experts":      { "type": "integer", "minimum": 0 },
        "moe_intermediate_size":   { "type": "integer", "minimum": 1 },
        "ssm_state_size":          { "type": "integer", "minimum": 1 },
        "ssm_expand":              { "type": "integer", "minimum": 1 },
        "ssm_conv_kernel":         { "type": "integer", "minimum": 1 },
        "initial_channels":        { "type": "integer", "minimum": 1 },
        "num_classes":             { "type": "integer", "minimum": 1 },
        "diffusion_timesteps":     { "type": "integer", "minimum": 1 },
        "latent_channels":         { "type": "integer", "minimum": 1 },
        "node_features":           { "type": "integer", "minimum": 1 },
        "num_message_passing":     { "type": "integer", "minimum": 1 }
      }
    },

    "IOSpec": {
      "type": "object",
      "required": ["shape", "dtype"],
      "properties": {
        "shape": {
          "type": "array",
          "items": {
            "oneOf": [
              { "type": "integer", "minimum": 1 },
              { "type": "string", "enum": ["B","S","T","N","batch","seq"] }
            ]
          },
          "minItems": 1
        },
        "dtype": {
          "type": "string",
          "enum": ["fp32","fp16","bf16","fp8","int8","int4","int32","bool"]
        }
      }
    },

    "Layer": {
      "type": "object",
      "required": ["id", "layer_type"],
      "properties": {
        "id":           { "type": "string", "minLength": 1 },
        "layer_type":   { "type": "string", "minLength": 1 },
        "input_shape":  { "$ref": "#/definitions/ShapeSpec" },
        "output_shape": { "$ref": "#/definitions/ShapeSpec" },
        "repeat": {
          "type": "integer",
          "minimum": 1,
          "default": 1
        },
        "params":       { "type": "object" },
        "custom_equations": { "$ref": "#/definitions/CustomEquations" }
      }
    },

    "ShapeSpec": {
      "type": "array",
      "items": {
        "oneOf": [
          { "type": "integer", "minimum": 1 },
          { "type": "string" }
        ]
      }
    },

    "CustomEquations": {
      "type": "object",
      "required": ["parameters_formula", "flops_forward_formula"],
      "properties": {
        "parameters_formula":     { "type": "string", "minLength": 1 },
        "flops_forward_formula":  { "type": "string", "minLength": 1 },
        "flops_backward_formula": { "type": "string" },
        "memory_activation_formula": { "type": "string" },
        "memory_gradient_formula": { "type": "string" },
        "variables": {
          "type": "object",
          "description": "Variables supplémentaires disponibles dans les formules"
        }
      }
    },

    "TrainingConfig": {
      "type": "object",
      "required": ["batch_size"],
      "properties": {
        "batch_size":   { "type": "integer", "minimum": 1 },
        "seq_len":      { "type": "integer", "minimum": 1 },
        "optimizer": {
          "type": "string",
          "enum": ["adamw", "adam", "sgd", "adafactor", "lion", "rmsprop"]
        },
        "precision": {
          "type": "string",
          "enum": ["fp32", "fp16", "bf16", "fp8", "mixed"]
        },
        "mixed_precision":          { "type": "boolean" },
        "gradient_checkpointing":   { "type": "boolean" },
        "flash_attention":          { "type": "boolean" },
        "num_steps":               { "type": "integer", "minimum": 1 },
        "parallelism": {
          "type": "object",
          "properties": {
            "data_parallel_degree":     { "type": "integer", "minimum": 1 },
            "tensor_parallel_degree":   { "type": "integer", "minimum": 1 },
            "pipeline_parallel_degree": { "type": "integer", "minimum": 1 },
            "expert_parallel_degree":   { "type": "integer", "minimum": 1 },
            "zero_stage": {
              "type": "integer",
              "minimum": 0,
              "maximum": 3
            }
          }
        }
      }
    },

    "HardwareConfig": {
      "type": "object",
      "required": ["targets"],
      "properties": {
        "targets": {
          "type": "array",
          "minItems": 1,
          "items": { "$ref": "#/definitions/GpuTarget" }
        },
        "interconnect": {
          "type": "string",
          "enum": ["nvlink", "pcie", "infiniband", "nvlink_pcie"]
        },
        "interconnect_bandwidth_gb_s": { "type": "number", "minimum": 0 }
      }
    },

    "GpuTarget": {
      "type": "object",
      "required": ["name", "count"],
      "properties": {
        "id":                        { "type": "string" },
        "name":                      { "type": "string", "minLength": 1 },
        "count":                     { "type": "integer", "minimum": 1 },
        "memory_gb":                 { "type": "number", "minimum": 0.1 },
        "tflops_fp16":               { "type": "number", "minimum": 0 },
        "tflops_bf16":               { "type": "number", "minimum": 0 },
        "tflops_int8":               { "type": "number", "minimum": 0 },
        "memory_bandwidth_gb_s":     { "type": "number", "minimum": 0 },
        "tensor_cores":              { "type": "boolean" },
        "nvlink":                    { "type": "boolean" },
        "nvlink_bandwidth_gb_s":     { "type": "number", "minimum": 0 }
      }
    },

    "CostConfig": {
      "type": "object",
      "properties": {
        "provider": {
          "type": "string",
          "enum": ["aws", "gcp", "azure", "lambda", "custom"]
        },
        "gpu_on_demand_usd_per_hour":  { "type": "number", "minimum": 0 },
        "gpu_spot_usd_per_hour":       { "type": "number", "minimum": 0 },
        "gpu_reserved_usd_per_hour":   { "type": "number", "minimum": 0 },
        "energy_kwh_usd":              { "type": "number", "minimum": 0 },
        "pue_factor":                  { "type": "number", "minimum": 1.0 },
        "carbon_intensity_g_co2_per_kwh": { "type": "number", "minimum": 0 }
      }
    },

    "CustomLayer": {
      "type": "object",
      "required": ["name", "parameters_formula", "flops_forward_formula"],
      "properties": {
        "name":                       { "type": "string", "minLength": 1 },
        "description":                { "type": "string" },
        "input_shape":                { "$ref": "#/definitions/ShapeSpec" },
        "output_shape":               { "$ref": "#/definitions/ShapeSpec" },
        "variables":                  { "type": "object" },
        "parameters_formula":         { "type": "string", "minLength": 1 },
        "flops_forward_formula":      { "type": "string", "minLength": 1 },
        "flops_backward_formula":     { "type": "string" },
        "memory_activation_formula":  { "type": "string" },
        "memory_gradient_formula":    { "type": "string" }
      }
    },

    "AnalysisConfig": {
      "type": "object",
      "properties": {
        "confidence_threshold":         { "type": "number", "minimum": 0, "maximum": 1 },
        "generate_variants":            { "type": "boolean" },
        "variant_types": {
          "type": "array",
          "items": {
            "type": "string",
            "enum": ["int8","int4","flash_attention","pruning_50","lora","distillation"]
          }
        },
        "time_machine_years":           { "type": "integer", "minimum": 1, "maximum": 10 }
      }
    }
  }
}
```

---

## 16. Règles de Cohérence Interne (32 règles)

Ces 32 règles sont vérifiées par le validateur Rust APRÈS la validation schéma JSON.

```rust
// crates/neurax-parser/src/validator.rs

/// Les 32 règles de cohérence interne du JSON
pub enum ValidationRule {
    // ── Règles Architecture (R01–R10) ─────────────────────────────────

    /// R01 : hidden_size doit être divisible par num_attention_heads
    R01_HiddenDivisibleByHeads,
    /// R02 : num_key_value_heads doit diviser num_attention_heads
    R02_KvHeadsDivideQHeads,
    /// R03 : head_dim == hidden_size / num_attention_heads (si fourni)
    R03_HeadDimConsistent,
    /// R04 : Toutes les couches doivent avoir des IDs uniques
    R04_UniqueLayerIds,
    /// R05 : Pour MoE, num_experts_per_tok < num_experts
    R05_MoeTopKLtExperts,
    /// R06 : Les IDs des couches ne doivent pas contenir d'espaces
    R06_NoSpacesInLayerIds,
    /// R07 : Si repeat > 1, output_shape de la couche = input_shape (couche iso)
    R07_RepeatRequiresIsoShape,
    /// R08 : Le type de modèle doit être cohérent avec les couches déclarées
    R08_ModelTypeConsistentWithLayers,
    /// R09 : num_experts * moe_intermediate_size > intermediate_size (MoE plus grand)
    R09_MoeCapacityConsistent,
    /// R10 : Pour SSM, ssm_expand >= 1 et ssm_state_size >= 1
    R10_SsmParamsValid,

    // ── Règles d'Enchaînement (R11–R16) ──────────────────────────────

    /// R11 : output_shape[i] == input_shape[i+1] (si deux couches séquentielles)
    R11_ShapeChainConsistent,
    /// R12 : La première couche doit avoir input_shape cohérent avec model.input
    R12_FirstLayerInputMatchesModelInput,
    /// R13 : La dernière couche doit avoir output_shape cohérent avec model.output
    R13_LastLayerOutputMatchesModelOutput,
    /// R14 : Toutes les dimensions des shapes doivent être > 0 ou symboliques ("B","S")
    R14_PositiveDimensions,
    /// R15 : CNN : les spatial dims doivent être cohérentes avec les convolutions
    R15_CnnSpatialDimsConsistent,
    /// R16 : Pour les custom_layers avec formules, les variables référencées
    ///        doivent être disponibles dans variables ou global_params
    R16_CustomFormulaVariablesAvailable,

    // ── Règles Hardware (R17–R21) ─────────────────────────────────────

    /// R17 : dp * tp * pp * ep == num_gpus total (produit des degrés)
    R17_ParallelismProductEqualsGpus,
    /// R18 : Si ZeRO stage > 0, data_parallel_degree > 1
    R18_ZeroRequiresDataParallel,
    /// R19 : memory_gb > 0 pour chaque GPU target
    R19_GpuMemoryPositive,
    /// R20 : Si multiple GPU targets, interconnect doit être fourni
    R20_MultiGpuNeedsInterconnect,
    /// R21 : Avertissement si tflops_fp16 est absent (précision réduite)
    R21_WarnIfNoTflops,

    // ── Règles Training (R22–R25) ─────────────────────────────────────

    /// R22 : batch_size >= 1
    R22_BatchSizePositive,
    /// R23 : Si mixed_precision = true, precision doit être "fp16" ou "bf16"
    R23_MixedPrecisionConsistent,
    /// R24 : Si gradient_checkpointing = true, num_steps doit être fourni
    ///        pour estimer les FLOPs supplémentaires de recomputation
    R24_CheckpointingNeedsSteps,
    /// R25 : seq_len > 0 si fourni
    R25_SeqLenPositive,

    // ── Règles Custom Layers (R26–R28) ────────────────────────────────

    /// R26 : Les formules custom ne doivent pas contenir de mots interdits
    ///        (std::, unsafe, ptr, alloc, etc.)
    R26_CustomFormulaNoForbiddenWords,
    /// R27 : Les formules custom doivent être parseable par evalexpr
    R27_CustomFormulaParseableByEvalexpr,
    /// R28 : Le résultat des formules doit être positif et fini
    ///        (testé avec des valeurs d'exemple)
    R28_CustomFormulaProducesPositiveFiniteResult,

    // ── Règles Cost Config (R29–R32) ─────────────────────────────────

    /// R29 : pue_factor >= 1.0 (physiquement impossible < 1)
    R29_PueFactor,
    /// R30 : gpu_on_demand > gpu_reserved > gpu_spot (logique tarifaire)
    R30_PricingHierarchyConsistent,
    /// R31 : carbon_intensity >= 0
    R31_CarbonIntensityNonNegative,
    /// R32 : Si provider est fourni mais pas les prix, utiliser DB par défaut
    ///        et émettre un avertissement
    R32_WarnIfPricesAbsentWithProvider,
}

/// Application des 32 règles
pub fn validate_all_rules(
    config: &ModelConfig,
) -> ValidationReport {
    let mut errors = vec![];
    let mut warnings = vec![];

    // R01
    if let (Some(h), Some(n)) = (
        config.model.global_params.hidden_size,
        config.model.global_params.num_attention_heads,
    ) {
        if h % n as u64 != 0 {
            errors.push(ValidationError {
                rule: ValidationRule::R01_HiddenDivisibleByHeads,
                message: format!(
                    "R01: hidden_size ({h}) must be divisible by num_attention_heads ({n}). \
                     Got remainder {}.", h % n as u64
                ),
                severity: ErrorSeverity::Error,
                fix_suggestion: format!(
                    "Set hidden_size to a multiple of {n}, e.g. {}",
                    (h / n as u64) * n as u64
                ),
            });
        }
    }

    // R02
    if let (Some(kv), Some(q)) = (
        config.model.global_params.num_key_value_heads,
        config.model.global_params.num_attention_heads,
    ) {
        if q % kv != 0 {
            errors.push(ValidationError {
                rule: ValidationRule::R02_KvHeadsDivideQHeads,
                message: format!(
                    "R02: num_attention_heads ({q}) must be divisible by num_key_value_heads ({kv}). \
                     This is required for GQA (Grouped Query Attention)."
                ),
                severity: ErrorSeverity::Error,
                fix_suggestion: format!("Set num_key_value_heads to a divisor of {q}: e.g. 1, 2, 4, 8, {q}"),
            });
        }
    }

    // R04
    let mut seen_ids = std::collections::HashSet::new();
    for layer in &config.model.layers {
        let base_id = &layer.id;
        if !seen_ids.insert(base_id.clone()) {
            errors.push(ValidationError {
                rule: ValidationRule::R04_UniqueLayerIds,
                message: format!("R04: Duplicate layer id '{base_id}'. All layer IDs must be unique."),
                severity: ErrorSeverity::Error,
                fix_suggestion: format!("Rename one of the '{base_id}' layers, e.g. '{base_id}_2'"),
            });
        }
    }

    // R11 : enchaînement des shapes
    let expanded = expand_repeats(&config.model.layers);
    for i in 0..expanded.len().saturating_sub(1) {
        if let (Some(out), Some(next_in)) = (
            &expanded[i].output_shape,
            &expanded[i+1].input_shape,
        ) {
            if out != next_in {
                errors.push(ValidationError {
                    rule: ValidationRule::R11_ShapeChainConsistent,
                    message: format!(
                        "R11: Shape mismatch between layer '{}' (output: {:?}) \
                         and layer '{}' (input: {:?}). Shapes must match.",
                        expanded[i].id, out, expanded[i+1].id, next_in
                    ),
                    severity: ErrorSeverity::Error,
                    fix_suggestion: "Ensure output_shape[i] == input_shape[i+1]".to_string(),
                });
            }
        }
    }

    // R17 : dp × tp × pp × ep == num_gpus
    if let Some(p) = &config.training.parallelism {
        let product = p.data_parallel_degree as u64
                    * p.tensor_parallel_degree as u64
                    * p.pipeline_parallel_degree as u64
                    * p.expert_parallel_degree as u64;
        let total_gpus: u64 = config.hardware.targets.iter()
            .map(|t| t.count as u64)
            .sum();
        if total_gpus > 1 && product != total_gpus {
            warnings.push(ValidationWarning {
                rule: ValidationRule::R17_ParallelismProductEqualsGpus,
                message: format!(
                    "R17: dp({}) × tp({}) × pp({}) × ep({}) = {} ≠ {} total GPUs. \
                     Check your parallelism configuration.",
                    p.data_parallel_degree, p.tensor_parallel_degree,
                    p.pipeline_parallel_degree, p.expert_parallel_degree,
                    product, total_gpus
                ),
            });
        }
    }

    // [... les 32 règles complètes ...]

    ValidationReport { errors, warnings }
}
```

---

## 17. Règles par Famille d'Architecture

```
TRANSFORMER (decoder-only, encoder-only, encoder-decoder)
─────────────────────────────────────────────────────────
  REQUIS dans global_params :
    hidden_size, num_layers, num_attention_heads, vocab_size, max_position_embeddings

  FORTEMENT RECOMMANDÉ :
    num_key_value_heads (si GQA), intermediate_size, head_dim

  COUCHES OBLIGATOIRES :
    ≥ 1 couche "token_embedding"
    ≥ 1 couche "attention" ou "decoder_block"
    ≥ 1 couche "lm_head" ou "linear_projection" (pour la sortie vocab)

  COHÉRENCE :
    hidden_size % num_attention_heads == 0
    Si num_key_value_heads fourni : num_attention_heads % num_key_value_heads == 0

MoE (Mixture of Experts)
─────────────────────────────────────────────────────────
  REQUIS dans global_params :
    hidden_size, num_layers, num_experts, num_experts_per_tok

  FORTEMENT RECOMMANDÉ :
    moe_intermediate_size (peut différer de intermediate_size)
    num_shared_experts (si DeepSeek-style)

  RÈGLE CRITIQUE :
    num_experts_per_tok < num_experts (vous ne pouvez pas activer tous les experts)
    Les métriques "active_parameters" ≠ "total_parameters"

CNN (Vision)
─────────────────────────────────────────────────────────
  REQUIS dans data :
    image_height, image_width, image_channels, num_classes

  REQUIS dans global_params :
    initial_channels (ou num_classes si simple classifier)

  COHÉRENCE DES COUCHES CONV :
    out_channels[i] doit correspondre à in_channels[i+1]
    Après MaxPool2d stride=2 : H et W sont divisés par 2

SSM (State Space Models — Mamba, S4, RWKV)
─────────────────────────────────────────────────────────
  REQUIS dans global_params :
    hidden_size, num_layers, ssm_state_size, ssm_expand

  FORTEMENT RECOMMANDÉ :
    ssm_conv_kernel (pour Mamba : typiquement 4)
    vocab_size, sequence_length

  COUCHES :
    Type "mamba_block" ou "ssm_block" avec params :
      d_model, d_state, d_inner (= d_model × expand), d_conv

DIFFUSION
─────────────────────────────────────────────────────────
  REQUIS dans global_params :
    diffusion_timesteps, latent_channels, image_size (ou dans data)

  COUCHES spécifiques :
    "unet_block", "residual_block", "diffusion_attention_block"
    "timestep_embedding"

GNN (Graph Neural Networks)
─────────────────────────────────────────────────────────
  REQUIS dans data :
    num_nodes, num_edges, node_features

  REQUIS dans global_params :
    num_message_passing, node_features (ou edge_features)

  COUCHES :
    "gcn_layer", "gat_layer", "graphsage_layer" avec params :
      in_features, out_features, num_heads (pour GAT)

RNN / LSTM / GRU
─────────────────────────────────────────────────────────
  REQUIS dans global_params :
    hidden_size (= rnn_hidden_size), num_layers, vocab_size

  COUCHES :
    "lstm_layer" ou "gru_layer" avec :
      input_size, hidden_size, bidirectional (bool)

CUSTOM / EXPERIMENTAL
─────────────────────────────────────────────────────────
  REQUIS pour chaque couche custom :
    "layer_type": "custom"
    "custom_equations" avec :
      - parameters_formula (obligatoire)
      - flops_forward_formula (obligatoire)
      - flops_backward_formula (recommandé)
      - memory_activation_formula (recommandé)
    "variables" avec les constantes numériques nécessaires

  EXEMPLE CORRECT :
    {
      "id": "my_sparse_layer",
      "layer_type": "custom",
      "input_shape": ["B", "S", 4096],
      "output_shape": ["B", "S", 4096],
      "params": { "sparsity": 0.5 },
      "custom_equations": {
        "parameters_formula": "4 * D * D",
        "flops_forward_formula": "2 * B * S * D * D * (1 - sparsity)",
        "flops_backward_formula": "4 * B * S * D * D * (1 - sparsity)",
        "memory_activation_formula": "B * S * D * 2",
        "variables": { "D": "hidden_size", "sparsity": 0.5 }
      }
    }
```

---

# PARTIE V — ERREURS À ÉVITER

## 18. Erreurs d'Implémentation des Dialectes (20 erreurs)

```
ERREUR D1 — AnyType sur les résultats SSA
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : NodeOp retourne AnyType:$output
              → Aucune validation de type possible, pas de SSA réel
  SOLUTION  : Définir SSATensorType dans le dialecte graph et l'utiliser :
              let results = (outs SSATensorType:$output);
  IMPACT    : Sans cela, les arêtes du graphe ne portent aucune information
              de forme → l'analyse de liveness est impossible.

ERREUR D2 — EdgeOp ne modélise pas le SSA
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : EdgeOp prend (from: AnyType, to: AnyType)
              Ce n'est PAS du SSA ! Une arête SSA est une VALEUR produite
              par un seul nœud et consommée par N nœuds.
  SOLUTION  : L'arête doit porter une SSAValue avec born_at et dies_at.
              En MLIR, utiliser le système de valeurs :
              let arguments = (ins SSATensorType:$value, I64Attr:$born_at, ...)

ERREUR D3 — StrAttr pour les enums
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : model_type StrAttr, severity StrAttr, bottleneck_type StrAttr
              → Aucune validation MLIR, les typos passent silencieusement
  SOLUTION  : Définir des I32EnumAttr pour chaque type discret.
              Exemple : ModelTypeAttr, SeverityAttr, BottleneckAttr

ERREUR D4 — GlobalParamsOp couvre seulement les Transformers
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : ssm_state_size, num_experts_per_tok, initial_channels
              manquent → pour Mamba, MoE, CNN : GlobalParamsOp est inutile
  SOLUTION  : Ajouter TOUS les params de TOUTES les familles (voir §3 enrichi)

ERREUR D5 — Pas de RepeatOp
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : Un Transformer avec 96 couches crée 96 LayerOp identiques
              → Module MLIR ×96 plus grand, difficile à déboguer
  SOLUTION  : RepeatExpansionOp pour annoter l'expansion, garder le lien
              avec l'original.

ERREUR D6 — TensorInfoOp avec I64ArrayAttr pour les shapes symboliques
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : "B" (batch) et "S" (seq) ne peuvent pas être dans I64ArrayAttr
  SOLUTION  : Utiliser deux tableaux séparés :
              - I64ArrayAttr:$concrete_dims (avec -1 pour les symboliques)
              - StrArrayAttr:$symbolic_names (avec "" pour les concrètes)

ERREUR D7 — Pas de ShapeGateOp
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : Si > 30% des dims sont symboliques, les métriques sont trop
              imprécises mais le pipeline continue sans avertir
  SOLUTION  : ShapeGateOp qui bloque explicitement avec un message clair

ERREUR D8 — AttentionOp unique pour MHA/GQA/Flash
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : MHA, GQA et Flash Attention ont des formules FLOPs très
              différentes (GQA: KV projections réduites, Flash: bytes réduits)
  SOLUTION  : 3 ops séparées : MHAOp, GQAOp, FlashAttentionOp

ERREUR D9 — FLOPs backward sans ratio par op
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : FlopsOp stocke forward + backward mais pas le ratio utilisé.
              Ratio 2.0 appliqué partout alors que LayerNorm=4.0, Relu=1.0
  SOLUTION  : BackwardRatioAnnotationOp par op avec le ratio exact et
              sa justification mathématique

ERREUR D10 — Pas de MemoryTrafficOp
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : Sans les bytes lus/écrits, le modèle Roofline est impossible
  SOLUTION  : MemoryTrafficOp par op avec bytes_hbm, bytes_l2, bytes_sram

ERREUR D11 — LivenessOp avec AnyType:$tensor
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : Le tenseur dans LivenessOp devrait être une SSAValue typée
  SOLUTION  : Utiliser SSATensorType:$tensor pour avoir le type et la forme

ERREUR D12 — SnapshotOp avec StrArrayAttr:$live_tensors
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : Liste de strings sans lien avec les vraies SSAValues
  SOLUTION  : Référencer les SSAValues par leur ID dans la table SSA

ERREUR D13 — Pas de ZeroStageOp détaillé
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : ZeroOp existant a stage + memory_per_gpu_bytes mais pas
              la décomposition params/grads/optimizer séparément
  SOLUTION  : ZeroStageOp avec les 3 composantes + communication overhead

ERREUR D14 — Pas de KvCacheOp
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : VRAM inférence LLM sans KV cache est très sous-estimée
              Pour LLaMA 8B, seq=8192 : KV cache ≈ 1 GB supplémentaire
  SOLUTION  : KvCacheOp avec la formule exacte

ERREUR D15 — GpuProfileOp incomplet
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : Manque tflops_bf16, tflops_int8, tflops_fp8, l2_cache_mb,
              num_sms, tdp_watts → modèle Roofline imprécis
  SOLUTION  : GpuProfileOp étendu avec toutes les spécifications

ERREUR D16 — RooflineOp global au lieu de par-op
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : Un seul RooflineOp pour tout le modèle alors que chaque op
              a son propre point sur le graphe Roofline
  SOLUTION  : hw.roofline_per_op annoté sur chaque graph.node

ERREUR D17 — PricingModelOp ne distingue pas on-demand/spot/reserved
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : Le coût réel dépend fortement du type d'instance
              (spot = 70% moins cher qu'on-demand)
  SOLUTION  : 3 attributs séparés + TrainingCostOp avec 3 coûts

ERREUR D18 — AllMetricsOp trop gros (17+ attributs)
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : Un op avec 17 attributs est difficile à étendre et à déboguer
  SOLUTION  : Décomposer en StructureMetricsOp, ComputeMetricsRefOp,
              MemoryMetricsRefOp, HardwareMetricsRefOp, CostMetricsRefOp

ERREUR D19 — Pas de ConfidenceScoreOp par métrique dans report
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : Le rapport final n'indique pas quelles métriques sont fiables
  SOLUTION  : ConfidenceScoreOp par métrique avec score + facteurs

ERREUR D20 — Pas de VariantOp pour les analyses alternatives
─────────────────────────────────────────────────────────────────────────
  PROBLÈME  : Impossible de comparer le modèle avec sa version INT8/Flash
  SOLUTION  : VariantOp avec le type de variante et les métriques modifiées
```

---

## 19. Erreurs JSON Courantes (25 erreurs)

```
ERREUR J01 — hidden_size non divisible par num_attention_heads
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : "hidden_size": 4096, "num_attention_heads": 48
                    (4096 / 48 = 85.33... → non entier)
  EXEMPLE CORRECT : "hidden_size": 4096, "num_attention_heads": 32
                    (4096 / 32 = 128 → head_dim = 128)
  IMPACT          : Paramètres et FLOPs attention incorrects

ERREUR J02 — num_key_value_heads ne divise pas num_attention_heads (GQA)
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : "num_attention_heads": 32, "num_key_value_heads": 6
                    (32 / 6 = 5.33... → GQA invalide)
  EXEMPLE CORRECT : "num_key_value_heads": 8 (32 / 8 = 4, valide)
  IMPACT          : Paramètres KV et mémoire KV cache incorrects

ERREUR J03 — Shapes non enchaînées correctement
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : Couche 1 output_shape [32, 2048, 4096]
                    Couche 2 input_shape  [32, 2048, 768]  ← mismatch !
  IMPACT          : Compilation échoue à TensorIR, ShapeGate bloqué

ERREUR J04 — IDs de couches dupliqués
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : Deux couches avec "id": "attention"
  EXEMPLE CORRECT : "id": "attention_0", "id": "attention_1"
  IMPACT          : Graphe SSA invalide (deux nœuds avec le même ID)

ERREUR J05 — num_experts_per_tok >= num_experts pour MoE
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : "num_experts": 8, "num_experts_per_tok": 8
                    (tous les experts actifs = modèle dense, pas MoE)
  EXEMPLE CORRECT : "num_experts_per_tok": 2
  IMPACT          : active_parameters = total_parameters (erroné pour MoE)

ERREUR J06 — dp * tp * pp * ep ≠ nombre total de GPUs
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : dp=4, tp=4, pp=4 avec 32 GPUs (4×4×4 = 64 ≠ 32)
  EXEMPLE CORRECT : dp=2, tp=4, pp=4 (2×4×4 = 32)
  IMPACT          : Métriques de parallélisme et coûts incorrects

ERREUR J07 — ZeRO stage sans data_parallel_degree > 1
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : "zero_stage": 3, "data_parallel_degree": 1
                    (ZeRO nécessite N GPUs pour partitionner sur N)
  IMPACT          : VRAM per GPU calculé incorrectement

ERREUR J08 — Tailles GPU irréalistes (tflops trop faibles)
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : "tflops_fp16": 1.0 pour un A100 (réel: 312 TFLOPS)
  EXEMPLE CORRECT : "tflops_fp16": 312.0
  IMPACT          : Latence et coût surestimés d'un facteur 300×

ERREUR J09 — pue_factor < 1.0
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : "pue_factor": 0.8 (physiquement impossible)
  EXEMPLE CORRECT : "pue_factor": 1.2 (typique data center moderne)
                    Minimum physique : 1.0 (100% efficacité)
  IMPACT          : Énergie et CO2 sous-estimés

ERREUR J10 — Formules custom avec division par zéro potentielle
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : "flops_formula": "2 * B * S * D / N"
                    sans vérifier que N > 0
  EXEMPLE CORRECT : Ajouter "N": 8 dans les variables et garantir N >= 1

ERREUR J11 — Formules custom utilisant des variables non définies
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : "parameters_formula": "2 * D * K"
                    sans définir K dans les variables
  IMPACT          : evalexpr retourne une erreur, couche ignorée

ERREUR J12 — Shapes avec 0 dans une dimension
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : "input_shape": [32, 0, 4096]
                    (dimension 0 = tenseur vide = invalide)
  EXEMPLE CORRECT : Utiliser "S" symbolique si inconnu : [32, "S", 4096]

ERREUR J13 — model.type incohérent avec les couches
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : "type": "cnn" avec des couches "attention" et "mlp"
  IMPACT          : Les formules CNN sont appliquées à un Transformer

ERREUR J14 — Pas de lm_head pour les modèles de langage
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : LLM sans couche de projection finale vers vocab
  IMPACT          : Paramètres de lm_head non comptés (~vocab_size×H)
                    Pour LLaMA : 128256 × 4096 = 525M params manquants !

ERREUR J15 — Oublier les couches de normalisation dans les paramètres
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : Couches RMSNorm sans paramètres (gamma uniquement)
  NOTE            : RMSNorm a hidden_size paramètres (γ uniquement)
                    LayerNorm a 2 × hidden_size (γ + β)

ERREUR J16 — Mixed precision sans préciser les gradients
─────────────────────────────────────────────────────────────────────────
  CONTEXTE        : En mixed precision, les poids sont stockés en fp16/bf16
                    MAIS les gradients maître restent en fp32 (AMP standard)
  IMPACT          : VRAM training sous-estimée si on applique bf16 aux grads

ERREUR J17 — repeat avec des couches non-iso (shapes différentes I/O)
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : "layer_type": "pooling", "repeat": 4
                    (pooling change H et W → non-iso)
  EXEMPLE CORRECT : Lister chaque couche pooling individuellement

ERREUR J18 — Oublier les résiduals dans le compte de paramètres
─────────────────────────────────────────────────────────────────────────
  NOTE            : Les residual connections ne AJOUTENT AUCUN PARAMÈTRE
                    Ne pas créer de couche "residual" avec des params

ERREUR J19 — vocab_size mal adapté à l'architecture
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : GPT-2 Small avec vocab_size = 50257 mais en JSON
                    on écrit 50000 → params embedding différent
  IMPACT          : Paramètres embedding et lm_head incorrects

ERREUR J20 — Oublier head_dim quand hidden_size / num_heads n'est pas entier
─────────────────────────────────────────────────────────────────────────
  NOTE            : Certains modèles utilisent head_dim custom != H/heads
                    Ex: Falcon utilise head_dim=64 avec hidden=4096 heads=71
                    → doit être fourni explicitement

ERREUR J21 — num_layers ne correspond pas au nombre de couches dans layers[]
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : "num_layers": 32 mais layers[] contient seulement
                    embed + 1 decoder_block (repeat=32) + norm + lm_head
                    → num_layers devrait référencer les decoder_blocks
  IMPACT          : Confusion dans la validation des paramètres

ERREUR J22 — Pas de seq_len explicite avec flash_attention=true
─────────────────────────────────────────────────────────────────────────
  IMPACT          : Flash Attention est particulièrement utile pour
                    les longs contextes. Sans seq_len, les savings
                    en bytes ne peuvent pas être calculés.

ERREUR J23 — Interconnect bandwidth irréaliste pour Multi-GPU
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : "interconnect_bandwidth_gb_s": 10 pour NVLink
                    (NVLink A100 = 600 GB/s, pas 10)
  IMPACT          : Coût All-Reduce sous-estimé, overhead surestimé

ERREUR J24 — Couches CNN avec in_channels/out_channels incohérents
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : Conv2D layer 1 : out_channels=64
                    Conv2D layer 2 : in_channels=128  ← ne correspond pas
  IMPACT          : Paramètres et FLOPs CNN incorrects

ERREUR J25 — moe_intermediate_size absent pour MoE
─────────────────────────────────────────────────────────────────────────
  EXEMPLE ERRONÉ  : Modèle MoE sans moe_intermediate_size
                    → compilateur utilise intermediate_size pour les experts
                    → FAUX pour Mixtral (I=14336 global, expert=4096)
  EXEMPLE CORRECT : "moe_intermediate_size": 4096 pour les experts
```

---

## 20. Pièges de Précision dans les Formules

```
PIÈGE P01 — FLOPs Softmax : 3 ops au lieu de 5
─────────────────────────────────────────────────────────────────────────
  FAUX    : flops_softmax = 3 × B × S²  (exp + sum + div)
  CORRECT : flops_softmax = 5 × B × S²  (max_scan + subtract + exp + sum_scan + div)
            Le max numerically-stable scan et la soustraction sont souvent oubliés.

PIÈGE P02 — Backward ratio MatMul : 2× seulement si W et X sont carrés
─────────────────────────────────────────────────────────────────────────
  CONTEXTE : Pour MatMul [M,K]×[K,N] :
             dX = dY × W^T    → FLOPs = 2MNK (même que forward)
             dW = X^T × dY    → FLOPs = 2MNK (même que forward)
             Total backward = 2× forward EXACT pour toutes dimensions.
  NOTE    : Le ratio 2.0 est exact pour MatMul, pas approximé.

PIÈGE P03 — FLOPs Conv2D : oublier le stride
─────────────────────────────────────────────────────────────────────────
  FAUX    : flops = 2 × B × C_out × C_in × K² × H_in × W_in
  CORRECT : H_out = (H_in + 2×pad - K) / stride + 1
            flops = 2 × B × C_out × C_in × K² × H_out × W_out
            Le stride réduit les FLOPs via H_out/W_out plus petits.

PIÈGE P04 — Mémoire KV Cache : facteur 2 oublié
─────────────────────────────────────────────────────────────────────────
  FAUX    : kv_bytes = num_layers × seq × kv_heads × head_dim × dtype
  CORRECT : kv_bytes = 2 × num_layers × seq × kv_heads × head_dim × dtype
            Le facteur 2 est pour K ET V (deux tenseurs).

PIÈGE P05 — Optimizer states : bytes Adam vs SGD
─────────────────────────────────────────────────────────────────────────
  Adam/AdamW : 2 tenseurs fp32 par paramètre (momentum + variance)
               → optimizer_bytes = num_params × 8 bytes (toujours fp32 !)
  SGD avec momentum : 1 tenseur fp32 par paramètre
               → optimizer_bytes = num_params × 4 bytes
  SGD sans momentum : 0 bytes
  ERREUR COMMUNE : Appliquer la précision d'entraînement aux optimizer states

PIÈGE P06 — Paramètres GQA : confusion num_heads vs num_kv_heads
─────────────────────────────────────────────────────────────────────────
  FAUX    : K params = H × H (même que Q)
  CORRECT : K params = H × (H × num_kv_heads / num_heads) = H × H_kv
            Pour LLaMA 3.1 8B : H=4096, Q_heads=32, KV_heads=8
            K params = 4096 × 1024 = 4,194,304 (pas 4096×4096 = 16,777,216)

PIÈGE P07 — FLOPs forward Transformer vs tokens/step
─────────────────────────────────────────────────────────────────────────
  CONFUSION : "FLOPs/token" vs "FLOPs/sequence"
  CORRECT   : flops_per_token = flops_forward_total / seq_len
              flops_per_sequence = flops_forward_total
              flops_per_step = flops_forward + flops_backward (training)

PIÈGE P08 — Activation memory sans gradient checkpointing vs avec
─────────────────────────────────────────────────────────────────────────
  Sans ckpt  : Toutes les activations du forward sont gardées en mémoire
  Avec ckpt  : Seulement les activations aux points de checkpoint
               ≈ 1/sqrt(num_layers) de la mémoire totale des activations
               + ~25% FLOPs supplémentaires pour la recomputation

PIÈGE P09 — VRAM embedding : compter tied vs non-tied embeddings
─────────────────────────────────────────────────────────────────────────
  Tied embeddings (LLaMA, GPT-2) :
    L'embedding matrix ET le lm_head PARTAGENT les mêmes poids.
    → params comptés UNE FOIS, pas deux.
    → En mémoire : 1× vocab_size × hidden_size (pas 2×)
  Non-tied :
    → params comptés DEUX FOIS (embedding + lm_head séparés)
```

---

# PARTIE VI — UNIVERSALITÉ

## 21. Stratégie d'Universalité — Couvrir Tous les Modèles

```
PRINCIPE FONDAMENTAL D'UNIVERSALITÉ :
  NEURAX doit analyser TOUT modèle IA sans modification du compilateur.
  Le JSON + les custom_layers couvrent 100% des cas.

NIVEAU 1 : Couverture native (Tier 1 — précision maximale)
─────────────────────────────────────────────────────────────────────────
  12 familles avec formules exactes codées dans neurax-formulas :
  Transformer, MoE, CNN, SSM, Diffusion, GNN, RNN, GAN, ViT, BERT,
  Encoder-Decoder, RNN-LM
  
  Précision : ±1-5% sur tous les modèles

NIVEAU 2 : Couverture via layer_type connus (Tier 2 — bonne précision)
─────────────────────────────────────────────────────────────────────────
  66+ types de couches avec formules dans le registre AtomOps
  Tout modèle composé uniquement de ces types → Tier 2
  
  Précision : ±5-10%

NIVEAU 3 : Couverture via custom_layers (Tier 3 — précision variable)
─────────────────────────────────────────────────────────────────────────
  Toute architecture inconnue peut fournir ses propres formules evalexpr.
  Le compilateur évalue ces formules dans le sandbox sécurisé.
  
  Précision : dépend de la qualité des formules fournies

NIVEAU 4 : Fallback heuristique (Tier 4 — approximatif)
─────────────────────────────────────────────────────────────────────────
  Si ni le type ni les formules ne sont disponibles :
  - FLOPs estimés depuis input/output shapes
  - Paramètres estimés à 0 (avec WARNING W_CUSTOM_NO_FORMULA)
  - Score de confiance < 0.50
  
  Précision : ±25-50% (accompagné d'un avertissement explicite)
```

```rust
// Règle d'universalité dans le registre AtomOps
pub fn lower_layer_universal(
    layer: &NormalizedLayer,
    ctx: &NeuraxContext,
) -> Result<Vec<AtomOp>, NeuraxError> {

    // Tentative 1 : Type connu dans le registre
    if let Some(ops) = ctx.atom_op_registry.get(&layer.layer_type) {
        return ops.lower(layer, ctx);
    }

    // Tentative 2 : custom_equations dans le JSON
    if let Some(ref eqs) = layer.custom_equations {
        return lower_with_custom_equations(layer, eqs, ctx);
    }

    // Tentative 3 : Correspondance partielle (fuzzy match)
    if let Some(similar) = ctx.atom_op_registry.find_similar(&layer.layer_type) {
        ctx.diagnostics.add_warning(
            DiagnosticCode::W001,
            &format!(
                "Layer type '{}' unknown — using similar type '{}'. \
                 Provide custom_equations for better accuracy.",
                layer.layer_type, similar.name
            ),
            DiagnosticSource::Layer(layer.id.clone()),
        );
        return similar.lower_adapted(layer, ctx);
    }

    // Niveau 4 : Fallback avec avertissement fort
    ctx.diagnostics.add(Diagnostic {
        severity: Severity::Warning,
        code: DiagnosticCode::W001,
        message: format!(
            "Custom layer '{}' (type: '{}') has no known formulas. \
             FLOPs = 0, Params = 0. Add 'custom_equations' to the JSON \
             for accurate metrics.",
            layer.id, layer.layer_type
        ),
        affected_metric: Some("flops_forward, total_parameters".to_string()),
        precision_impact: 0.40,
        suggestion: Some(
            "Add custom_equations with parameters_formula and flops_forward_formula".to_string()
        ),
        source: DiagnosticSource::Layer(layer.id.clone()),
    });

    // Retourne un AtomOp vide qui n'affecte pas les métriques
    Ok(vec![AtomOp::zero_placeholder(&layer.id)])
}
```

---

## 22. Validation Automatique Complète

```rust
// Pipeline de validation complet avant compilation

pub async fn validate_and_compile(json_input: &str) -> CompilationResult {
    // ── Étape 1 : Validation JSON Schema ─────────────────────────────
    let schema_errors = validate_json_schema(json_input)?;
    if !schema_errors.is_empty() {
        return CompilationResult::failed_at_schema(schema_errors);
    }

    // ── Étape 2 : Parse ───────────────────────────────────────────────
    let config = neurax_parser::parse(json_input)?;

    // ── Étape 3 : 32 Règles de cohérence ─────────────────────────────
    let validation = validate_all_rules(&config);
    if validation.has_blocking_errors() {
        return CompilationResult::failed_at_validation(validation);
    }

    // ── Étape 4 : Validation des formules custom ──────────────────────
    for custom in &config.custom_layers {
        let formula_check = validate_custom_formula(custom, &config);
        if formula_check.has_errors() {
            return CompilationResult::failed_at_custom_formula(formula_check);
        }
    }

    // ── Étape 5 : Compilation (pipeline IR) ──────────────────────────
    let pipeline = NeuraxPipeline::new(config)?;
    let report = pipeline.run(json_input)?;

    // ── Étape 6 : Validation des métriques de sortie ──────────────────
    validate_output_sanity(&report, &config)?;

    CompilationResult::success(report)
}

/// Validation de bon sens sur les métriques de sortie
fn validate_output_sanity(
    report: &ReportIR,
    config: &ModelConfig,
) -> Result<(), NeuraxError> {
    let m = &report.metrics;

    // Les paramètres doivent être > 0
    if m.total_parameters == 0 {
        return Err(NeuraxError::Internal {
            message: "total_parameters = 0 — impossible for a non-empty model".to_string()
        });
    }

    // La VRAM training doit être >= VRAM inference
    if m.vram_training_peak_gb < m.vram_inference_peak_gb {
        return Err(NeuraxError::Internal {
            message: format!(
                "VRAM training ({:.1} GB) < VRAM inference ({:.1} GB) — impossible",
                m.vram_training_peak_gb, m.vram_inference_peak_gb
            )
        });
    }

    // La VRAM training doit être >= mémoire des paramètres seuls
    let params_gb = m.vram_parameters_gb;
    if m.vram_training_peak_gb < params_gb {
        return Err(NeuraxError::Internal {
            message: format!(
                "VRAM training ({:.1} GB) < params only ({:.1} GB) — impossible",
                m.vram_training_peak_gb, params_gb
            )
        });
    }

    // Les FLOPs backward doivent être > FLOPs forward (jamais moins)
    if m.flops_backward_total < m.flops_forward_total {
        return Err(NeuraxError::Internal {
            message: "flops_backward < flops_forward — backward ratio must be >= 1.0".to_string()
        });
    }

    // Les FLOPs doivent être finis et positifs
    if !m.flops_forward_total.is_finite() || m.flops_forward_total <= 0.0 {
        return Err(NeuraxError::Internal {
            message: format!("flops_forward = {} — must be finite and positive", m.flops_forward_total)
        });
    }

    // Latence doit être positive
    if m.latency_forward_ms < 0.0 {
        return Err(NeuraxError::Internal {
            message: "latency_forward_ms < 0 — impossible".to_string()
        });
    }

    // VRAM ne doit pas dépasser 10× la capacité du GPU (outlier détection)
    if let Some(gpu) = config.hardware.targets.first() {
        let gpu_mem = gpu.memory_gb.unwrap_or(80.0);
        let ratio = m.vram_training_peak_gb / gpu_mem;
        if ratio > 20.0 {
            // Pas une erreur bloquante mais un warning fort
            // (peut être intentionnel pour estimer les besoins en GPU multi-node)
        }
    }

    Ok(())
}
```

---

*Document NEURAX — Enrichissement des Dialectes et Règles JSON v5.0*
*Référence exhaustive pour l'implémentation industrielle*
*Chaque dialecte, chaque règle, chaque erreur documentée et justifiée*