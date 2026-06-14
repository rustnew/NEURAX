# NEURAX IR Parser — Design Notes

## Pipeline overview

```
JSON file
    │
    ▼
parse_json_to_ir()          ← json_parser.rs
    │  - detects repeat groups
    │  - lifts equations to IrExpr
    │  - maps globals
    ▼
IrDocument                  ← ir.rs
    │  (in-memory IR)
    ├──► render_neurax()     ← neurax_serializer.rs  → .neurax file
    └──► lower_to_config()   (future)                → ModelConfig
```

---

## Key design decision: the interleaved repeat problem

The GraphTransformer JSON has this pattern:

```
gat_layer_0
gnn_norm_0
gat_layer_1
gnn_norm_1
...
gat_layer_7
gnn_norm_7
```

The two types are **interleaved**, not grouped. A naive "find consecutive same-prefix runs"
algorithm would produce:

```
[SINGLE] gat_layer_0
[SINGLE] gnn_norm_0
[SINGLE] gat_layer_1
...
```

Because consecutive layers don't share the same base name.

### Solution: Paired Repeat Detection

The repeat detector needs a second pass that looks for **periodic patterns** across a window:

```
Period 1: [gat_0]          → no repeat
Period 2: [gat_0, norm_0, gat_1, norm_1] → period=2, repeats 8 times
```

Algorithm:
1. Try period lengths P = 1, 2, 3, ...
2. For period P, check if layers[0..P] is structurally equal to layers[P..2P], [2P..3P], etc.
3. If a period of length P repeats at least twice, emit a RepeatBlock with sub-template of P layers.
4. Stop at the first (smallest P) that works.

For the GraphTransformer:
- P=1: gat_layer_0 ≠ gnn_norm_0 → no
- P=2: (gat_0, norm_0) ≈ (gat_1, norm_1) ≈ ... ≈ (gat_7, norm_7) → YES

This produces the ideal output:
```neurax
repeat 8 as i {
  layer gat_layer_%i { ... }
  layer gnn_norm_%i  { ... }
}
```

This is the correct behaviour implemented in the .neurax output file.
The current `detect_repeats()` in json_parser.rs implements P=1 only.
The P>1 case is the next implementation step.

---

## Global reference resolution

Currently, the parser stores globals as literal IrValue::Int / IrValue::Float.
A future "reference binding" pass should:

1. Scan all layer params for values that match a global's value.
2. If a param's value equals a global's value (e.g. `hidden_size = 512` in params,
   and `hidden_size = 512` in globals), replace IrValue::Int(512) with
   IrValue::GlobalRef("hidden_size").

This makes the .neurax output use `global.hidden_size` instead of the raw `512`,
which is more maintainable and matches the design intent.

Heuristic: only bind if the global key name appears as a substring of the param key name,
OR if the value is shared by 3+ params (likely not a coincidence).

---

## Equation AST parsing (deferred)

Currently, custom_equations strings are stored as IrExpr::Raw("...") — a raw string.
A future pass can parse these into full IrExpr trees:

```
"2 * N * E * H * D * D_head"
→ BinOp(
    Lit(2), Mul,
    BinOp(Var("N"), Mul,
      BinOp(Var("E"), Mul,
        BinOp(Var("H"), Mul,
          BinOp(Var("D"), Mul, Var("D_head"))))))
```

This enables:
- Syntax checking at parse time
- Dimensional analysis (are the units consistent?)
- Symbolic simplification
- Code generation (emit Rust/Python expressions directly)

The parser for this is a standard Pratt parser — straightforward to implement with nom.

---

## IR → ModelConfig lowering (future)

The final step in the pipeline is lowering IrDocument → ModelConfig.
This pass:
1. Expands RepeatBlocks back into flat Vec<Layer> (substituting %i)
2. Resolves GlobalRef values to their concrete IrValue
3. Evaluates IrExpr::BinOp trees to concrete numbers where possible
4. Maps IrLayer.layer_type string → LayerType enum
5. Maps IrValue → the exact Rust types expected by LayerParams

Once this lowering pass exists, the entire pipeline is:

```
.json  ──► parse_json_to_ir() ──► IrDocument ──► lower_to_config() ──► ModelConfig
.neurax ──► parse_neurax_to_ir() ─────────────────────────────────────► ModelConfig
```

Both input formats produce identical ModelConfig — the rest of NEURAX is untouched.
