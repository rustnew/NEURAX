

## Move Green AI to Production Tab as Training Optimizer

### Overview
Relocate the Green AI initialization panel from the Export dialog to the **Production** workspace tab, and enhance it to display optimal training hyperparameters (dropout, learning rate, etc.) alongside weights and biases -- essentially becoming a "Training Configuration Optimizer" dashboard.

### What Changes

1. **Remove Green AI from Export Panel**
   - Remove the "Green AI" tab from the Export dialog's 5-tab layout (reduce to 4 tabs: Export Formats, NEURAX IR, PyTorch, Rust)
   - Remove the `GreenAIPanel` import from `ExportPanel.tsx`

2. **Create a new Production workspace component** (`src/components/production/ProductionWorkspace.tsx`)
   - Full-page layout that fills the Production tab
   - Top section: a toolbar/header bar with the Green AI branding and key action buttons (Copy Code, Export ONNX)
   - Main content area split into sections:
     - **Optimal Weights & Biases** -- initialization method selector (Xavier, He, LSUV, etc.) with the architecture-aware recommendation, gain/sparsity controls, and per-layer weight summary
     - **Hyperparameters** -- new section showing recommended training hyperparameters:
       - Learning rate (with suggested value based on architecture size)
       - Dropout rate
       - Weight decay
       - Warmup steps
       - Optimizer choice (Adam, AdamW, SGD)
       - Gradient clipping threshold
     - **Sustainability Metrics** -- the existing metrics (epochs saved, compute hours saved, gradient flow score, convergence boost, memory optimization)
   - All values are computed from the current canvas nodes/connections

3. **Add hyperparameter recommendation logic** (`src/utils/weightInitialization.ts`)
   - New function `getRecommendedHyperparams(nodes, connections, config)` that returns suggested hyperparameters based on:
     - Model size (total parameters) to suggest learning rate (e.g., smaller LR for larger models)
     - Architecture type to suggest dropout (e.g., higher for dense networks, lower for conv-heavy)
     - Presence of normalization layers to adjust weight decay
     - Number of layers to suggest warmup steps
   - New `HyperparameterConfig` interface

4. **Wire Production tab in Index.tsx**
   - Pass `nodes`, `connections`, and `modelName` to a new `productionContent` prop on `WorkspaceTabs`
   - Render `<ProductionWorkspace>` as the production tab content (replacing the current placeholder)

5. **Update WorkspaceTabs**
   - No structural changes needed -- it already supports `productionContent` prop; just needs to receive the actual component

### Technical Details

**New file:** `src/components/production/ProductionWorkspace.tsx`
- Accepts props: `nodes: CanvasNode[]`, `connections: Connection[]`, `modelName: string`
- Reuses the existing `initializeArchitecture`, `generateGreenAIONNX`, and `getRecommendedInit` utilities
- Adds a new hyperparameters section with sliders/inputs for each parameter, pre-filled with recommended values
- Users can adjust any value; the sustainability metrics update in real-time

**Modified files:**
- `src/utils/weightInitialization.ts` -- add `HyperparameterConfig` type and `getRecommendedHyperparams()` function
- `src/components/panels/ExportPanel.tsx` -- remove the Green AI tab and `GreenAIPanel` import
- `src/pages/Index.tsx` -- pass `productionContent={<ProductionWorkspace ... />}` to `WorkspaceTabs`

**Hyperparameter recommendation heuristics:**
- Learning rate: `0.001` baseline, scaled down for models > 10M params
- Dropout: `0.1` for transformers, `0.2` for dense/MLP, `0.05` for conv-heavy
- Weight decay: `0.01` default, `0.0` if no normalization layers present
- Warmup steps: `~5%` of estimated total steps based on layer count
- Gradient clipping: `1.0` default, adjustable
- Optimizer: AdamW recommended for transformers, Adam for others

