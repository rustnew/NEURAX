# État d'Implémentation — Fonctionnalités Frontend Neurax-UI

> Audit croisé frontend ↔ backend : quelles fonctionnalités sont complètes, partielles ou manquantes côté backend.  
> Généré le : 2026-06-25  
> Services backend : `neurax-service` (Rust, port 9098) · `neurax-agent` (Python, port 8099)

---

## Légende

| Symbole | Signification |
|---|---|
| ✅ | Implémenté et fonctionnel — frontend + backend connectés |
| ⚠️ | Partiellement implémenté — fonctionne avec limitations notables |
| ❌ | Non implémenté — UI présente, pas de backend correspondant |

---

## 1. Page Architecture

### 1.1 Canvas & manipulation des nœuds

| Fonctionnalité | État | Détail |
|---|---|---|
| Ajout de blocs (palette drag & drop) | ✅ | Frontend seul — aucun appel backend requis |
| Déplacement / redimensionnement des nœuds | ✅ | Frontend seul |
| Multi-sélection (box select, Ctrl+clic) | ✅ | Frontend seul |
| Connexions entre nœuds | ✅ | Frontend seul |
| Suppression de connexions | ✅ | Frontend seul |
| Duplication de nœuds | ✅ | Frontend seul |
| Suppression de nœuds | ✅ | Frontend seul |
| Groupement / dégroupement de nœuds | ✅ | Frontend seul |
| Zoom / Pan / Fit to screen | ✅ | Frontend seul |
| Minimap | ✅ | Frontend seul |
| Sélection de la famille d'architecture | ✅ | Frontend seul — change la palette |
| Sauvegarde du canvas | ✅ | localStorage — aucun backend |
| Nouveau canvas / Clear canvas | ✅ | Frontend seul |

### 1.2 Inspector Panel (nœud / groupe)

| Fonctionnalité | État | Détail |
|---|---|---|
| Édition des paramètres d'un nœud (nom, numérique, dropdown, toggle) | ✅ | Frontend seul — state local |
| Édition des paramètres d'un groupe (nom, Repeat ×N) | ✅ | Frontend seul |
| Input Shape / Output Shape (lecture seule) | ✅ | Rempli par `POST /analyze` via neurax-ir |
| **Compiler View** — FLOPs, Latence, VRAM, Params (par couche) | ✅ | Rempli par `POST /analyze` — données réelles compilateur Rust |
| Compiler Signals (Tensor Route, Model Phase, Arithmetic Intensity, Precision) | ✅ | Rempli par `POST /analyze` |
| Diagnostics attachés au nœud | ✅ | Rempli par `POST /analyze` |
| Lien "Jump to Warnings" | ✅ | Frontend seul — navigation vers RightPanel Issues |

### 1.3 Panneau droit — RightPanelTabs

| Onglet | Fonctionnalité | État | Détail |
|---|---|---|---|
| **Tune** | Réglages hyperparamètres + recommandations | ✅ | Recommandations issues de `/analyze` |
| **Params** | Total paramètres, VRAM peak, FLOPs, profondeur, tableau par couche | ✅ | Rempli par `POST /analyze` |
| **Metrics** | FLOPs, VRAM, latence, throughput, distribution | ✅ | Rempli par `POST /analyze` |
| **Deep** | Arithmetic intensity, roofline, fragmentation mémoire, stabilité numérique | ✅ | Rempli par `POST /analyze` |
| **Issues** | Liste erreurs & warnings compilateur avec compteurs | ✅ | Rempli par `POST /analyze` |
| Collapse / redimensionnement du panneau | ✅ | Frontend seul |

### 1.4 Barre de navigation (TopNav)

| Fonctionnalité | État | Détail |
|---|---|---|
| **Bouton Run Analysis** | ✅ | `POST /analyze` — pipeline Rust neurax-ir complet |
| Chargement des presets | ✅ | `GET /presets` + `GET /presets/{id}` — implémenté neurax-service |
| Sélection du hardware (GPU, précision, batch size) | ✅ | Valeurs envoyées dans l'`env` de `POST /analyze` |
| Liste des GPUs supportés | ✅ | `GET /hardware` — implémenté neurax-service |
| Export (panneau d'export) | ⚠️ | Génère du code Python localement — pas de binaire ONNX réel via backend |
| Import (panneau d'import) | ✅ | Import JSON canvas — frontend seul |
| Pricing | ⚠️ | UI présente — paiement Stripe requis (voir §Billing) |

### 1.5 Chat IA — Agent (AIChatDrawer)

| Outil agent | État | Détail |
|---|---|---|
| `add_node` | ✅ | Traité côté frontend (`handleAgentToolEvent`) + snapshot Python |
| `set_node_params` | ✅ | Traité côté frontend + snapshot Python |
| `move_node` | ✅ | Traité côté frontend + snapshot Python |
| `connect` | ✅ | Traité côté frontend + snapshot Python (avec détection de cycles et fan-in) |
| `disconnect` | ✅ | Traité côté frontend + snapshot Python |
| `delete_node` | ✅ | Traité côté frontend + snapshot Python |
| `clear_canvas` | ✅ | Traité côté frontend + snapshot Python |
| `set_family` | ✅ | Traité côté frontend + snapshot Python |
| `set_hw_config` | ✅ | Traité côté frontend + snapshot Python |
| `navigate_to` | ✅ | Traité côté frontend uniquement (pas de persistance dans snapshot Python) |
| `run_analysis` | ✅ | Déclenche `handleRunAnalysis` → `POST /analyze` |
| `select_node` | ✅ | Traité côté frontend uniquement |
| Interface chat (envoi de prompts, historique) | ✅ | `POST /runs` + SSE `GET /runs/{id}/events` |
| Affichage du plan de l'agent (steps) | ✅ | Stream SSE `plan` event |
| Auto-analyse après chaque action structurelle | ✅ | `triggerAgentAutoAnalysis()` → debounce 650ms → `POST /analyze` |
| Panneau de crédit / limite | ⚠️ | UI présente — crédits non connectés à un compteur backend réel |

---

## 2. Page Simulation

> **Note importante** : Tous les graphes ci-dessous reçoivent leurs données de la réponse statique de `POST /analyze`. Il n'y a **pas** de streaming SSE en temps réel depuis le compilateur Rust — la page "Real-Time" affiche un snapshot post-compilation.

### 2.1 Onglet Real-Time

| Graphe | État | Détail |
|---|---|---|
| 1.1 — Global Progress | ⚠️ | Données de `analysis.compilation.total_progress` — **snapshot post-analyse**, pas de streaming live |
| 1.2 — Timeline des Phases | ⚠️ | Données de `analysis.compilation.phase_timeline` — snapshot post-analyse |
| 1.3 — Live Diagnostics Feed | ⚠️ | Données de `analysis.diagnostics` — snapshot post-analyse |
| 1.4 — Partial Metrics (Live) | ⚠️ | Données de `analysis.live_trace.partial_metrics` — snapshot post-analyse |
| 1.5 — Confidence Score Live | ✅ | Données de `analysis.confidenceScore` — valeur réelle compilateur |
| 1.6 — Throughput Instantané | ⚠️ | Données de `analysis.live_trace.throughput_trace` — snapshot post-analyse |

### 2.2 Onglet Results

| Graphe | État | Détail |
|---|---|---|
| 2.1 — Model Size (Parameters) | ✅ | `totalParams` + distribution par type — données réelles `/analyze` |
| 2.2 — FLOPs by Op Type | ✅ | `opsDistribution` — données réelles `/analyze` |
| 2.3 — Latency Breakdown | ✅ | `compilation.phase_timeline` — données réelles `/analyze` |
| 2.4 — Key Stats Strip | ✅ | 6 métriques compilateur — données réelles `/analyze` |
| 2.5 — Confidence Score | ✅ | `confidenceScore` — données réelles `/analyze` |
| 2.6 — Hardware Fit Score | ✅ | Calculé à partir de `peakVramBytes`, `gpuTflops`, `gpuBandwidthGbs` |
| 2.7 — Cost Summary (Treemap) | ✅ | Calculé localement depuis `totalFlops`, `peakVramBytes`, `latencyMs` |
| 2.8 — Dialect Distribution | ✅ | `opsDistribution` — données réelles `/analyze` |

### 2.3 Onglet Per Layer

| Graphe | État | Détail |
|---|---|---|
| 3.1 — FLOPs per Layer | ✅ | `perLayer[].flops` — données réelles `/analyze` |
| 3.2 — VRAM per Layer | ✅ | `perLayer[].vramMb` — données réelles `/analyze` |
| 3.3 — Latency per Layer | ✅ | `perLayer[].latencyMs` — données réelles `/analyze` |

### 2.4 Onglet Memory

| Graphe | État | Détail |
|---|---|---|
| 4.1 — Memory Heatmap (Timeline) | ✅ | `live_trace.memory_heatmap` — données réelles `/analyze` |
| 4.2 — VRAM Liveness | ✅ | `live_trace.memory_liveness` — données réelles `/analyze` |
| 4.3 — Peak VRAM Breakdown | ✅ | `activationVramBytes`, `paramVramBytes`, `peakVramBytes` — données réelles |
| 4.5 — Gradient Memory (Training) | ✅ | `perLayer[].gradientMemMb` — données réelles `/analyze` |
| 4.6 — KV Cache Growth (LLM) | ✅ | Calculé depuis `kvCacheGbPerToken` — données réelles `/analyze` |

### 2.5 Onglet Comparison

| Graphe | État | Détail |
|---|---|---|
| Affichage device / précision / batch size | ⚠️ | 3 champs informatifs affichés — fonctionnel |
| Comparaisons cross-hardware | ❌ | **Stub** — commentaire dans le code : *"Reserved for true cross-hardware report outputs"*. Aucun backend ne génère ces données |
| Comparaisons cross-précision | ❌ | Non implémenté |
| Comparaisons de variantes de modèles | ❌ | Non implémenté |

### 2.6 Onglet Optimization

| Graphe | État | Détail |
|---|---|---|
| 6.1 — Roofline Model | ✅ | `arithmeticIntensity`, `gpuTflops`, `gpuBandwidthGbs` — données réelles `/analyze` |
| 6.2 — Bottleneck Pareto (80/20) | ✅ | `perLayer[].flops` — données réelles `/analyze` |
| 6.3 — Compute vs Memory Bound | ✅ | Dérivé de `rooflinePosition` et `confidenceScore` |
| 6.4 — Optimization Opportunities | ✅ | Calculé localement depuis les métriques `/analyze` |
| 6.5 — Layer Fusion Candidates | ✅ | Calculé localement depuis la topologie + `perLayer` |

### 2.7 Onglet Training

| Métrique | État | Détail |
|---|---|---|
| Train Cost ($) | ✅ | `trainingCostUsd` — données réelles `/analyze` |
| Duration (h) | ✅ | `trainingTimeHours` — données réelles `/analyze` |
| Energy (kWh) | ✅ | `energyKwh` — données réelles `/analyze` |
| Carbon Footprint (kg CO₂) | ✅ | `co2Kg` — données réelles `/analyze` |

### 2.8 Onglet Debugging

| Graphe | État | Détail |
|---|---|---|
| 8.1 — Diagnostic Severity | ✅ | Dérivé de `diagnostics` + `warnings` — données réelles `/analyze` |
| 8.2 — Diagnostics by Layer | ✅ | Dérivé de `diagnostics` par couche |
| 8.3 — Shape Confidence | ✅ | Dérivé de `confidenceScore` + `diagnostics.shapeConf` |
| 8.5 — OpKind Distribution | ✅ | `opsDistribution` — données réelles `/analyze` |
| 8.6 — Unsupported Ops / Fallbacks | ✅ | `unsupportedOps` — données réelles `/analyze` |
| 8.7 — Resolution Distribution | ✅ | Dérivé de `diagnostics.resolution` |
| 8.8 — Penalty Impact Waterfall | ✅ | Dérivé de `confidenceScore`, `diagnostics`, `warnings` |

---

## 3. Page Production

| Fonctionnalité | État | Détail |
|---|---|---|
| Sélection méthode d'initialisation (8 méthodes) | ✅ | Calcul entièrement frontend — `weightInitialization.ts` |
| Badge "BEST" + "Use Recommended" | ✅ | Logique frontend basée sur les types de nœuds présents |
| Options avancées (Gain Factor, Sparsity) | ✅ | Frontend seul |
| Métriques rapides (Epochs Saved, Hours Saved, Data Efficiency) | ✅ | Calcul frontend |
| Résumé des couches entraînables | ✅ | Filtrage frontend sur les nœuds |
| Hyperparamètres (LR, Dropout, Weight Decay, Warmup, Grad Clip, Optimizer) | ✅ | Frontend state seul — aucun backend |
| Bouton "Reset to Recommended" | ✅ | Logique frontend |
| Sustainability Impact (Gradient Flow, Convergence Boost, Memory Saved) | ✅ | Calcul frontend |
| **Copier le code Python** | ✅ | Génération code Python côté frontend — aucun appel backend |
| **Exporter en .py (Green AI ONNX)** | ✅ | Téléchargement fichier `.py` généré localement |
| Export binaire `.onnx` réel | ❌ | Aucun endpoint backend n'exécute le code ni ne génère le binaire ONNX |
| Connexion à un pipeline d'entraînement réel | ❌ | Les hyperparamètres ne sont envoyés à aucun moteur d'entraînement |

---

## 4. Page Time Machine

| Fonctionnalité | État | Détail |
|---|---|---|
| Sliders What-If (Growth Rate, Budget, Horizon, Hardware) | ✅ | Frontend — déclenche automatiquement `POST /timemachine` |
| **Cost Timeline (multi-scénarios)** | ✅ | `POST /timemachine` → neurax-ir — données réelles compilateur |
| **Cost Breakdown** (Compute / Storage / Network / Egress) | ✅ | `POST /timemachine` → neurax-ir |
| **Carbon Impact** (Baseline / Optimized / Green Regions) | ✅ | `POST /timemachine` → neurax-ir |
| **Recommendations** (priorité, savings, timing) | ✅ | `POST /timemachine` → neurax-ir |
| Badge de rupture budget (firstBreakYear) | ✅ | Calculé depuis la réponse `/timemachine` |
| **Compliance View** (EU AI Act, CSRD, DSA) | ⚠️ | Données **statiques** codées en dur dans le frontend — pas de backend. Fonctionnelle en tant que référentiel mais pas connectée aux métriques réelles du modèle |
| Chargement / états d'erreur | ✅ | Géré dans le composant |

---

## 5. Page Inference Intelligence

### 5.1 Panneau gauche — Inference Controls

| Fonctionnalité | État | Détail |
|---|---|---|
| Sliders Sampling Strategy (Temperature, Top-k, Top-p, Beam Width, etc.) | ❌ | UI fonctionnelle mais **non connectée à aucun moteur d'inférence**. Aucun endpoint `/infer` dans neurax-service |
| Context Configuration (Prompt Length, Max Tokens, Sliding Window, KV Cache) | ❌ | UI seule — aucun backend |
| Model Behavior (Attention Type, MoE Router Mode, Quantization Level) | ❌ | UI seule — aucun backend |
| Stability Stress Test (toggles Long-context, Adversarial, Temp modes) | ❌ | UI seule — aucun backend |

### 5.2 Panneau droit — Behavior Prediction Dashboard

| Widget | État | Détail |
|---|---|---|
| Generation Stability Index | ❌ | Message affiché : *"Behavior prediction metrics are not available yet"* |
| Entropy Evolution | ❌ | Infrastructure UI présente — aucun backend |
| Noise Schedule Curve | ❌ | Infrastructure UI présente — aucun backend |
| Hallucination Risk Card | ❌ | Infrastructure UI présente — aucun backend |
| Attention Focus Simulation | ❌ | Infrastructure UI présente — aucun backend |
| State Stability (SSM) | ❌ | Infrastructure UI présente — aucun backend |
| Context Degradation | ❌ | Infrastructure UI présente — aucun backend |
| Sampling Volatility | ❌ | Infrastructure UI présente — aucun backend |
| Router Stability (MoE) | ❌ | Infrastructure UI présente — aucun backend |
| Inference Risk Overview | ❌ | Infrastructure UI présente — aucun backend |

---

## 6. Fonctionnalités système & infrastructure

| Fonctionnalité | État | Détail |
|---|---|---|
| Health check (`GET /health`) | ✅ | Implémenté dans neurax-service |
| Authentification Supabase (mode prod) | ⚠️ | Requiert `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY`. **En dev : bypass automatique** via `NEURAX_DEBUG_NOAUTH=true` |
| Plan utilisateur (`GET /me`) | ⚠️ | Fonctionnel en dev (retourne `plan: "elite"`). Requiert Supabase en prod |
| Billing Checkout Stripe (`POST /billing/checkout`) | ⚠️ | Endpoint backend complet — **requiert** clés Stripe + Supabase. Non fonctionnel localement sans configuration |
| Billing Portal Stripe (`POST /billing/portal`) | ⚠️ | Idem ci-dessus |
| Stripe Webhook (`POST /stripe/webhook`) | ⚠️ | Idem ci-dessus |
| Validation de plugin (`POST /plugin/validate`) | ⚠️ | Endpoint backend implémenté — **aucune UI ne l'appelle** dans l'application actuelle |
| Sauvegarde cloud des projets | ❌ | Les canvas sont sauvegardés uniquement en **localStorage**. Aucun endpoint de persistance multi-session dans neurax-service |
| Export binaire ONNX réel | ❌ | Génère uniquement du code Python `.py` — pas de compilation/exécution côté serveur |

---

## Résumé global

| Page | ✅ Fonctionnel | ⚠️ Partiel | ❌ Manquant |
|---|---|---|---|
| **Architecture** | 28 fonctionnalités | 3 (Export, Pricing, Crédits agent) | 0 |
| **Simulation** | 29 graphes / métriques | 7 (Real-Time = post-analyse, Comparison = stub) | 3 (cross-hardware, cross-précision, variants) |
| **Production** | 11 fonctionnalités | 0 | 2 (binaire ONNX réel, pipeline entraînement) |
| **Time Machine** | 7 fonctionnalités | 1 (Compliance = statique) | 0 |
| **Inference Intelligence** | 0 | 0 | **14** (tous les contrôles + tous les widgets dashboard) |
| **Système** | 2 (health, dev-auth) | 4 (Stripe, Supabase, plugin/validate) | 2 (sauvegarde cloud, binaire ONNX) |

---

## Points critiques à implémenter

> Ces éléments ont une UI complète mais aucun backend correspondant.

1. **Inference Intelligence complète** — Créer un endpoint `/infer` ou `/inference/simulate` dans neurax-service qui accepte les paramètres de sampling (temperature, top-k, top-p, etc.) et retourne les métriques comportementales (stability, entropy, hallucination risk, etc.).

2. **Real-Time streaming compilateur** — Créer un endpoint SSE `/analyze/stream` dans neurax-service pour émettre les phases de compilation en temps réel au lieu d'un snapshot post-analyse.

3. **Sauvegarde cloud des projets** — Ajouter des endpoints CRUD (`POST /projects`, `GET /projects`, etc.) dans neurax-service avec persistance Supabase.

4. **Comparison cross-hardware** — Étendre `POST /analyze` pour accepter plusieurs configurations hardware et retourner des résultats comparatifs.

5. **Configuration Stripe + Supabase** — Nécessaire pour activer la facturation et l'authentification en production.

---

*Fichier généré depuis l'audit croisé de `neurax-ui/src/` (frontend) et `neurax-service/src/`, `neurax-agent/`, `neurax-ir/src/` (backend).*
