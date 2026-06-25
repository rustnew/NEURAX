# Bilan Complet des Fonctionnalités — Frontend Neurax-UI

> Audit exhaustif de toutes les fonctionnalités présentes sur chaque page du frontend.  
> Généré le : 2026-06-24

---

## Table des matières

1. [Architecture](#1-architecture)
2. [Simulation](#2-simulation)
3. [Production](#3-production)
4. [Time Machine](#4-time-machine)
5. [Inference Intelligence](#5-inference-intelligence)

---

## 1. Architecture

Espace de conception du modèle IA. C'est la page principale depuis laquelle toutes les autres pages reçoivent les données (nœuds, connexions, résultats d'analyse).

---

### 1.1 Barre de navigation globale (TopNav)

| Fonctionnalité | But |
|---|---|
| **Sélection d'architecture** | Choisir la famille du modèle (Transformer, CNN, RNN, MoE, Diffusion, Custom) — change la palette de blocs disponibles |
| **Chargement de preset** | Sélectionner un modèle pré-défini (GPT-2, LLaMA, etc.) pour peupler le canvas automatiquement |
| **Run Analysis** | Déclenche la compilation complète via le service Rust (neurax-service) et remplit toutes les métriques des pages Simulation, Time Machine, etc. |
| **Nouveau canvas** | Crée un canvas vide (dialog de confirmation) |
| **Sauvegarder** | Sauvegarde la configuration du canvas localement (localStorage) |
| **Exporter** | Ouvre le panneau d'export (ONNX, PyTorch, code Python) |
| **Importer** | Ouvre le panneau d'import d'un modèle existant |
| **Clear Canvas** | Supprime tous les nœuds et connexions |
| **Chat IA (bouton)** | Ouvre/ferme le tiroir de l'agent IA |
| **Pricing** | Ouvre la page de tarification |

---

### 1.2 Panneau gauche — Palette de blocs

| Fonctionnalité | But |
|---|---|
| **Catalogue de blocs par catégorie** | Liste tous les blocs disponibles selon l'architecture sélectionnée (attention, normalization, MLP, convolution, embedding, etc.) |
| **Drag & drop vers le canvas** | Glisser un bloc depuis la palette et le déposer sur le canvas pour l'ajouter comme nœud |
| **Filtre par catégorie** | Naviguer entre les catégories de blocs (Core, Attention, Memory, etc.) |

---

### 1.3 Canvas central (ArchitectureCanvas)

#### Navigation & vue

| Fonctionnalité | But |
|---|---|
| **Zoom in / Zoom out** | Boutons et molette de souris — zoom de 10% à 400% |
| **Mode Pan (outil main)** | Cliquer-glisser pour déplacer la vue |
| **Mode Select (outil pointeur)** | Mode par défaut pour sélectionner et déplacer les nœuds |
| **Fit to Screen** | Recentre et ajuste le zoom pour afficher tout le canvas |
| **Minimap** | Vue miniature en bas à droite montrant l'ensemble du canvas et la zone visible |

#### Manipulation des nœuds

| Fonctionnalité | But |
|---|---|
| **Déplacer un nœud** | Cliquer-glisser un bloc pour le repositionner |
| **Sélectionner un nœud** | Clic simple — ouvre l'Inspector Panel en bas |
| **Multi-sélection (box select)** | Dessiner un rectangle en cliquant sur le fond du canvas |
| **Multi-sélection (Ctrl+Clic)** | Ajouter/retirer des nœuds de la sélection |
| **Sélectionner tout** | Raccourci clavier pour tout sélectionner |
| **Dupliquer un nœud** | Via menu contextuel ou raccourci — crée une copie du bloc |
| **Supprimer un nœud** | Touche Suppr ou bouton dans l'Inspector |

#### Connexions

| Fonctionnalité | But |
|---|---|
| **Créer une connexion** | Tirer un fil depuis le port de sortie d'un nœud vers le port d'entrée d'un autre |
| **Sélectionner une connexion** | Clic sur un fil — le met en surbrillance |
| **Supprimer une connexion** | Clic + Suppr sur une connexion sélectionnée |

#### Groupes

| Fonctionnalité | But |
|---|---|
| **Grouper des nœuds** | Sélectionner plusieurs nœuds puis "Group Selected" — crée un groupe (ex: bloc Transformer N×) |
| **Déplacer un groupe** | Déplace tous les nœuds du groupe ensemble |
| **Dégrouper** | Via menu contextuel — dissout le groupe |
| **Supprimer un groupe** | Supprime le groupe et ses nœuds membres |

#### Menu contextuel (clic droit)

| Fonctionnalité | But |
|---|---|
| **Dupliquer** | Crée une copie du nœud cliqué |
| **Supprimer** | Supprime le nœud ou la connexion |
| **Group Selected** | Groupe les nœuds sélectionnés |

---

### 1.4 Panneau inférieur — Inspector Panel

Apparaît quand un nœud ou groupe est sélectionné. Redimensionnable verticalement.

#### Inspection d'un nœud

**Colonne gauche — Paramètres**

| Fonctionnalité | But |
|---|---|
| **Champ Name** | Modifier le nom affiché du bloc sur le canvas |
| **Champ Type** | Lecture seule — type technique du bloc (ex: `multi_head_attention`) |
| **Input Shape** | Lecture seule — forme du tenseur d'entrée calculée par le compilateur |
| **Output Shape** | Lecture seule — forme du tenseur de sortie calculée par le compilateur |
| **Paramètres dynamiques (numériques)** | Modifier les hyperparamètres du bloc (d_model, num_heads, ffn_dim, etc.) — champ texte numérique |
| **Paramètres dropdown** | Sélectionner une valeur parmi des options (activation: relu/gelu/tanh, padding: same/valid, etc.) |
| **Paramètres booléens** | Toggle oui/non pour les options binaires (bias, causal, etc.) |
| **Paramètres en lecture seule** | Valeurs calculées non modifiables |

**Colonne droite — Compiler View (télémétrie par couche)**

| Fonctionnalité | But |
|---|---|
| **Compiler FLOPs** | Nombre d'opérations floating-point pour ce bloc spécifiquement, fourni par le compilateur Rust |
| **Estimated Latency** | Latence estimée de ce bloc sur le hardware sélectionné (ms ou µs) |
| **Activation VRAM** | Mémoire VRAM occupée par les activations de ce bloc (MB/GB) |
| **Parameter Count** | Nombre de paramètres entraînables de ce bloc |
| **Tensor Route** | Affiche `inputShape → outputShape` — route des tenseurs à travers ce bloc |
| **Model Phase** | Phase actuelle du compilateur (Idle, Compiling, etc.) |
| **Arithmetic Intensity** | Ratio FLOPs/bytes du modèle global (FLOPs par byte de mémoire accédée) |
| **Precision** | Précision sélectionnée (fp16, bf16, fp32, int8) |
| **Diagnostics du bloc** | Liste des warnings du compilateur attachés spécifiquement à ce nœud |
| **Lien "Jump to Warnings"** | Clic navigue vers l'onglet Issues du panneau droit |

#### Inspection d'un groupe

| Fonctionnalité | But |
|---|---|
| **Champ Group Name** | Modifier le nom du groupe |
| **Repeat Count (×N)** | Définir combien de fois le groupe est répété (ex: 12 pour un Transformer 12 layers) |
| **Éditeur par bloc du groupe** | Pour chaque nœud du groupe : mêmes champs de paramètres que ci-dessus (nom, numériques, dropdowns, booleans) |

---

### 1.5 Panneau droit — RightPanelTabs

Panneau redimensionnable horizontalement, collapsible. 5 onglets :

#### Tab Tune — Analyse & hyperparamètres

| Fonctionnalité | But |
|---|---|
| **Recommandations du compilateur** | Liste les suggestions d'optimisation (quantization, pruning, architecture changes) |
| **Réglages hyperparamètres globaux** | Ajuster les paramètres du modèle global depuis ce panneau |
| **Score de confiance** | Indicateur de fiabilité de l'analyse en cours |

#### Tab Params — Comptage des paramètres

| Fonctionnalité | But |
|---|---|
| **Total paramètres** | Nombre total de paramètres du modèle (formaté en M ou B) |
| **Peak VRAM** | Mémoire VRAM maximale nécessaire |
| **FLOPs estimés** | Total des opérations floating-point |
| **Profondeur du graphe** | Nombre de couches en série (depth) |
| **Tableau par couche** | Liste toutes les couches avec : nom, params, flops, VRAM, latence |
| **Tab Mémoire** | Répartition activations / poids / gradients |
| **Tab Par couche** | Breakdown détaillé par couche |

#### Tab Metrics — Vue d'ensemble

| Fonctionnalité | But |
|---|---|
| **FLOPs totaux** | Total compute du modèle |
| **VRAM peak** | Mémoire maximale requise |
| **Latence** | Temps d'inférence estimé |
| **Throughput** | Tokens/seconde estimés |
| **Métriques de distribution** | Répartition par type d'opération |

#### Tab Deep — Métriques avancées

| Fonctionnalité | But |
|---|---|
| **Métriques compute avancées** | Arithmetic intensity, roofline position, complexité algorithmique |
| **Métriques mémoire avancées** | Fragmentation, défrag savings, virtual savings, stratégie recommandée |
| **Métriques de stabilité** | Exposant de Lyapunov, chaos index, score de robustesse, fallback FP32 |
| **Métriques comportementales** | Load balance, memory contention, cache locality, sensibilité numérique |

#### Tab Issues — Problèmes & diagnostics

| Fonctionnalité | But |
|---|---|
| **Compteur erreurs / warnings** | Affiche le nombre d'erreurs et warnings totaux |
| **Liste des issues** | Chaque warning/erreur avec : sévérité colorée, message, code, nodeId |
| **Jump to first compiler warning** | Scroll automatique vers le premier warning compilateur quand déclenché depuis l'Inspector |

---

### 1.6 Chat IA — AIChatDrawer

| Fonctionnalité | But |
|---|---|
| **Interface de chat texte** | Envoyer des prompts en langage naturel à l'agent IA |
| **Contexte automatique** | Le snapshot du canvas (nœuds, connexions, hardware, warnings, onglet actif) est envoyé automatiquement à chaque message |
| **Outil add_node** | L'agent ajoute un bloc sur le canvas |
| **Outil connect** | L'agent crée une connexion entre deux nœuds |
| **Outil disconnect** | L'agent supprime une connexion existante |
| **Outil delete_node** | L'agent supprime un nœud |
| **Outil select_node** | L'agent sélectionne un nœud (ouvre l'Inspector) |
| **Outil set_hardware** | L'agent change la configuration matérielle (GPU, précision, batch size) |
| **Outil set_architecture** | L'agent change la famille d'architecture |
| **Outil run_analysis** | L'agent déclenche une analyse complète |
| **Outil navigate_to** | L'agent navigue vers un onglet de workspace (architecture, simulation, production, inference, timemachine) |
| **Outil update_params** | L'agent modifie les paramètres d'un nœud existant |
| **Affichage des actions** | Chaque outil utilisé est résumé dans la bulle de réponse de l'agent |
| **Historique de conversation** | Les échanges précédents sont conservés et renvoyés au LLM pour le contexte |
| **Auto-analysis** | Après chaque action structurelle, l'agent peut déclencher automatiquement une analyse |

---

### 1.7 AgentRunModal

| Fonctionnalité | But |
|---|---|
| **Modal d'exécution** | Affiche les étapes de l'agent en cours (streaming SSE) |
| **Barre de progression** | Indique l'avancement du run agent |
| **Liste des actions** | Affiche chaque outil appelé par l'agent avec ses arguments |

---

## 2. Simulation

Page d'analyse graphique du modèle conçu. Affiche 8 catégories de graphes alimentés par le résultat de compilation Rust.

---

### 2.1 Onglet Real-Time — Compilation en direct (6 graphes)

| # | Graphe | But |
|---|---|---|
| 1.1 | **Global Progress** | Barre de progression de la compilation (phase courante + % avancement) |
| 1.2 | **Timeline des Phases** | Barre colorée segmentée montrant chaque phase compilateur (durée ms, statut completed/inprogress/failed) |
| 1.3 | **Live Diagnostics Feed** | Liste scrollable de tous les diagnostics émis durant la compilation (catégorie, sévérité, message, suggestion) |
| 1.4 | **Partial Metrics (Live)** | Area chart de l'activité partielle au fil du temps (visible si le compilateur émet des métriques partielles) |
| 1.5 | **Confidence Score Live** | Jauge circulaire SVG du score de confiance (0-100, coloré vert/orange/rouge) |
| 1.6 | **Throughput Instantané** | Line chart tokens/seconde émis par le compilateur en temps réel |

---

### 2.2 Onglet Results — Rapport global (8 graphes)

| # | Graphe | But |
|---|---|---|
| 2.1 | **Model Size (Parameters)** | Donut chart de la distribution des paramètres (Weights, Embedding, Bias, Normalization) avec total formaté (M/B) |
| 2.2 | **FLOPs by Op Type** | Bar chart horizontal des GFLOPs consommés par type d'opération (top 6) |
| 2.3 | **Latency Breakdown** | Bar chart des latences par phase compilateur (ms) — si données de phase disponibles |
| 2.4 | **Key Stats Strip** | Bande de 6 métriques textuelles : FLOPs totaux, VRAM peak, Intensité arithmétique, Score de confiance, Throughput, Latence |
| 2.5 | **Confidence Score** | Jauge circulaire avec label textuel (Compiler-backed / Estimated with caution / Low confidence) et description |
| 2.6 | **Hardware Fit Score** | Score /100 d'adéquation GPU (VRAM vs besoin, utilisation GPU, throughput) avec label (Great fit / Good fit / Under-utilised) |
| 2.7 | **Cost Summary (Treemap)** | Treemap proportionnel FLOPs / VRAM / Latence pour visualiser l'équilibre compute vs mémoire |
| 2.8 | **Dialect Distribution** | Camembert des familles d'opérations (DenseSeq, SparseMoE, ConvGrid, Other) |

---

### 2.3 Onglet Per Layer — Breakdown par couche (7 graphes)

| # | Graphe | But |
|---|---|---|
| 3.1 | **FLOPs per Layer** | Bar chart des GFLOPs par couche — identifie les couches les plus coûteuses en compute |
| 3.2 | **VRAM per Layer** | Area chart de la mémoire (MB) par couche — identifie les couches les plus gourmandes en mémoire |
| 3.3 | **Latency per Layer** | Line chart de la latence (ms) par couche — identifie les goulots d'étranglement temporels |

---

### 2.4 Onglet Memory — Deep dive VRAM (6 graphes)

| # | Graphe | But |
|---|---|---|
| 4.1 | **Memory Heatmap (Timeline)** | Heatmap (couches × steps) colorée verte/jaune/orange selon l'activité mémoire — visualise la durée de vie des tenseurs |
| 4.2 | **VRAM Liveness** | Area chart de l'occupation VRAM (MB) au fil des steps de compilation — montre les pics de mémoire |
| 4.3 | **Peak VRAM Breakdown** | Donut chart : Activations / Poids (Weights) / Buffers temporaires — avec total en centre et % par catégorie |
| 4.5 | **Gradient Memory (Training)** | Bar chart stacké Forward + Backward memory par couche — pertinent pour l'entraînement |
| 4.6 | **KV Cache Growth (LLM)** | Line chart de la taille du cache KV (MB) en fonction de la longueur de séquence — montre la croissance O(N) |

---

### 2.5 Onglet Comparison — Benchmarks & variantes (9 slots)

| Fonctionnalité | But |
|---|---|
| **Current device** | Affiche le GPU actuellement sélectionné |
| **Current precision** | Affiche la précision sélectionnée (fp16, bf16, etc.) |
| **Batch size** | Affiche la taille de batch utilisée pour l'analyse |
| *(Slots réservés)* | Infrastructure prête pour comparaisons cross-hardware et cross-précision futures |

---

### 2.6 Onglet Optimization — Roofline & goulots d'étranglement (5 graphes)

| # | Graphe | But |
|---|---|---|
| 6.1 | **Roofline Model** | Graphe roofline avec Memory roof (rouge), Compute roof (vert) et point du modèle actuel (bleu) — positionne le modèle par rapport aux limites hardware |
| 6.2 | **Bottleneck Pareto (80/20)** | Combo bar+line chart : FLOPs par couche (barres) + courbe cumulative % — identifie les 20% de couches qui font 80% du compute |
| 6.3 | **Compute vs Memory Bound** | Pie chart en 3 parts : Compute-bound / Memory-bound / Mixed — déduit du roofline position et score de confiance |
| 6.4 | **Optimization Opportunities** | Barres de progression pour chaque opportunité (Quantization, Sparsity, Attention Fusion, etc.) avec score et description |
| 6.5 | **Layer Fusion Candidates** | Liste des paires de couches candidates à la fusion (ex: Linear+ReLU) avec gain estimé % et difficulté d'intégration |

---

### 2.7 Onglet Training — Coût & runtime (6 champs)

| Métrique | But |
|---|---|
| **Train Cost ($)** | Coût total estimé d'un entraînement complet en dollars |
| **Duration (h)** | Durée estimée de l'entraînement en heures |
| **Energy (kWh)** | Consommation électrique estimée en kWh |
| **Carbon Footprint (kg CO₂)** | Empreinte carbone estimée de l'entraînement |

---

### 2.8 Onglet Debugging — Diagnostics & confiance (8 graphes)

| # | Graphe | But |
|---|---|---|
| 8.1 | **Diagnostic Severity** | Pie chart de distribution des diagnostics par sévérité (critical/warning/info/hint) |
| 8.2 | **Diagnostics by Layer** | Matrice couche × catégorie (Shape, Memory, Parallel, Op, Config, General) — case colorée rouge/orange/bleu selon le nombre de diagnostics |
| 8.3 | **Shape Confidence** | Bar chart du niveau de confiance de résolution des dimensions tensorielles (Global, Partial, Unresolved) |
| 8.5 | **OpKind Distribution** | Pie chart des types d'opérations présents dans le graphe (top 6) |
| 8.6 | **Unsupported Ops / Fallbacks** | Liste des opérations non supportées avec sévérité, message de fallback et nombre d'occurrences |
| 8.7 | **Resolution Distribution** | Bar chart du statut de résolution des tenseurs (resolved, partial, unresolved) |
| 8.8 | **Penalty Impact Waterfall** | Waterfall chart montrant l'impact de chaque type de pénalité sur le score de confiance final |

---

## 3. Production

Page de préparation du modèle pour l'entraînement — initialisation des poids et configuration des hyperparamètres (Green AI).

---

### 3.1 Toolbar

| Fonctionnalité | But |
|---|---|
| **Copy Code** | Copie le code Python Green AI généré dans le presse-papier |
| **Export ONNX** | Télécharge un fichier `.py` contenant le modèle pré-initialisé exportable en ONNX |

---

### 3.2 Colonne gauche — Poids & biais optimaux

**Métriques rapides**

| Métrique | But |
|---|---|
| **Epochs Saved** | Nombre d'époques d'entraînement économisées grâce à la pré-initialisation |
| **Hours Saved** | Heures de calcul économisées |
| **Data Efficiency** | Gain d'efficacité sur les données (+%) |

**Sélection de la méthode d'initialisation**

| Méthode | But |
|---|---|
| **Xavier Normal** | Pour les activations linéaires — équilibre variance entrée/sortie |
| **Xavier Uniform** | Variante uniforme de Xavier |
| **He Normal** | Optimisé pour ReLU — double la variance de Xavier |
| **He Uniform** | Variante uniforme de He |
| **Sparse** | Initialisation sparse (zéros avec quelques valeurs non nulles) — réduit la mémoire |
| **Orthogonal** | Matrice orthogonale — stabilité des gradients pour les RNN |
| **Identity** | Matrice identité — pour les connexions résiduelles |
| **Random Normal** | Initialisation gaussienne standard |
| **Badge BEST** | Indique la méthode recommandée automatiquement selon le type de modèle |
| **Bouton "Use Recommended"** | Applique automatiquement la meilleure méthode |
| **Tooltip par méthode** | Affiche description complète + formule mathématique au survol |

**Options avancées (collapsibles)**

| Option | But |
|---|---|
| **Gain Factor (slider 0.1–3.0)** | Facteur multiplicatif appliqué à l'initialisation |
| **Sparsity (slider 50%–99%)** | Uniquement pour la méthode Sparse — pourcentage de poids à zéro |

**Résumé des couches entraînables**

| Fonctionnalité | But |
|---|---|
| **Liste des couches** | Affiche les badges de toutes les couches entraînables (dense, conv2d, attention, transformer, layernorm, batchnorm) |

---

### 3.3 Colonne droite — Hyperparamètres d'entraînement

| Paramètre | But |
|---|---|
| **Learning Rate (slider)** | Taux d'apprentissage (1e-5 à 1e-2, format exponentiel) |
| **Dropout (slider 0–0.5)** | Taux de dropout pour la régularisation |
| **Weight Decay (slider 0–0.1)** | Décroissance des poids pour la régularisation L2 |
| **Warmup Steps (slider 0–5000)** | Nombre de steps de warmup du learning rate scheduler |
| **Gradient Clipping (slider 0.1–5.0)** | Valeur maximale du gradient (évite l'explosion des gradients) |
| **Optimizer (select)** | Sélection de l'optimiseur : Adam / AdamW / SGD |
| **Bouton "Reset to Recommended"** | Remet tous les hyperparamètres aux valeurs recommandées calculées selon l'architecture |

**Sustainability Impact**

| Métrique | But |
|---|---|
| **Gradient Flow Score** | Score de qualité du flux de gradient (0–100) — plus élevé = convergence plus stable |
| **Convergence Boost** | Gain de vitesse de convergence estimé (%) par rapport à une init aléatoire |
| **Memory Saved (Sparse)** | % de mémoire économisée par la sparsité — visible uniquement pour la méthode Sparse |

---

## 4. Time Machine

Page de projection temporelle du coût, de l'empreinte carbone et de la conformité réglementaire du modèle sur plusieurs années.

---

### 4.1 Panneau gauche — Paramètres What-If

| Paramètre | But |
|---|---|
| **User Growth (%/yr, slider 0–500%)** | Taux de croissance annuel des utilisateurs — impacte le coût de déploiement |
| **Annual Budget (slider $10k–$5M)** | Budget annuel maximum — une ligne rouge sera tracée sur les graphes |
| **Horizon (slider 1–10 ans)** | Durée de la projection en années |
| **Target Hardware (select)** | Choisir le GPU cible de déploiement : NVIDIA A100 / H200 / B100 |

**Recommendations**

| Fonctionnalité | But |
|---|---|
| **Liste de recommandations** | Chaque recommandation affiche : priorité (high/medium/low), titre, description, économies potentielles, timing optimal |
| **Chargement automatique** | Recalcul de la projection et des recommandations à chaque changement de paramètre (debounce 600ms) |

---

### 4.2 Vue Cost Timeline

| Fonctionnalité | But |
|---|---|
| **Graphe multi-scénarios** | 3 courbes : Nominal (75%), Optimistic (15%), Pessimistic (10%) — en milliers de $ par mois |
| **Ligne de budget limite** | Trait rouge pointillé horizontal au niveau du budget mensuel défini |
| **Points de rupture (Breaking Points)** | Points rouges sur la courbe Nominal indiquant l'année où le budget mensuel est dépassé |
| **Card Year 1 Cost** | Coût annuel estimé la première année |
| **Card Year N Cost** | Coût annuel estimé la dernière année de la projection |
| **Card Cost Growth %** | Croissance totale du coût entre l'année 1 et l'année N |
| **Card Break Point** | Première année de dépassement du budget (ou "None") |

---

### 4.3 Vue Cost Breakdown

| Fonctionnalité | But |
|---|---|
| **Area chart stacké par composant** | Décomposition du coût total par : Compute / Storage / Network / Egress — évolution année par année |

---

### 4.4 Vue Carbon Impact

| Fonctionnalité | But |
|---|---|
| **Area chart CO₂** | 3 courbes : Baseline (tonnes CO₂/yr), Optimized, With Green Regions — projection de l'empreinte carbone |
| **Card Baseline Reduction %** | Réduction de CO₂ obtenue avec les optimisations vs baseline |
| **Card Green Regions %** | Réduction maximale possible en passant sur des régions cloud vertes |
| **Card CO₂ Saved** | Tonnes de CO₂ économisées par an à la dernière année de projection |

---

### 4.5 Vue Compliance

| Fonctionnalité | But |
|---|---|
| **EU AI Act Phase 1 (2027)** | Limite 300 GFLOPs/req — statut "upcoming" ou dans le scope de la projection |
| **EU AI Act Phase 2 (2028)** | Limite 150 GFLOPs/req — contrainte plus sévère |
| **Carbon Reporting CSRD (2026)** | Obligation de reporting carbone — statut actif |
| **Digital Services Act (2026)** | Conformité DSA — statut actif |
| **Badge statut** | "Active" ou "X yr away" selon si la réglementation est dans le scope temporel |
| **Note de mise en garde** | Rappel que les limites réglementaires peuvent évoluer |

---

## 5. Inference Intelligence

Page de simulation du comportement d'inférence du modèle selon différentes configurations de décodage.

---

### 5.1 Panneau gauche — Inference Controls

#### Section Sampling Strategy

| Paramètre | But |
|---|---|
| **Temperature (slider 0–2)** | Contrôle la "créativité" du modèle — 0 = déterministe, 2 = très aléatoire |
| **Top-k (slider 1–100)** | Limite le vocabulaire aux k tokens les plus probables à chaque step |
| **Top-p (slider 0–1)** | Nucleus sampling — garde les tokens couvrant p% de la probabilité cumulée |
| **Beam Width (slider 1–10)** | Nombre de séquences parallèles en beam search (1 = greedy) |
| **Repetition Penalty (slider 1–2)** | Pénalise la répétition des tokens déjà générés |
| **Presence Penalty (slider 0–2)** | Pénalise les tokens déjà présents dans le contexte |
| **Frequency Penalty (slider 0–2)** | Pénalise les tokens fréquents dans le contexte |

#### Section Context Configuration

| Paramètre | But |
|---|---|
| **Prompt Length (slider 128–32768 tokens)** | Longueur du prompt d'entrée — impacte le coût du prefill |
| **Max Output Tokens (slider 64–8192)** | Nombre maximum de tokens à générer |
| **Sliding Window (toggle)** | Active l'attention à fenêtre glissante pour les longs contextes |
| **KV Cache Reuse (toggle)** | Réutilise le cache KV entre les générations pour économiser du compute |

#### Section Model Behavior

| Paramètre | But |
|---|---|
| **Architecture Type** | Lecture seule — affiche la famille d'architecture héritée du canvas (transformer, cnn, moe, etc.) |
| **Attention Type (select)** | Standard / Flash Attention / Linear Attention — impacte le profil de performance |
| **MoE Router Mode (select)** | Visible uniquement pour architecture MoE — Top-k Routing / Expert Choice / Soft Routing |
| **Quantization Level (select)** | FP32 / FP16 / BF16 / INT8 / INT4 — précision numérique des poids à l'inférence |

#### Section Stability Stress Test

| Test | But |
|---|---|
| **Long-context Simulation (toggle)** | Simule le comportement avec des contextes étendus (dégradation des performances) |
| **Adversarial Prompt (toggle)** | Simule des patterns d'entrée adversariaux (robustesse) |
| **High-temperature Mode (toggle)** | Force temperature=2 pour tester la diversité maximale |
| **Low-temperature Mode (toggle)** | Force temperature=0 pour tester le comportement déterministe |

---

### 5.2 Panneau droit — Behavior Prediction Dashboard

> **État actuel** : Infrastructure complète implémentée, données mock supprimées.  
> Le message affiché est : *"Behavior prediction metrics are not available yet — once the backend exposes inference/behavior signals, this dashboard will render them."*

Les widgets suivants sont architecturalement présents et s'activeront quand le backend exposera les données :

| Widget | But prévu |
|---|---|
| **Generation Stability Index** | Jauge circulaire : Stable / Drift / Unstable / Chaotic — évalue la cohérence de la génération |
| **Entropy Evolution** | Histogramme de l'entropie par position de token — détecte les pics d'incertitude |
| **Noise Schedule Curve** | Courbe du bruit (pour les modèles diffusion) |
| **Hallucination Risk Card** | Badge Low/Medium/High + barre de confiance + description du risque |
| **Attention Focus Simulation** | Visualisation colorée des tokens par niveau d'attention (heat map textuelle) |
| **State Stability (SSM)** | Jauge circulaire de cohérence de l'état caché pour les modèles SSM/Mamba |
| **Context Degradation** | Barre de progression de la fenêtre de contexte effective restante |
| **Sampling Volatility** | Diversité (%) vs Déterminisme (%) — métriques de variabilité de l'échantillonnage |
| **Router Stability (MoE)** | Score de consistance du routage + distribution par expert (histogramme) |
| **Inference Risk Overview** | Tableau de 4 risques : Coherence / Overconfidence / Collapse (MoE) / Degeneration — chacun Low/Medium/High |

---

## Résumé par page

| Page | Nb de fonctionnalités majeures | État |
|---|---|---|
| **Architecture** | ~45 fonctionnalités | ✅ Complet et opérationnel |
| **Simulation** | ~41 graphes et vues | ✅ Complet — données réelles du compilateur Rust |
| **Production** | ~22 fonctionnalités | ✅ Complet et opérationnel |
| **Time Machine** | ~18 fonctionnalités | ✅ Complet — données réelles du compilateur Rust |
| **Inference Intelligence** | ~22 paramètres + 10 widgets prévus | ⚠️ Contrôles complets, dashboard comportemental en attente de données backend |

---

*Fichier généré automatiquement depuis l'audit du code source `neurax-ui/src`.*
