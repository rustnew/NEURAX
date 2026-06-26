## Plan d'Implémentation Final — NEURAX (vérifié, complet, 100 % fonctionnel)

Ce document est le plan d’exécution définitif, après vérification croisée avec le **bilan fonctionnel du frontend** et l’**état d’implémentation actuel**. Chaque fonctionnalité manquante, partielle ou orpheline du frontend est couverte par une phase précise. L’objectif est d’atteindre **100 % de complétude** sans aucune régression.

---

### Vérification de couverture

| Fonctionnalité manquante ou partielle | Phase(s) responsable(s) |
|--------------------------------------|--------------------------|
| **Inference Intelligence** – 22 contrôles + 10 widgets sans backend | Phase 1 |
| **Simulation Real‑Time** – 6 graphes « live » en réalité statiques | Phase 2 |
| **Simulation Comparison** – cross‑hardware, cross‑précision, variantes de modèles | Phase 3 |
| **Sauvegarde cloud des projets** – localStorage uniquement | Phase 4 |
| **Export ONNX binaire réel** – seul du code Python est généré | Phase 5 |
| **Agent IA – panneau de crédit / limite** – UI orpheline | Phase 6 (crédits + Stripe) |
| **Time Machine Compliance** – données statiques | Phase 6 (endpoint de configuration réglementaire dynamique) |
| **Pricing / Stripe / Supabase** – non configurés pour la production | Phase 6 |
| **Authentification et rôles** – bypass dev uniquement | Phase 6 |
| **Tests E2E, documentation, déploiement** | Phase 7 |

**Aucune fonctionnalité du bilan frontend n’est laissée de côté.** Les variantes de modèles (Comparison) sont ajoutées en Phase 3, la compliance devient dynamique en Phase 6, et les crédits de l’agent sont intégrés à la facturation en Phase 6.

---

# Plan d’Implémentation Final

## 1. État des lieux (rappel)

| Page | % actuel | Écarts |
|------|:--------:|--------|
| Architecture | 90 % | Export ONNX réel, crédits agent |
| Simulation | 85 % | Streaming live, Comparison vide (hardware, précision, variantes) |
| Production | 80 % | Pas d’ONNX binaire |
| Time Machine | 95 % | Compliance statique |
| Inference Intelligence | **0 %** | Aucune donnée réelle |
| Infra / Système | 60 % | Auth/billing dev-only, pas de persistance cloud |

---

## 2. Architecture cible (inchangée)

```
neurax-service (Rust, actix-web)
├── POST /analyze                    (existant)
├── GET  /analyze/stream             (NOUVEAU – SSE)
├── GET  /analyze/result/{job_id}    (NOUVEAU – récupération du rapport après stream)
├── POST /inference/simulate         (NOUVEAU)
├── POST /projects                   (NOUVEAU)
├── GET  /projects                   (NOUVEAU)
├── GET  /projects/{id}              (NOUVEAU)
├── PUT  /projects/{id}              (NOUVEAU)
├── DELETE /projects/{id}            (NOUVEAU)
├── POST /export/onnx                (NOUVEAU)
├── GET  /credits                    (NOUVEAU – solde et consommation agent)
├── POST /compliance/config          (NOUVEAU – données réglementaires dynamiques)
├── POST /billing/checkout           (existant – à finaliser)
├── POST /billing/portal             (existant – à finaliser)
├── POST /stripe/webhook             (existant – à finaliser)
├── GET  /health                     (existant)
└── GET  /me                         (existant)

neurax-ir (bibliothèque)
├── 10 dialectes existants
├── dialect_inference                (NOUVEAU)
└── events channel                   (NOUVEAU pour streaming)
```

---

## 3. Phases d’implémentation

### Phase 1 – Inference Intelligence (critique, 2 semaines)

**Objectif :** donner vie aux 22 contrôles et 10 widgets de la page Inference Intelligence.

#### 1.1 Nouveau dialecte IR : `dialect_inference`
- Fichier : `neurax-ir/src/dialect_inference.rs`
- Entrée : `ModelConfig` + `InferenceParams` (sampling, context, behavior, stress tests)
- Sortie : `InferenceReport` contenant **tous les widgets demandés** :
  - `stability_index` : score 0-100 + label (Stable/Drift/Unstable/Chaotic)
  - `entropy_evolution` : histogramme par position de token
  - `noise_schedule` (si diffusion)
  - `hallucination_risk` : niveau, score, description
  - `attention_focus` : heatmap textuelle
  - `state_stability` (SSM)
  - `context_degradation` : % de fenêtre effective restante
  - `sampling_volatility` : diversité % / déterminisme %
  - `router_stability` (MoE) : distribution experts, consistance
  - `risk_overview` : 4 risques (Coherence, Overconfidence, Collapse, Degeneration)
- Algorithmes fondés sur les propriétés structurelles et les paramètres de sampling/théorie de l’information.
- **Tests unitaires** : 15 cas (transformer, MoE, SSM, diffusion, extrêmes).

#### 1.2 Endpoint `POST /inference/simulate`
- Payload : `InferenceRequest` complet.
- Appel à `neurax-ir` avec le nouveau dialecte.
- Authentification requise (identique à `/analyze`), timeout 30 s.
- **Tests d’intégration** : 5 scénarios complets.

#### 1.3 Raccordement frontend
- Remplacement de l’état “not available yet” par un appel à l’API (debounce 800 ms).
- Activation des 10 widgets avec les données reçues.
- Gestion des états de chargement / erreur (déjà prévue).

**Livrables :**
- Module `dialect_inference` testé
- Route `/inference/simulate` opérationnelle
- Page Inference Intelligence 100 % fonctionnelle

---

### Phase 2 – Streaming temps réel de la compilation (important, 1 semaine)

**Objectif :** transformer les 6 graphes « Real‑Time » en véritable flux live.

#### 2.1 Trait `AnalysisEventEmitter` dans `neurax-core`
- Trait avec méthode `emit(event: AnalysisEvent)`.
- Implémentation par défaut : `NoOpEmitter` (ne fait rien).
- Implémentation pour le service web : `BroadcastEmitter` utilisant `tokio::sync::broadcast`.

#### 2.2 Événements dans le pipeline
Chaque passe du pipeline émet :
- `phase_start` / `phase_end` (nom, timestamp)
- `diagnostic` (sévérité, message, couche)
- `partial_metric` (FLOPs partiels, mémoire)
- `progress` (pourcentage global)

#### 2.3 Endpoints associés
- `POST /analyze` : retourne immédiatement un `job_id` (l’analyse est lancée en arrière‑plan).
- `GET /analyze/stream?job_id=...` : flux SSE des événements.
- `GET /analyze/result/{job_id}` : rapport final JSON une fois terminé.

#### 2.4 Adaptation du frontend
- Remplacement du snapshot statique par une connexion `EventSource`.
- Les composants de graphes se mettent à jour en temps réel.

**Livrables :**
- Mécanisme d’événements + émetteur broadcast
- SSE endpoint fonctionnel
- Frontend Real‑Time entièrement live

---

### Phase 3 – Comparaisons multi‑hardware, multi‑précision et variantes de modèles (modéré, 4 jours)

**Objectif :** remplir l’onglet Comparison avec toutes les dimensions prévues.

#### 3.1 Extension de `POST /analyze`
- Le champ `env.hardware_configs` devient un tableau `HardwareConfig[]` (rétrocompatible avec un seul objet).
- Chaque `HardwareConfig` peut spécifier le GPU **et** la précision.
- L’analyse est exécutée pour chaque configuration ; le rapport inclut un tableau `comparisons`.

#### 3.2 Comparaison de variantes de modèles
- L’utilisateur peut sélectionner jusqu’à 3 projets sauvegardés (cf. Phase 4).
- Un nouvel endpoint `POST /analyze/compare` accepte plusieurs `model_config` et une seule `HardwareConfig` (ou une par modèle).
- Le rapport de comparaison alimente les mêmes graphes.

#### 3.3 Frontend
- Sélecteur multi‑GPU et multi‑précision dans le panneau de lancement.
- Possibilité d’ajouter des modèles depuis “My Projects” pour comparaison.
- L’onglet Comparison affiche les résultats dynamiquement.

**Livrables :**
- `/analyze` multi‑configurations
- `/analyze/compare` pour variantes de modèles
- Onglet Comparison complètement fonctionnel

---

### Phase 4 – Sauvegarde cloud des projets (modéré, 4 jours)

**Objectif :** persistance multi‑session et base pour la comparaison de variantes.

#### 4.1 Endpoints CRUD sur Supabase
- Table `projects` : `id`, `user_id`, `name`, `canvas_data` (JSON), `model_config`, `created_at`, `updated_at`.
- `POST /projects` → crée un projet.
- `GET /projects` → liste paginée.
- `GET /projects/{id}` → détails.
- `PUT /projects/{id}` → mise à jour.
- `DELETE /projects/{id}` → suppression.
- Validation JSON côté serveur.

#### 4.2 Frontend
- Menu “My Projects” avec liste, chargement, suppression.
- Bouton “Save to cloud” dans la TopNav.
- Synchronisation automatique optionnelle (débounce).

**Livrables :**
- API CRUD complète
- UI de gestion de projets intégrée

---

### Phase 5 – Export ONNX binaire réel (modéré, 3 jours)

**Objectif :** produire un fichier `.onnx` valide, pas seulement du code Python.

#### 5.1 Construction du graphe ONNX
- Utilisation de `tract` ou `onnx-rs` pour créer le graphe à partir de `ModelConfig`.
- Première version : architecture seule (poids aléatoires selon la méthode d’init choisie).
- Vérification de la validité via l’API Python `onnx.checker` dans les tests d’intégration.

#### 5.2 Endpoint `POST /export/onnx`
- Payload : `model_config` + éventuellement `initialization_method`.
- Réponse : binaire `.onnx` (Content-Type `application/octet-stream`).

#### 5.3 Frontend
- Le bouton “Export ONNX” du panneau Export déclenche le téléchargement du binaire.

**Livrables :**
- Module de génération ONNX en Rust
- Endpoint `/export/onnx`
- Téléchargement fonctionnel depuis l’UI

---

### Phase 6 – Mise en production et finalisation des services (1 semaine)

**Objectif :** rendre le produit déployable, sécurisé et monétisable.

#### 6.1 Configuration Stripe et crédits agent
- Variables d’environnement `STRIPE_*` en production.
- Test du flux : inscription → checkout → webhook → mise à jour plan dans Supabase.
- **Crédits agent** :
  - Endpoint `GET /credits` → solde restant.
  - Chaque appel à l’agent consomme un crédit (selon le plan).
  - Les crédits sont réinitialisés mensuellement (via cron ou webhook Stripe).

#### 6.2 Authentification Supabase (mode production)
- Désactivation de `NEURAX_DEBUG_NOAUTH`.
- Protection de tous les endpoints sensibles.
- Pages de redirection pour email de vérification.

#### 6.3 Compliance dynamique (Time Machine)
- Endpoint `GET /compliance/config` qui renvoie les seuils actuels (EU AI Act phases, CSRD, DSA) stockés dans Supabase ou un fichier de config.
- Frontend mis à jour pour appeler cet endpoint au chargement.

#### 6.4 Déploiement
- Dockerfiles multi‑étages pour `neurax-service` et `neurax-agent`.
- `docker-compose.yml` complet (service, agent, nginx pour le frontend buildé, Supabase local optionnel).
- Documentation de déploiement (`DEPLOYMENT.md`).

**Livrables :**
- Stripe opérationnel en production
- Crédits agent fonctionnels
- Compliance dynamique
- Images Docker prêtes

---

### Phase 7 – Audit final, tests E2E et documentation (1 semaine)

**Objectif :** garantir la qualité et la maintenabilité.

#### 7.1 Tests de non‑régression
- Exécution complète des suites existantes (`cargo test`, `npm test`, `pytest`) avant chaque fusion.
- Ajout de tests end‑to‑end avec **Playwright** couvrant les 5 pages et les parcours critiques :
  - Création canvas → Run Analysis → navigation Simulation, Time Machine, Inference.
  - Sauvegarde cloud, export ONNX, comparaison de modèles.

#### 7.2 Audit de sécurité
- Vérification des JWT sur chaque endpoint privé.
- Rate limiting sur `/analyze`, `/inference/simulate`, `/export/onnx`.
- Validation des payloads (taille, échappement).

#### 7.3 Documentation finale
- `API_REFERENCE.md` : chaque endpoint avec exemple de requête/réponse.
- `DESIGN.md` : mise à jour avec nouveaux dialectes et flux.
- `DEPLOYMENT.md` : instructions pas‑à‑pas.
- `CHANGELOG.md` : version 1.0.

**Livrables :**
- Suite E2E complète
- Rapport d’audit
- Documentation exhaustive

---

## 4. Stratégie de tests

| Type de test | Cible | Outils |
|--------------|-------|--------|
| Unitaires | Nouveaux dialectes, utilitaires | `cargo test`, `pytest` |
| Intégration | Endpoints HTTP, interaction DB | `actix_web::test`, `pytest` |
| End‑to‑end | Parcours utilisateur sur les 5 pages | Playwright (TypeScript) |
| Performance | Temps de réponse < 2 s pour modèles types | `wrk` / `k6` |
| Non‑régression | Toute la base de code | GitHub Actions (CI) |

Pipeline CI : déclenchement à chaque PR, blocage si échec.

---

## 5. Documentation associée

Fichiers à créer ou mettre à jour :

- `API_REFERENCE.md`
- `DESIGN.md`
- `DEPLOYMENT.md`
- `CONTRIBUTING.md`
- `CHANGELOG.md`

---

## 6. Planning prévisionnel (6 semaines)

| Semaine | Phase | Contenu principal |
|---------|-------|-------------------|
| 1‑2 | 1 | Dialecte inference + endpoint `/inference/simulate` + raccordement UI |
| 3 | 2 | Streaming temps réel (events, SSE, mise à jour UI) |
| 3‑4 | 3 | Comparaison multi‑hardware, multi‑précision, variantes de modèles |
| 4 | 4 | Sauvegarde cloud (API projects + UI) |
| 4‑5 | 5 | Export ONNX binaire |
| 5 | 6 | Stripe, crédits agent, compliance, Docker, déploiement |
| 6 | 7 | Audit, tests E2E, documentation finale |

**Équipe cible :** 2 développeurs backend + 1 frontend.

---

## 7. Conclusion

Ce plan final couvre **chaque fonctionnalité manquante** identifiée dans le bilan du frontend.  
Il est conçu pour **ne rien casser**, en étendant l’existant sans refonte, avec des tests de non‑régression systématiques.  
À l’issue des 6 semaines, **NEURAX sera 100 % fonctionnel sur l’ensemble des 5 pages**, prêt pour une démonstration publique et une adoption en production.

---

*Plan vérifié le 25 juin 2026 – Aucune fonctionnalité orpheline, dépendances résolues, couverture complète.*