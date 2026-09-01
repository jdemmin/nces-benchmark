# GLOSSARY.md

This document is the canonical vocabulary for the NCES benchmark project. A term defined here means exactly one thing in code, log output, JSON artifacts, CLI help text, and prose.

For instructions on running the project, see `README.md`. For settings values, see `input/*.json`. This file defines terms only; it deliberately duplicates neither.

---

## 1. What the project measures

The benchmark answers one question:

> **How much does the quality of entity embeddings affect a neural concept learner's ability to recover a target concept from examples?**

Two **embedding conditions** are compared while every other factor — knowledge base, learning problems, splits, NCES architecture, epochs, seed — is held constant:

| Condition | Entity embeddings supplied to NCES |
| --- | --- |
| `dice` | Trained on the knowledge graph by a DICE model, selected from a hyperparameter search. |
| `random` | Deterministic uniform random vectors, identical CSV schema. Dimensionality aligns with NCES embedding dimension. |

Any difference in the resulting metrics is therefore attributable to the embeddings alone.

### The learning problem, formally

Given a knowledge base \( K \), a target concept \( T \), positive examples \( E^+ \), and negative examples \( E^- \), the learner must find a concept expression \( C \) such that, for \( K' = K \cup \{T \equiv C\} \):

$$ \forall e^+ \in E^+ : K' \models C(e^+) \qquad\text{and}\qquad \forall e^- \in E^- : K' \not\models C(e^-) $$

The benchmark scores how close NCES gets to this ideal, measured over the **extension** of the hypothesis rather than over the sampled examples (§6).

---

## 2. Core domain terminology

Description Logic and ontology-learning terms, describing the subject matter independent of this project's implementation.

### Ontology structure

- **Knowledge base** — the ontology being learned from: an OWL file and its benchmark context.
- **Knowledge graph** — the graph view of a knowledge base, as entities and RDF triples. The form DICE trains on.
- **RDF triple** — a `(subject, predicate, object)` statement extracted from the knowledge base. Only IRI-to-IRI triples are used; literals are dropped, as they cannot be embedded as entities.
- **TBox** — terminological knowledge: the class hierarchy and concept definitions.
- **ABox** — assertional knowledge: individuals, their types, and their property assertions.
- **Individual / instance** — a concrete named entity.
- **Class / atomic concept** — a named ontology class such as `Mammal`, with no further composition.
- **Property / role** — a predicate connecting individuals, such as `hasChild`.
- **IRI** — an Internationalized Resource Identifier naming an ontology entity.
- **Local name** — the final fragment of an IRI after the last `#` or `/`. NCES indexes its embedding matrix by local name, so every IRI crossing into NCES is reduced first.

### Concepts and expressions

- **Concept expression** — a DL expression built from classes, roles, and constructors. Rendered to DL syntax (`⊓`, `⊔`, `¬`, `∃`, `∀`) for storage and reporting.
- **Composite concept** — a concept expression combining multiple classes or roles: intersection, union, complement, or a restriction.
- **Target concept** — the concept the learner is expected to recover.
- **Hypothesis** — the concept expression NCES proposes as its answer.
- **Extension** — the set of individuals satisfying an expression under the ontology semantics, computed by the reasoner. Always qualified: *target extension* or *hypothesis extension*.
- **Semantic equivalence** — the target extension and hypothesis extension are exactly equal. The strictest success criterion the benchmark reports; a hypothesis can score high F1 without being semantically equivalent.
- **Complexity** — the multi-dimensional characterization of a target concept's structure and semantic difficulty. Always serialized as an object, never a bare integer. Its fields divide into **structural** measures, computed from the expression alone, and **hardness** measures, which require the reasoner.
- **DL length** — the token count of a rendered DL expression: atomic concepts and constructors each score 1. One structural field within complexity.
- **Nesting depth** — the maximum quantifier nesting of a concept expression. `A ⊓ B` has depth 0; `∃ r.A` has depth 1; `∃ r.(A ⊓ ∃ s.B)` has depth 2.
- **Constructor profile** — the multiset of DL constructors occurring in an expression, as a mapping from constructor to occurrence count.
- **Expressivity class** — the smallest DL fragment containing an expression: `EL`, `ALC`, or `ALCHIQD`. Determined by which constructors are present, independent of the `beyond_alc` setting that permitted them.
- **Hardness** — the reasoner-derived, extension-dependent subset of complexity's fields: extension ratio, atomic baseline F1, and redundancy. Distinguished from structural complexity because it depends on the knowledge base, not only the expression. Not a single number.
- **Extension ratio** — \(\lvert T \rvert / \lvert U \rvert\), the fraction of all individuals in the target extension. Values near 0 or 1 indicate degenerate class balance.
- **Atomic baseline F1** — the best F1 achievable by any single atomic class, computed over extensions. The floor a hypothesis must clear to demonstrate non-trivial learning.
- **Lift** — a hypothesis's F1 minus the atomic baseline F1 of its learning problem. May be negative. The primary outcome of the inferential evaluation (§7.2).
- **Redundant target concept** — a target concept whose extension equals that of some atomic class, making it learnable far below its structural complexity.

### Learning problems

- **Learning problem** — the canonical unit of work: one target concept plus its positive and negative example individuals.
- **Positive examples** (`pos_example`) — individuals belonging to the target concept.
- **Negative examples** (`neg_example`) — individuals not belonging to it.
- **Non-degenerate problem** — a learning problem with at least one positive and one negative example. Degenerate problems are rejected at construction time, as a learner cannot be meaningfully scored on them.

---

## 3. Project components

### Models

- **Embedding model** — the generic term for the entity-embedding model trained on the knowledge graph.
- **DICE model** — the specific architecture, provided by the `dicee` library and configured in `src/models/dice.py`.
- **Entity embedding** — the vector representation of a single ontology entity. Preferred over bare "embedding" when precision matters.
- **NCES model** — the Neural Class Expression Synthesizer from Ontolearn; the concept learner under evaluation.
- **Entity index mapping** — the mapping from an entity's identifier to its row in the embedding matrix.

> **Naming note.** The third-party library is `dicee` (lowercase, as installed from PyPI). This project's module wrapping it is `src/models/dice.py`. Referring to the library, write `dicee`; referring to the module or the model family, write DICE.

### Execution units

- **Benchmark run** — one execution of the full workflow for one (knowledge base, seed) pair. The atomic unit of the benchmark; produces exactly one report.
- **Benchmark suite** — all benchmark runs in a single invocation, across every selected knowledge base and seed.
- **Embedding condition** — `dice` or `random`; the benchmark's independent variable (§1).
- **Hyperparameter trial** — one attempted DICE configuration within a benchmark run's embedding search.
- **Selection metric** — validation MRR, falling back to test MRR; chooses the winning hyperparameter trial.
- **Seed** — the random seed controlling triple splitting, DICE training, and the random embedding baseline. The reproducibility handle; never a synonym for "run" or "dataset".

### Configuration

- **Benchmark configuration** — the complete resolved settings for one invocation, assembled from the four `input/*.json` files plus CLI overrides, and embedded verbatim in every report.
- **Settings file** — one of the four JSON files in `input/`: `project_settings.json`, `data_generation_settings.json`, `embedding_settings.json`, `nces_settings.json`.
- **Split ratio DICE** (`split_ratio_dice`) — the train/validation/test proportions for the RDF-triple split.
- **Split ratio learning problems** (`split_ratio_learning_problems`) — the train/test proportions for the learning-problem split.

---

## 4. Code and workflow vocabulary

### Modules

| Module | Responsibility |
| --- | --- |
| `src/config.py` | Typed settings dataclasses; translates project JSON field names into upstream keyword arguments. |
| `src/paths.py` | Output directory layout and knowledge-base resolution. |
| `src/logging_utils.py` | Console and per-run file logging. |
| `src/data/complexity.py` | Complexity computation: structural measures from DL expressions, hardness measures from the reasoner. |
| `src/data/ontology.py` | OWL parsing, RDF-triple extraction, individual enumeration, extension computation. |
| `src/data/lp.py` | Learning-problem generation, the canonical schema, and splitting. |
| `src/data/results.py` | Result dataclasses returned across the benchmark. |
| `src/models/dice.py` | DICE dataset preparation, embedding training, entity-embedding export, random baseline. |
| `src/models/dice_smac.py` | Hyperparameter search via SMAC3, using a random forest surrogate. |
| `src/models/dice_grid_search.py` | Hyperparameter search via grid search. |
| `src/models/hpo_search_utils.py` | Shared helpers for hyperparameter search. |
| `src/models/nces.py` | NCES training-data preparation, training, and hypothesis evaluation. |
| `src/benchmarking/inference.py` | The inferential evaluation stage (§7). |
| `src/benchmarking/metrics.py` | Extension-based metric calculation and complexity aggregation. |
| `src/benchmarking/runner.py` | The **orchestrator**: coordinates every stage across knowledge bases, seeds, and conditions. |
| `src/__main__.py` | The `nces-benchmark` CLI entry point. |

### Pipeline stages

Each stage's output is the next stage's input.

1. **Ontology parsing** — read the OWL file; extract RDF triples and enumerate individuals.
2. **Learning-problem generation** — produce target concepts with example sets (§5).
3. **Hardness annotation** — compute each target concept's extension, extension ratio, atomic baseline F1, and redundancy flag; populate the hardness fields of its complexity. Extensions are cached for reuse during NCES evaluation.
4. **Learning-problem splitting** — partition problems into disjoint train/test sets.
5. **Embedding stage** — write DICE triple splits, run the hyperparameter search, export the winning entity embeddings, and generate the random baseline.
6. **NCES training** — train the learner on the train split, once per embedding condition.
7. **NCES evaluation** — for each held-out problem, synthesize a hypothesis, compute its extension, and score it against the target extension.
8. **Result aggregation** — summarize per run, per knowledge base, and per suite.
9. **Inferential evaluation** — assemble the paired design and produce the estimates and verdicts of §7.

> Stages 1–4 are seed-independent: they are computed once per knowledge base and reused across all seeds. Their determinism derives from a individual seed, not from the benchmark seed(s). Per default this independent seed is `0`. **Ontology parsing** requires no seed at all.

### Project-specific coding conventions

- **Lazy import** — importing a heavy dependency inside a function rather than at module level, so that `nces-benchmark --help` does not load `torch`.
- **Artifact** — any generated file: JSON, CSV, log, or saved model state.
- **Report** — the structured JSON result for one benchmark run.
- **Summary** — a descriptive aggregate across multiple runs. Distinct from an **evaluation** (§7), which is inferential.
- **Checkpoint** — saved model weights reloaded for evaluation.
- **Postprocessing** — transformations after generation, chiefly the IRI-to-local-name reduction NCES requires.

---

## 5. Learning-problem generation

Learning problems are generated by `generate_learning_problems` in `src/data/lp.py`, which wraps `ontolearn.lp_generator.LPGen`. `LPGen` performs a refinement-based search, writes raw output to `LPs.json`, and the project normalizes that output into the canonical schema below — expanding local names to full IRIs and rejecting degenerate problems.

### Settings requiring translation

Generation is controlled by `input/data_generation_settings.json`. Upstream spells several fields differently; the translations live in `DataGenerationSettings.lpgen_kwargs()`.

| Project field | Upstream name | Note |
| --- | --- | --- |
| `num_rand_samples` | `max_num_lps` | Learning problems to retain. |
| `max_child_len` | `max_child_length` | Max length of a refinement child. |
| `beyond_alc` | `beyond_alc` | Upstream exposes a boolean, not a named operator set. `src/config.py` maps `ALCHIQD → True`, `ALC → False`. |
| `downsample_refinements` | `downsample_refinements` | **Must be `true` whenever `refinement_expressivity < 1.0`.** |
| `refinement_expressivity` | `refinement_expressivity` | Fraction of the refinement space explored. |

Remaining fields (`kbs`, `depth`, `sample_fillers_count`, `num_sub_roots`, `min_num_pos_examples`) pass through unchanged; see the settings file for values.

- **`rho`** — in `KB2Data.__init__`, the local variable holding the constructed `ExpressRefinement` operator, after the conventional DL symbol \(\rho\) for a refinement operator. It is **not** a settable parameter. Three fields shape it: `beyond_alc` toggles the five `use_*` constructor flags as a group, `refinement_expressivity` becomes `expressivity`, and `downsample_refinements` becomes `downsample`.

### Learning-problem schema

```json
{
  "id": "lp_0c6694368bd6",
  "target_concept": "male ⊓ ∃ hasChild.person",
  "pos_example": ["http://example.com/father#stefan"],
  "neg_example": ["http://example.com/father#anna"],
  "complexity": {
    "dl_length": 4,
    "depth": 1,
    "constructors": { "⊓": 1, "∃": 1 },
    "num_atomic_classes": 2,
    "num_roles": 1,
    "expressivity": "EL",
    "extension_size": 2,
    "extension_ratio": 0.33,
    "atomic_baseline_f1": 0.8,
    "redundant": false
  },
  "num_pos": 1,
  "num_neg": 1
}
```

- `id` — stable identifier, invariant across seeds. This is the **join key** of the inferential evaluation (§7.1). It is generated through a hash over the problems's target concept and its positive and negative examples.
- `target_concept` — DL-syntax string.
- `pos_example` / `neg_example` — sorted lists of **full IRIs**. These are *sampled* examples, not the full extension: `num_pos` and `num_neg` count the samples, while `extension_size` counts the target extension over all individuals. In the example above, `extension_ratio = 0.33` with `extension_size = 2` implies \(\lvert U \rvert = 6\).
- `complexity` — the object of §2. The four hardness fields (`extension_size`, `extension_ratio`, `atomic_baseline_f1`, `redundant`) are `null` immediately after generation and populated by the hardness annotation stage. Structural fields are always present.
- `num_pos` / `num_neg` — derived automatically.

The full set is persisted grouped by DL length in `learning_problems.json`; the splits are written as flat lists.

### Splitting

Learning problems are split at the **problem** level, never the example level: splitting one problem's examples across train and test would leak the target concept into evaluation. Splits are disjoint by construction, deterministic in the data-generation settings hash, and stratified by a configurable complexity field — by default `dl_length`.

---

## 6. Metrics

Metrics are **descriptive**: they score one hypothesis against one target extension, for one problem, under one condition. Contrast §7, which is inferential.

Both the hypothesis and the target concept are parsed into OWL class expressions and handed to the reasoner, which returns their extensions. Scoring against extensions over all individuals — rather than against the sampled example sets — prevents a hypothesis that merely separates the provided examples from scoring perfectly.

### Definitions

Let \( P \) be the hypothesis extension, \( T \) the target extension, and \( U \) all individuals in the knowledge base.

| Metric | Definition |
| --- | --- |
| **Precision** | \( \lvert P \cap T \rvert / \lvert P \rvert \) |
| **Recall** | \( \lvert P \cap T \rvert / \lvert T \rvert \) |
| **F1** | Harmonic mean of precision and recall. |
| **Accuracy** | Fraction of \( U \) classified correctly, counting true negatives. Dominated by true negatives when `extension_ratio` is small. |
| **Jaccard** | \( \lvert P \cap T \rvert / \lvert P \cup T \rvert \). Not independent of F1: \( J = F1 / (2 - F1) \) for a fixed pair of sets. |
| **Semantic equivalence** | \( P = T \) exactly. A per-problem boolean; aggregated as a rate, never averaged as a score. |
| **Lift** | F1 minus `atomic_baseline_f1`. Negative when the hypothesis underperforms the best atomic class. |

All ratios return `0.0` on an empty denominator rather than raising.

### Confusion matrix

Reconstructed per problem from the extension counts, with \(\lvert U \rvert\) taken from `target_extension.total`:

$$ TP = \lvert P \cap T \rvert \qquad FP = \lvert P \rvert - TP \qquad FN = \lvert T \rvert - TP \qquad TN = \lvert U \rvert - TP - FP - FN $$

A matrix with any negative cell is inconsistent: it is logged and skipped, never clamped.

- **Matthews correlation coefficient (MCC)** — the skew-robust classification metric, in \([-1, 1]\):

$$ \mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}} $$

  Computed in log space: the denominator is a product of four marginals each as large as the knowledge base, and that product overflows float64 on large ontologies well before the individual counts do. Returns `0.0` when any marginal is zero, matching scikit-learn's degenerate-case convention.

- **Mean MCC** — the unweighted mean over problems; weights every learning problem equally. The quantity the paired analysis operates on.
- **Pooled MCC** — computed from the summed confusion matrix; dominated by problems with large extensions.

> Mean and pooled MCC are distinct results and the divergence is informative: it indicates the embedding effect is concentrated in problems of a particular extension size. Never write "MCC" unqualified.

### Ranking metrics (DICE only)

- **MRR** — mean reciprocal rank. The basis of the selection metric (§3).
- **H@10 / Hits@10** — whether the correct entity appears in the top ten predictions. Reported, not used for selection.
- **KvsAll** — the DICE scoring mode evaluating a query against all entities.

### Per-problem result fields

| Field | Contents |
| --- | --- |
| `id`, `target_concept`, `complexity` | Copied from the learning problem. |
| `hypotheses` | The serialized NCES hypothesis, in DL syntax. |
| `num_pos` / `num_neg` | Counts of the *sampled* examples. |
| `target_positive_count` / `target_negative_count` | Counts over the *full* target extension. |
| `target_extension_size` | Object with `positive`, `negative`, `total`. |
| `accuracy`, `precision`, `recall`, `f1`, `jaccard`, `lift`, `mcc`, `semantic_equivalence` | The metrics above. |
| `runtime_seconds` | Wall-clock time for this problem. |
| `error` | Present only if NCES raised; the problem is then excluded from every aggregate. |

### Run-level fields

| Field | Contents |
| --- | --- |
| `best_embedding_config` | The selected DICE configuration with its score and metrics. |
| `search_trials` | Every attempted hyperparameter trial, including failures. |
| `validation_error` | Explanatory text when validation MRR was unavailable and the test-MRR fallback was used. |

A failed trial or a failed learning problem is **recorded, not fatal**. The benchmark completes and reports partial results rather than aborting.

---

## 7. Inferential evaluation

Implemented in `src/benchmarking/inference.py`, which consumes `SingleRunResult` artifacts written under `nces/`. It reads artifacts; it never re-runs the learner.

> **Naming note.** The pipeline stage is **evaluation**; the module is `inference.py`. "Evaluation" names the stage and its artifact, "inference" the statistical machinery inside it. Never "analysis", "statistics", or "significance testing". When **NCES evaluation** (§4, stage 7) could be meant, write **inferential evaluation** explicitly.

### 7.1 Unit of analysis

- **Atomic observation** — one (learning problem, seed) pair for one outcome. The unit of analysis for the entire stage.
- **Paired difference** — the response variable, always \(d_{ij} = m^{\text{dice}}*{ij} - m^{\text{random}}*{ij}\) for problem \(i\), seed \(j\), outcome \(m\). Differencing cancels the target concept, examples, split, knowledge base, NCES architecture, and seed shared by the two conditions — exactly the isolation the design claims.
- **Paired observation** — the serialized record of one atomic observation: both conditions' values plus the complexity fields needed to bucket or regress it. The `PairedObservation` dataclass.
- **Paired hypotheses** — both conditions' hypothesis strings for one (problem, seed) pair, stored side by side. The `PairedHypotheses` dataclass. Enables the identical-hypothesis fraction and per-concept inspection without re-running anything.
- **Paired design** — every paired observation for one knowledge base, across all outcomes, plus the paired hypotheses, the surviving seeds and problems, and the failure counts. The `PairedDesign` dataclass; serialized as the `design` block (§7.9).
- **Join key** — the learning problem's `id`. Because generation and splitting are seed-independent (§5), an `id` identifies the same target concept in every seed, which is what makes the per-problem random intercept estimable.
- **Pairing** — matching a `dice` result to a `random` result by join key within one seed. A problem contributes an observation for an outcome only when both conditions produced a usable value.
- **Unpaired problem** — a problem present in one condition but not the other. Recorded in `unpaired_problem_ids`; excluded from every estimate.

### 7.2 Outcome hierarchy

The set of outcomes is **pre-specified** — fixed before the numbers are seen. This is what keeps the multiplicity problem small.

| Layer | Outcome(s) | Role |
| --- | --- | --- |
| **Primary** | `lift` | Confirmatory. The single headline claim. |
| **Confirmatory secondary** | `lift` trend in `dl_length` | Confirmatory. Does the advantage grow with complexity? |
| **Mechanism** | `precision`, `recall`, `hypothesis_extension_size` | Descriptive. Never tested confirmatorily. |
| **Robustness** | `lift`, nonparametrically | Agreement check against the primary. Not a second primary. |
| **Exploratory** | `mcc`, `accuracy`, `jaccard`, `semantic_equivalence`, crossed with the bucketings | Screening only. BH-screened, labeled `"role": "exploratory"`. |

- **Confirmatory** — pre-specified, tested, and permitted to support a claim. Exactly two coefficients per knowledge base carry this status: \(\beta_0\) and \(\beta_1\).
- **Descriptive** — reported without a p-value, to interpret the confirmatory result.
- **Exploratory** — screened, not claimed.

Why `lift` is primary: it is F1 minus atomic baseline F1, so it is the only outcome distinguishing learning from recovering a good atomic class. Why the others are demoted: `jaccard` is a monotone reparameterization of F1 (§6), so testing both tests one thing twice; `accuracy` is dominated by true negatives at small extension ratios; `semantic_equivalence` is a rate, not a score.

> **MCC's dual placement.** `mcc` is an *exploratory outcome* in the grid (§7.8). The *classification summary* that reports mean and pooled MCC (§7.7) is *mechanism*. These are different layers and the distinction is deliberate.

### 7.3 Primary analysis — the crossed mixed model

An intercept-only crossed random-effects model on paired differences, fit per knowledge base:

$$ d_{ij} = \beta_0 + u_{\text{seed}(j)} + u_{\text{problem}(i)} + \varepsilon_{ij} $$

| Term | Name | Meaning |
| --- | --- | --- |
| \(\beta_0\) | **mean embedding effect** | The headline estimate. Always reported with an interval. |
| \(u_{\text{seed}(j)}\) | **seed random intercept** | Absorbs DICE/SMAC/NCES training variance shared by every problem in a run. Omitting it treats \(n_{\text{seeds}} \times \lvert\text{test}\rvert\) observations as independent and yields anticonservative p-values. |
| \(u_{\text{problem}(i)}\) | **problem random intercept** | Absorbs concept-level heterogeneity persisting across seeds. Estimable only because the join key is seed-invariant. |
| \(\varepsilon_{ij}\) | residual | — |

- **Crossed** — seeds and problems are crossed, not nested: every problem appears under every seed.
- **Variance decomposition** — the triple (`var_seed`, `var_problem`, `var_residual`). A **result**, not a diagnostic: `var_problem` ≫ `var_seed` means the advantage varies more across concepts than across training runs, which tells a reader whether a single run is trustworthy.
- **REML** — restricted maximum likelihood, the fitting method. `_profile_reml` optimizes the profiled REML criterion over the two variance ratios \((\sigma^2_{\text{seed}}/\sigma^2_\varepsilon,\ \sigma^2_{\text{problem}}/\sigma^2_\varepsilon)\), then back-solves the generalized least-squares estimate.
- **Marginal covariance** — \(V = \sigma^2_{\text{seed}} Z_s Z_s^\top + \sigma^2_{\text{problem}} Z_p Z_p^\top + \sigma^2_\varepsilon I\).
- **Denominator degrees of freedom** — deliberately conservative: \(\min(n_{\text{seeds}} - 1,\ n_{\text{problems}} - 1)\), reported as `df_method: "satterthwaite-conservative"`. With few seeds the seed variance is poorly estimated and the naive \(n - 1\) is far too optimistic.
- **Reduced fit** — when the design cannot identify two variance components (fewer than two seeds or two problems), the model degrades to a paired t-interval, reported as `df_method: "naive-t"` with a `note`. Reported, not raised.

`df_method` takes exactly three values across the stage: `"satterthwaite-conservative"` and `"naive-t"` in the primary block, `"cluster-robust"` in the trend block (§7.5).

### 7.4 Interval agreement

Because `lift` differences are bounded and spike at zero, the model-based interval is not trusted alone.

- **Model-based interval** (`ci95`) — the t-interval from \(\beta_0\) and its standard error, on the conservative degrees of freedom.
- **Cluster bootstrap interval** (`bootstrap_ci95`) — a percentile bootstrap in which **the seed is the resampling unit**. Whole seeds are resampled with replacement, preserving the within-run dependence the seed random intercept exists to absorb. Individual observations are never resampled. `bootstrap_resamples` records the count.
- **Agreement** (`agreement`) — whether the two intervals reach the same zero-exclusion verdict:

| Model CI | Bootstrap CI | Conclusion | `agreement` |
| --- | --- | --- | --- |
| Excludes 0 | Excludes 0 | Both find an effect | `"agree"` |
| Contains 0 | Contains 0 | Neither excludes zero | `"agree"` |
| Excludes 0 | Contains 0 | Different conclusions | `"disagree-trust-bootstrap"` |
| Contains 0 | Excludes 0 | Different conclusions | `"disagree-trust-bootstrap"` |
| Invalid or unavailable bootstrap | — | Cannot compare | `"bootstrap-unavailable"` |

  `"agree"` therefore asserts only that the two methods concur, in either direction. The label encodes the tie-break rule: on disagreement the bootstrap wins, and the conjunction verdict (§7.9) uses it.

### 7.5 Confirmatory secondary — the complexity trend

$$ d_{ij} = \beta_0 + \beta_1\,(\text{dl\_length}_i - \overline{\text{dl\_length}}) + \varepsilon_{ij} $$

- **Trend predictor** — `dl_length`, always centered, hence named `dl_length_centered` in the artifact. Centering is required, not cosmetic: uncentered, \(\beta_0\) is the effect at length zero, which does not exist. `predictor_mean` records the centering constant.
- **Complexity trend** (\(\beta_1\)) — change in the embedding effect per additional DL token. \(\beta_1 > 0\): embeddings help more on complex concepts. \(\beta_1 < 0\): more on simple ones. \(\approx 0\): flat.
- **Contiguous predictor** — `dl_length` is never binned. Binning discards information, costs power, makes the boundaries an arbitrary forking path, and reintroduces the multiplicity the trend test exists to remove.
- **Cluster-robust standard error** — the trend is fit by OLS with a CR1 covariance clustered on seed, with `df = n_clusters - 1`. A pragmatic stand-in for the full crossed model with a slope. Under a balanced design the point estimate is identical to the mixed-model fixed effect, so `trend.beta_0` equals `primary.beta_0` by construction and is reported for completeness rather than as an independent quantity.
- **Trend covariate** — `extension_ratio`, added in a second fit, addressing a real confound: longer concepts tend to have smaller extensions, and small extensions destabilize F1. Reported as `covariate_adjusted_beta_1`, `covariate_adjusted_ci95`, `covariate_adjusted_p`.
- **Survives covariate adjustment** (`survives_covariate_adjustment`) — `true` when the adjusted interval still excludes zero **and** the adjusted slope keeps the sign of the unadjusted one. Both conditions, not just the p-value.
- **Unidentified slope** — when `dl_length` is constant across paired problems, the slope does not exist; the result carries `NaN` and a `note` rather than a number.

### 7.6 Robustness layer

The nonparametric agreement check, in two steps:

- **Collapse over seeds** — average each problem's paired difference across its seeds, yielding one value per problem, as a mapping from `id` to difference. Reduces per-problem noise by roughly \(\sqrt{n_{\text{seeds}}}\) and makes across-problem independence defensible. Never "pool".
- **Wilcoxon signed-rank with Pratt zeros** — `zero_method` is fixed at `"pratt"`, not configurable. Ties at exactly zero are common because NCES frequently synthesizes the identical hypothesis under both conditions. Wilcoxon's original rule discards zeros before ranking and thereby inflates the remaining ranks; Pratt ranks the zeros, then drops them from the sum.
- **Sign-flip null** (`null`) — the p-value comes from flipping the signs of the observed differences, not from the asymptotic normal approximation, which is unreliable with a large zero mass and few nonzero observations. Enumerated exactly at \(n \le 20\) (`"exact-signflip"`), Monte Carlo above (`"monte-carlo-signflip"`).
- **Hodges–Lehmann estimator** (`hodges_lehmann`) — the median of pairwise Walsh averages; the effect size matching the Wilcoxon test. **Not** the median of the differences. Reported alongside `mean_difference`, because a significant Wilcoxon with a near-zero mean is possible when the zero mass is large and the nonzero tails are asymmetric.
- **Sign test** (`sign_test_p`) — an assumption-light binomial check on the nonzero differences.
- **Win / loss / tie triple** — the immediately interpretable summary, e.g. "dice wins 27, loses 32, ties 0".
- **Zero fraction** (`n_zero / n_total`) — a **result**, not a diagnostic: the fraction of problems where the embedding condition changed nothing about the score. When large, "embeddings rarely alter NCES's output, but when they do it is usually an improvement" is a sharper finding than any p-value.
- **Identical-hypothesis fraction** (`identical_hypothesis_fraction`) — the fraction of paired hypotheses whose strings are byte-identical. The stronger sibling of the zero fraction: identical scores can arise from different expressions, identical strings cannot.
- **Degenerate contrast** — every paired difference exactly zero. Reported with `null: "degenerate"` and \(p = 1\); no test is meaningful.

> `robustness.mean_difference` is the unweighted mean of collapsed per-problem differences, whereas `primary.beta_0` is a GLS estimate under \(V\). The two coincide under a balanced design with no missing cells, but the equality is **not guaranteed**; divergence is expected behaviour, not a defect.

### 7.7 Mechanism layer

Descriptive characterization of *how* embeddings alter hypotheses. It exists to interpret the primary result — for instance to test the "broader hypotheses" reading of a recall-heavy, precision-flat pattern — not to generate additional claims. It carries no p-value by design.

- **Mechanism summary** — per outcome (`precision`, `recall`, `hypothesis_extension_size`): `mean_dice`, `mean_random`, `mean_difference`, `hodges_lehmann`, and the win/loss/tie triple.
- **Extension size summary** (`extension_sizes`) — \(\lvert P \rvert\) against \(\lvert T \rvert\), the direct test of the breadth reading. Two ratios are reported because extension sizes are skewed and they answer different questions:
  - `dice_over_target_ratio` / `random_over_target_ratio` — ratio of means; dominated by the largest target extensions.
  - `mean_per_problem_dice_ratio` / `mean_per_problem_random_ratio` — mean of per-problem ratios; weights every problem equally.
  - `n_problems_with_target_size` — problems for which a target extension size was available, and hence the denominator of the per-problem ratios. Less than `n_problems` when the hardness annotation stage could not supply a size.
- **Empty-hypothesis rate** (`empty_hypothesis_rate_dice` / `_random`) — how often \(\lvert P \rvert = 0\). A clean failure mode; a reduction under `dice` is a result.
- **Classification summary** (`classification`) — a list with one entry per `condition`, each carrying `n_observations`, `mean_mcc`, `pooled_mcc`, `pooled_matrix`, `mean_accuracy`, and `degenerate_count`.
  - `n_observations` counts **atomic observations**, i.e. problems × seeds — not collapsed problems. It is therefore larger than the `n_problems` reported elsewhere in the same knowledge base by a factor of the seed count.
  - `pooled_matrix` — the summed confusion matrix, with keys `true_positive`, `false_positive`, `false_negative`, `true_negative`.
  - `degenerate_count` — observations whose confusion matrix had a zero marginal, and whose MCC was therefore defined as `0.0` (§6).

### 7.8 Exploratory grid and multiplicity

- **Exploratory grid** — the cross product of exploratory outcomes × bucketings × buckets, plus one unbucketed cell per outcome. Each cell is one **exploratory finding**. The unbucketed cell is encoded with both `bucketing` and `bucket` set to `null`.
- **Bucketing** — a complexity field used to partition observations: `depth`, `expressivity`, or `extension_ratio`. The `extension_ratio` bucketing is the only legitimate binning in the evaluation, because there it is a degeneracy indicator rather than a contiguous predictor being tested for trend. Its five edges are fixed: \([0.00,0.05)\), \([0.05,0.25)\), \([0.25,0.75)\), \([0.75,0.95)\), \([0.95,1.00]\).
- **Minimum cell size** — a bucket with fewer than five collapsed problems is skipped rather than reported with a meaningless interval. Skipped buckets are logged and simply absent from `findings`; their absence is not recorded in the artifact, so a bucketing will often contribute fewer than its full complement of cells.
- **Benjamini–Hochberg screening** — step-up FDR control at \(q = 0.10\) over the whole exploratory family. FDR rather than family-wise error, because family-wise control over a grid this size is self-defeating when the goal is screening.
- **Discovery** (`discovery`) — an exploratory finding whose `p_adjusted` is at or below \(q\). A candidate for follow-up, never a claim.

Adjustment policy, in full:

| Comparison set | Adjustment |
| --- | --- |
| Primary \(\beta_0\), across knowledge bases | **None.** The claim is a conjunction, not a disjunction; requiring simultaneous rejection everywhere is already conservative, and correcting a conjunction makes it needlessly weaker. |
| Confirmatory \(\beta_1\) | **None.** One pre-specified coefficient per knowledge base. |
| Exploratory grid | Benjamini–Hochberg at \(q = 0.10\). |

The policy is restated verbatim in the artifact's suite-level `multiplicity` block, whose three keys — `primary`, `trend`, `exploratory` — hold the human-readable justification for each row above.

> Holm appears nowhere in the confirmatory path. That is the point of the hierarchy: replacing \(k\) per-bucket tests with one trend coefficient **removes** the multiplicity rather than correcting for it.

### 7.9 Verdicts and artifacts

- **Evaluation result** — the inferential result for one knowledge base: `primary`, `trend`, `robustness`, `mechanism`, `extension_sizes`, `exploratory`, `classification`, `design`, `outcome_unavailable`, and `notes`.
- **Suite evaluation** — evaluation results across every knowledge base (`knowledge_bases`), plus `multiplicity`, `conjunction_holds`, and `conjunction_statement`.
- **Design block** (`design`) — the serialized `PairedDesign` metadata:

| Field | Contents |
| --- | --- |
| `knowledge_base` | The knowledge base's name. |
| `n_seeds`, `seeds` | Count and explicit list of seeds surviving pairing. |
| `n_problems` | Problems surviving pairing. |
| `n_observations` | Atomic observations, i.e. paired (problem, seed) pairs. |
| `unpaired_problem_ids` | Problems present under one condition only; excluded from every estimate. |
| `error_counts` | Per-condition count of problems that failed under NCES. |

- **Conjunction claim** — the confirmatory claim is "dice beats random on **every** knowledge base". It is a conjunction, which is why no correction is applied across knowledge bases.
- **Conjunction verdict** (`conjunction_holds`) — `true` only when every knowledge base's chosen interval lies strictly above zero. The chosen interval is the bootstrap one when `agreement == "disagree-trust-bootstrap"`, otherwise the model-based one. A knowledge base with no estimable primary counts as a **failure**, not a skip.
- **Conjunction statement** (`conjunction_statement`) — the human-readable form, e.g. "dice > random on 3/4 knowledge bases (95% interval excluding zero). Conjunction does not hold." All results are reported regardless of outcome.
- **Outcome unavailable** (`outcome_unavailable`) — a list of outcome names the design could not supply at all. The canonical case: `lift` requires `atomic_baseline_f1` from the hardness annotation stage, and without it the primary does not run; the evaluation then reports the robustness layer in its place and records the substitution in `notes`.
- **Notes** (`notes`, and per-block `note`) — free text recording every layer that could not be estimated. Each layer is guarded independently: a failed layer is noted and the remaining layers still run, consistent with the suite's design to finish and report rather than abort.
- **Evaluation artifact** — the JSON emitted by `write_evaluation`, sitting next to the descriptive summaries so the analysis is auditable and the family of tests is explicit.
- **Paired-observations artifact** — the optional second file holding every paired observation and every paired-hypothesis record. Worth persisting separately: it is the evaluation's input, and it enables per-concept inspection without re-running anything.

### 7.10 Failure semantics

The stage distinguishes three kinds of shortfall, and the vocabulary keeps them apart:

| Situation | Handling | Recorded as |
| --- | --- | --- |
| No runs at all, or no seed carrying both conditions | `InferenceError` — the paired design cannot be assembled | exception |
| A layer cannot be estimated (unidentified slope, too few observations, optimizer failure) | The layer is skipped; the rest still runs | `notes`, `note`, `outcome_unavailable` |
| A single problem failed under NCES | Excluded from every aggregate | `design.error_counts` |

> **`InferenceError`** is raised only when the paired design itself is impossible. Everything downstream degrades and reports.

### 7.11 Evaluation naming rules

Extends §9.

| Write this | Not this | Why |
| --- | --- | --- |
| evaluation stage | analysis, statistics stage | Names the pipeline stage. |
| inferential evaluation | evaluation | When NCES evaluation could be meant. |
| paired difference | delta, gap, improvement, gain | The response variable. |
| atomic observation | sample, data point | The unit of analysis. |
| paired design | dataset, matrix | The assembled input. |
| primary outcome | main metric | Exactly one: `lift`. |
| confirmatory / exploratory | significant / not significant | Names the layer, not the result. |
| mean embedding effect | effect, difference | \(\beta_0\), fully qualified. |
| complexity trend | slope, interaction | \(\beta_1\) on centered `dl_length`. |
| seed / problem random intercept | random effect | Always say which. |
| variance decomposition | variance diagnostics | It is a result. |
| collapse over seeds | pool, aggregate | The specific averaging step. |
| zero fraction | tie rate, no-op rate | It is a result. |
| identical-hypothesis fraction | same-output rate | The stronger sibling of the zero fraction. |
| discovery | finding, significant result | A BH-screened candidate, never a claim. |
| conjunction verdict | overall result, pass/fail | The claim is a conjunction. |
| cluster bootstrap | bootstrap | The seed is the resampling unit. |
| Pratt zeros | zero handling | The specific rule, and it is fixed. |
| Hodges–Lehmann estimator | median difference | It is not the median of the differences. |
| mean MCC / pooled MCC | MCC | They differ and the difference is informative. |

---

## 8. Output layout

> This is the structure at runtime. However, at the end of every benchmark run the corresponding seed
> directory is compressed into a zip archive. At the end of the benchmark suite the whole benchmark
> directory is compressed into a zip archive.

```text
Output/
└── <benchmark_name>/                      e.g. benchmark1
    ├── benchmark_summary.json             descriptive aggregate, all knowledge bases
    ├── suite_evaluation.json              the suite evaluation (§7.9)
    ├── paired_observations     dice vs random direct paired differences
    └── <knowledge_base>/
        ├── embeddings_data/               RDF-triple splits
        ├── nces_data/                     learning problems, splits, data-settings hash
        ├── ontology_parse_data/           triples and all individuals
        └── seed<N>/
            ├── embeddings/
            │   ├── <Model>.csv            trained DICE entity embeddings
            │   ├── <Model>_random.csv     random embedding baseline
            │   ├── best_report.json       dice metrics of the best recorded trial
            │   ├── embedding_report.json  trial metrics and triple counts
            │   ├── data/                  train.txt, valid.txt, test.txt
            │   └── trial_NN_<Model>/      per-trial DICE run directory
            ├── nces/
            │   ├── nces_report.json       the benchmark run report
            │   ├── results/               per-problem results for this run
            │   └── trained_models/
            │       ├── dice/              weights, dice condition
            │       └── random/            weights, random condition
            └── logs/
                └── <knowledge_base>_<seed>.log
```

- `embeddings/data/` holds the RDF-triple split for DICE.
- `nces_data/` holds `learning_problems.json`, the two `*_problems.json` split files, and `nces_train_data.json`.
- `trained_models/<condition>/` is separated per condition so the two conditions never share weights.

---

## 9. Naming rules

Apply these in code, identifiers, log messages, JSON keys, and prose. §7.11 extends this table for the evaluation stage.

| Write this | Not this | Why |
| --- | --- | --- |
| learning problem | task, instance, benchmark instance, example set | One canonical unit of work. |
| target concept | target class, ground truth | Reserve "class" for named ontology entities. |
| knowledge base | ontology, dataset | "Dataset" means a storage path or generic collection. |
| hypothesis | generated concept, prediction, output | The learner's proposed expression. |
| entity embedding | embedding | Precision, when the referent could be ambiguous. |
| embedding model | — | The generic notion. |
| DICE model | dicee model | The specific architecture. |
| benchmark run | experiment, NCES run | One (knowledge base, seed) execution. |
| benchmark summary | results file | Descriptive aggregate across runs. |
| configuration | config dict, settings blob | The resolved settings. |
| artifact | result file, output file | Any generated file. |
| report | result file | The structured per-run JSON. |
| seed | — | Never a synonym for run or dataset. |
| complexity | difficulty, level | The multi-dimensional object, never a scalar. |
| DL length | complexity, length, concept length | The scalar token count; one field within complexity. |
| hardness | semantic difficulty, hardness score | The reasoner-derived fields, collectively. Not a single number. |
| nesting depth | depth | `depth` alone is the LPGen search parameter. |
| lift | improvement, delta, gain | F1 over atomic baseline F1. |
| atomic baseline F1 | baseline, trivial score | Fully qualified; the project has other baselines. |
| target / hypothesis extension | extension | Always qualified. |
| number of learning problems | problem_count | Clearer in CLI help. |
| property / role | relation | Ontology terminology. |
| output directory | output dir | Full words in prose. |

---

## 10. Parameters requiring project-specific explanation

Values live in `input/*.json`; this section defines only names whose meaning is not self-evident or that differ from upstream. For learning-problem generation, see §5.

### NCES

Parameters fixed by the project rather than configured:

| Parameter | Value | Rationale |
| --- | --- | --- |
| `auto_train` | `False` | Training must be explicit and driven by the project's own train split. |
| `load_pretrained` | `False`, then `True` | Train from scratch, then reload the saved weights for evaluation. |
| `sorted_examples` | `True` | Stable example ordering. |
| `path_of_embeddings` | condition-specific CSV | The only parameter that differs between conditions. |
| `path_of_trained_models` | `trained_models/<condition>/` | Keeps the conditions' weights separate. |

Configured parameters needing a note:

- **`learner_name`** — one of `LSTM`, `GRU`, `SetTransformer`. Passed upstream as the single-element list `learner_names`.
- **`embedding_dim`** — the entity-embedding width NCES expects; **must match the embedding CSV** produced by the embedding stage.
- **`num_seeds`** — the learner's *internal* seeds. Unrelated to the benchmark seed (§3).

### DICE

- **`hpo_backend`** — selects the hyperparameter search: `smac` (`dice_smac.py`) or grid search (`dice_grid_search.py`).
- **`random_seed`** — the benchmark seed, passed through per run.
- **`scoring_technique`** — the DICE scoring mode; `KvsAll` evaluates a query against all entities.
- **Hyperparameter grid** — used only when `hpo_backend` is not `smac`. `search_best_embedding_setting` evaluates the cross product of {base dimension, doubled dimension} × {base batch size, halved batch size}, giving four trials. The winner is chosen by the selection metric (§3).

### CLI flags

Every flag overrides the corresponding settings-file field.

| Flag | Overrides |
| --- | --- |
| `--input-dir` | Location of the settings files. |
| `--output-dir` | Root of the output tree. |
| `--benchmark-name` | `project.benchmark_name` |
| `--datasets` | `data_generation.kbs` |
| `--seeds` | `project.seeds` |
| `--embedding-conditions` | `project.embedding_conditions` |
| `--embedding-model` | `embedding.model_name` |
| `--dice-epochs` | `embedding.epochs` |
| `--nces-epochs` | `nces.epochs` |
| `--hpo-backend` | `embedding.hpo_backend` |
| `--log-level` | Logging verbosity. |

---

## 11. Schema versions

| Version | Change |
| --- | --- |
| v1 | `complexity` is an integer, the DL-expression length. |
| v2 | `complexity` becomes an object; the v1 integer survives as `complexity.dl_length`. |
| v3 | No backwards compatibility for `complexity`; it is a pure multi-dimensional object. |
| v4 | Adds the inferential evaluation artifact (§7). |

---

## 12. Alphabetical index

| Term | Section |
| --- | --- |
| ABox | §2 |
| Agreement | §7.4 |
| Artifact | §4 |
| Atomic baseline F1 | §2 |
| Atomic concept | §2 |
| Atomic observation | §7.1 |
| Benchmark configuration | §3 |
| Benchmark run | §3 |
| Benchmark suite | §3 |
| Benjamini–Hochberg screening | §7.8 |
| Bucketing | §7.8 |
| Checkpoint | §4 |
| Class | §2 |
| Classification summary | §7.7 |
| Cluster bootstrap interval | §7.4 |
| Cluster-robust standard error | §7.5 |
| Collapse over seeds | §7.6 |
| Complexity | §2 |
| Complexity trend | §7.5 |
| Composite concept | §2 |
| Concept expression | §2 |
| Confirmatory | §7.2 |
| Confusion matrix | §6 |
| Conjunction claim | §7.9 |
| Conjunction statement | §7.9 |
| Conjunction verdict | §7.9 |
| Constructor profile | §2 |
| Contiguous predictor | §7.5 |
| Crossed | §7.3 |
| Degenerate contrast | §7.6 |
| Denominator degrees of freedom | §7.3 |
| Descriptive | §7.2 |
| Design block | §7.9 |
| DICE model | §3 |
| Discovery | §7.8 |
| DL length | §2 |
| Embedding condition | §3 |
| Embedding model | §3 |
| Empty-hypothesis rate | §7.7 |
| Entity embedding | §3 |
| Entity index mapping | §3 |
| Evaluation artifact | §7.9 |
| Evaluation result | §7.9 |
| Exploratory | §7.2 |
| Exploratory grid | §7.8 |
| Expressivity class | §2 |
| Extension | §2 |
| Extension ratio | §2 |
| Extension size summary | §7.7 |
| F1 | §6 |
| Hardness | §2 |
| Hardness annotation | §4 |
| Hodges–Lehmann estimator | §7.6 |
| Hits@10 | §6 |
| Hypothesis | §2 |
| Hyperparameter trial | §3 |
| Identical-hypothesis fraction | §7.6 |
| Individual | §2 |
| `InferenceError` | §7.10 |
| Inferential evaluation | §7 |
| IRI | §2 |
| Jaccard | §6 |
| Join key | §7.1 |
| Knowledge base | §2 |
| Knowledge graph | §2 |
| KvsAll | §6 |
| Lazy import | §4 |
| Learning problem | §2 |
| Lift | §2, §6 |
| Local name | §2 |
| Marginal covariance | §7.3 |
| Matthews correlation coefficient | §6 |
| Mean embedding effect | §7.3 |
| Mean MCC | §6 |
| Mechanism summary | §7.7 |
| Minimum cell size | §7.8 |
| Model-based interval | §7.4 |
| MRR | §6 |
| NCES model | §3 |
| Negative examples | §2 |
| Nesting depth | §2 |
| Non-degenerate problem | §2 |
| Notes | §7.9 |
| Outcome unavailable | §7.9 |
| Paired design | §7.1 |
| Paired difference | §7.1 |
| Paired hypotheses | §7.1 |
| Paired observation | §7.1 |
| Paired-observations artifact | §7.9 |
| Pairing | §7.1 |
| Pipeline stages | §4 |
| Pooled MCC | §6 |
| Positive examples | §2 |
| Postprocessing | §4 |
| Precision | §6 |
| Pratt zeros | §7.6 |
| Primary outcome | §7.2 |
| Problem random intercept | §7.3 |
| Property | §2 |
| RDF triple | §2 |
| Recall | §6 |
| Reduced fit | §7.3 |
| Redundant target concept | §2 |
| REML | §7.3 |
| Report | §4 |
| `rho` | §5 |
| Robustness layer | §7.6 |
| Role | §2 |
| Seed | §3 |
| Seed random intercept | §7.3 |
| Selection metric | §3 |
| Semantic equivalence | §2, §6 |
| Settings file | §3 |
| Sign test | §7.6 |
| Sign-flip null | §7.6 |
| Split ratio DICE | §3 |
| Split ratio learning problems | §3 |
| Splitting | §5 |
| Summary | §4 |
| Suite evaluation | §7.9 |
| Survives covariate adjustment | §7.5 |
| Target concept | §2 |
| TBox | §2 |
| Trend covariate | §7.5 |
| Trend predictor | §7.5 |
| Unidentified slope | §7.5 |
| Unpaired problem | §7.1 |
| Variance decomposition | §7.3 |
| Wilcoxon signed-rank | §7.6 |
| Win / loss / tie triple | §7.6 |
| Zero fraction | §7.6 |

---

## Summary of changes

### Deleted

- §11 Quick reference — the source of the stale `complexity` = "DL-expression length" definition, and a duplicate of §2–§3 throughout.
- The duplicated `Hardness` entry (the malformed "**Hardness — the semantic**" heading).
- Duplicate definitions of `Lift` (three occurrences reduced to one canonical in §2, cross-referenced from §6).
- §9's value columns for NCES and DICE parameters; kept only names needing project-specific explanation. Old §9 is now §10.
- Generic coding terms (`parameter`/`argument`/`flag`, `metric`/`score`) — ordinary Python and English usage carrying no project-specific meaning. The `parameter`/`artifact`/`report` rows survive in §9's naming table where they express a convention.
- Editorializing prose: "The last two rows are the ones to watch", "This matters:", "the whole point of this section", the "Everything else is shared" paragraph, and the Nelder–Mead-from-four-starts implementation detail.
- The self-flagging stale note on result fields, resolved by the artifact.

### Corrected

- §5 splitting no longer says "stratified by a given value of `complexity`", which violated §9's own rule; now "a configurable complexity field — by default `dl_length`".
- Jaccard's dependence on F1 now appears where Jaccard is *defined* (§6), not only where it is demoted (§7.2).
- Confusion-matrix reconstruction and MCC moved from §7.7 (mechanism) to §6 (metrics), where they belong; §7.7 now references them.
- MCC's dual placement — exploratory *outcome* vs mechanism *summary* — stated in §7.2 where layers are assigned.
- `agreement` now documents all four rows of your chart, and clarifies that `"agree"` asserts concurrence in either direction.
- `df_method` enumerated exhaustively: three values across two blocks.
- `zero_method` documented as fixed at `pratt`, not configurable.

### Added from the artifact

- `design` block field table; `bootstrap_resamples`; `predictor_mean`; `covariate_adjusted_*`; `n_problems_with_target_size`; `degenerate_count`; `pooled_matrix` keys; the `bucketing: null` / `bucket: null` encoding for unbucketed cells; the suite-level `multiplicity` block; `outcome_unavailable` as a list; mechanism key names `mean_dice` / `mean_random`.
- `classification.n_problems` renamed to `n_observations`, with an explicit warning that it counts problems × seeds and will exceed `n_problems` elsewhere in the same knowledge base.
- Note that `robustness.mean_difference` equalling `primary.beta_0` is **not guaranteed** — divergence is expected behaviour, not a defect.
- Note that `trend.beta_0` equals `primary.beta_0` **by construction** under balance, and is reported for completeness.
- Minimum-cell-size behaviour: skipped buckets are logged, absent from `findings`, not otherwise recorded.
- Pipeline stage 9 (inferential evaluation), previously missing from the stage list.
- §12 alphabetical index.
