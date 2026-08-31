# Project Glossary

This document is the canonical vocabulary for the NCES benchmark project. It
exists so that a term means exactly one thing in code, log output, JSON
artifacts, CLI help text, and prose.

If you are looking for how to *run* the project, see `README.md`. This file
answers "what is this thing called, and why" — not "how do I use it".

---

## 1. What the project measures

The benchmark answers one question:

> **How much does the quality of entity embeddings affect a neural concept
> learner's ability to recover a target concept from examples?**

To answer it, the project holds everything constant except the embeddings fed
to NCES, and compares two **embedding conditions**:

| Condition | Embeddings supplied to NCES |
| --- | --- |
| `dice` | Entity embeddings trained on the knowledge graph by a DICE model, selected from a hyperparameter search. |
| `random` | Deterministic uniform random vectors using the identical CSV schema and dimensionality. |

Everything else — knowledge base, learning problems, splits, NCES
architecture, epochs, seed — is shared between the two conditions. Any
difference in the resulting metrics is therefore attributable to the
embeddings alone.

### The learning problem, formally

Given a knowledge base \( K \), a target concept \( T \), positive examples
\( E^+ \), and negative examples \( E^- \), the learner must find a concept
expression \( C \) such that, for \( K' = K \cup \{T \equiv C\} \):

$$ \forall e^+ \in E^+ : K' \models C(e^+) \qquad\text{and}\qquad \forall e^- \in E^- : K' \not\models C(e^-) $$

In words: the hypothesis must entail every positive example and no negative
example. The benchmark scores how close NCES gets to this ideal, measured over
the **extension** of the hypothesis rather than over the sampled examples
alone — see §6.

---

## 2. Core domain terminology

These terms come from Description Logics and ontology learning. They describe
the *subject matter*, independent of this project's implementation.

### Ontology structure

- **Knowledge base** — the ontology being learned from: an OWL file and its
  benchmark context. This is the preferred term; do not call it a "dataset"
  when you mean the ontology itself.
- **Knowledge graph** — the graph view of a knowledge base, as entities and
  RDF triples. This is the form DICE trains on.
- **RDF triple** — a `(subject, predicate, object)` statement extracted from
  the knowledge base. Only IRI-to-IRI triples are used; literals are dropped
  because they cannot be embedded as entities.
- **TBox** — terminological knowledge: the class hierarchy and concept
  definitions.
- **ABox** — assertional knowledge: individuals, their types, and their
  property assertions.
- **Individual / instance** — a concrete named entity, such as a specific
  person or animal.
- **Class / atomic concept** — a named ontology class such as `Mammal`, with
  no further composition.
- **Property / role** — a predicate connecting individuals, such as
  `hasChild` or `partOf`.
- **IRI** — an Internationalized Resource Identifier naming an ontology
  entity.
- **Local name** — the final fragment of an IRI after the last `#` or `/`.
  NCES indexes its embedding matrix by local name, so every IRI crossing into
  NCES is reduced first.

### Concepts and expressions

- **Concept expression** — a DL expression built from classes, roles, and
  constructors. Rendered to DL syntax (`⊓`, `⊔`, `¬`, `∃`, `∀`) for storage
  and reporting.
- **Composite concept** — a concept expression combining multiple classes or
  roles: intersection, union, complement, or a restriction.
- **Target concept** — the concept the learner is expected to recover. Always
  "target concept", never "target class".
- **Hypothesis** — the concept expression NCES proposes as its answer. Always
  "hypothesis", never "generated concept" or "prediction".
- **Extension** — the set of individuals satisfying an expression under the
  ontology semantics, computed by the reasoner. Qualify it: *target
  extension* or *hypothesis extension*.
- **Semantic equivalence** — the target extension and hypothesis extension
  are exactly equal. This is the strictest success criterion the benchmark
  reports; a hypothesis can score high F1 without being semantically
  equivalent.
- **Complexity** — the multi-dimensional characterisation of a target
  concept's structure and semantic difficulty.Serialized as an object,
  never a bare integer. Its fields divide into structural measures,
  computed from the expression alone, and hardness measures, which require
  the reasoner. Never "difficulty" or "level".
- **DL length** — the token count of a rendered DL expression: atomic
  concepts and constructors each score 1. One structural field within
  complexity. This was the entirety of complexity before v2; see §11.
- **Hardness** — The reasoner-derived, extension-dependent fields: extension
  ratio, atomic baseline, redundancy. A subset of complexity's fields.
- **Nesting depth** — the maximum quantifier nesting of a concept expression.
  A ⊓ B has depth 0; ∃ r.A has depth 1; ∃ r.(A ⊓ ∃ s.B) has depth 2.
- **Constructor profile** — the multiset of DL constructors occurring in an
  expression, as a mapping from constructor to occurrence count.
- **Expressivity class** — the smallest DL fragment containing an expression:
  EL, ALC, or ALCHIQD. Determined by which constructors are present,
  independent of the beyond_alc setting that permitted them.
- **Hardness — the semantic**, reasoner-derived component of complexity:
  extension ratio, atomic baseline F1, and redundancy. Distinguished from
  structural complexity because it depends on the knowledge base, not only
  the expression.
- **Extension ratio** — \(\lvert T \rvert / \lvert U \rvert\), the fraction
  of all individuals in the target extension. Values near 0 or 1 indicate
  a degenerate class balance.
- **Atomic baseline F1** — the best F1 achievable by any single atomic class,
  computed over extensions. The floor a hypothesis must clear to demonstrate
  non-trivial learning.
- **Lift** — a hypothesis's F1 minus the atomic baseline F1 of its learning
  problem. May be negative. The headline metric for non-trivial learning.
- **Redundant target concept** — a target concept whose extension equals that
  of some atomic class, making it learnable far below its structural complexity.

### Learning problems

- **Learning problem** — the canonical unit of work: one target concept plus
  its positive and negative example individuals. Never "task", "instance", or
  "benchmark instance".
- **Positive examples** (`pos_example`) — individuals belonging to the target
  concept.
- **Negative examples** (`neg_example`) — individuals not belonging to it.
- **Non-degenerate problem** — a learning problem with at least one positive
  and one negative example. Degenerate problems are rejected at construction
  time, since a learner cannot be meaningfully scored on them.

---

## 3. Project components

The vocabulary for this project's own moving parts.

### Models

- **Embedding model** — the generic term for the entity-embedding model
  trained on the knowledge graph.
- **DICE model** — the specific architecture, provided by the `dicee` library
  and configured in `src/models/dice.py`. Available names: `Decal`, `Keci`,
  `DualE`, `ComplEx`, `QMult`, `OMult`, `ConvQ`, `ConvO`, `ConEx`, `TransE`,
  `DistMult`, `Shallom`.
- **Entity embedding** — the vector representation of a single ontology
  entity. Prefer this over the bare word "embedding" when precision matters.
- **NCES model** — the Neural Class Expression Synthesizer from Ontolearn; the
  concept learner under evaluation. Its recurrent/attention learner is chosen
  from `LSTM`, `GRU`, or `SetTransformer`.
- **Entity index mapping** — the mapping from an entity's identifier to its
  row in the embedding matrix.

**Naming note.** The third-party library is `dicee` (lowercase, as
installed from PyPI). This project's module wrapping it is
`src/models/dice.py`. Referring to the library, write `dicee`; referring to
the module or the model family, write DICE.

### Execution units

- **Benchmark run** — one execution of the full workflow for one
  (knowledge base, seed) pair. This is the atomic unit of the benchmark and
  produces exactly one report.
- **Benchmark suite** — all benchmark runs in a single invocation, across
  every selected knowledge base and seed.
- **Embedding condition** — `dice` or `random`; see §1.
- **Hyperparameter trial** — one attempted DICE configuration within a
  benchmark run's embedding search.
- **Seed** — the random seed controlling triple splitting, learning-problem
  splitting, DICE training, and the random embedding baseline. A seed is the
  reproducibility handle; it is never a synonym for "run" or "dataset".

### Configuration

- **Benchmark configuration** — the complete resolved settings for one
  invocation, assembled from the four `input/*.json` files plus CLI
  overrides, and embedded verbatim in every report.
- **Settings file** — one of the four JSON files in `input/`. See §5.
- **Split ratio dice** — the train/validation/test proportions, `[0.8, 0.1, 0.1]` 
- **Split ratio learning problems** — train/test proportions `[0.8, 0.2]`.

---

## 4. Code and workflow vocabulary

### Modules

| Module | Responsibility |
| --- | --- |
| `src/config.py` | Typed settings dataclasses; translates project JSON field names into upstream keyword arguments. |
| `src/paths.py` | The canonical output directory layout and knowledge-base resolution. |
| `src/logging_utils.py` | Console and per-run file logging. |
| `src/data/complexity.py` | Complexity computation: structural measures from DL expressions, hardness measures from the reasoner.|
| `src/data/ontology.py` | OWL parsing, RDF-triple extraction, individual enumeration, extension computation. |
| `src/data/lp.py` | Learning-problem generation, the canonical schema, and splitting. |
| `src/data/results.py` | Contains several class definitions of results that are returned across the benchmark. |
| `src/models/dice.py` | DICE dataset preparation, embedding training entity-embedding export, random baseline. |
| `src/models/dice_smac.py` | Hyperparameter search via the SMAC3 framework. Uses a random forest surrogate. |
| `src/models/dice_grid_search.py` | Hyperparameter search via grid search. |
| `src/models/hpo_search_utils.py` | Helper class for hyperparameter search. |
| `src/models/nces.py` | NCES training-data preparation, training, and hypothesis evaluation. |
| `src/benchmarking/inference` The evaluation stage: paired design assembly, the crossed mixed model, the complexity trend, the nonparametric robustness layer, and the BH-screened exploratory grid. |
| `src/benchmarking/metrics.py` | Extension-based metric calculation and complexity aggregation. |
| `src/benchmarking/runner.py` | The **orchestrator**: coordinates every stage across knowledge bases, seeds, and conditions. |
| `src/__main__.py` | The `nces-benchmark` CLI entry point. |

### Pipeline stages

The workflow, in execution order. Each stage's output is the next stage's
input.

1. **Ontology parsing** — read the OWL file; extract RDF triples and
   enumerate individuals.
2. **Learning-problem generation** — produce target concepts with example
   sets (§5).
3. **Hardness annotation** — compute each target concept's extension, extension
    ratio, atomic baseline F1, and redundancy flag; populate the hardness
    fields of its complexity. Runs once per knowledge base, before splitting,
    and its extensions are cached for reuse during evaluation.
4. **Learning-problem splitting** — partition problems into disjoint
   train/test sets.
5. **Embedding stage** — write DICE triple splits, run the hyperparameter
   search, export the winning entity embeddings, and generate the random
   baseline.
6. **NCES training** — train the learner on the train split, once per
   embedding condition.
7. **NCES evaluation** — for each held-out problem, synthesize a hypothesis,
   compute its extension, and score it against the target extension.
8. **Result aggregation** — summarize per run, per knowledge base, and per
   suite.
> `Learning-problem generation`, `Hardness annotation`, and
> `Learning-problem splitting` are computed once per seed and reused in all
> consecutive seeds. `Learning-problem generation` and
> `Learning-problem splitting` are independent of the seeds and use their own
> which is derived from the hashed values of the corresponding
> `data_generation_settings.json`.

### Standard coding terms

Used in their ordinary Python meanings; listed only where this project has a
convention.

- **Parameter** — a named input accepted by a function or CLI option.
  **Argument** — the concrete value passed at runtime.
- **Flag** — a CLI switch such as `--datasets` or `--seeds`.
- **Lazy import** — importing a heavy dependency inside a function rather
  than at module level. Used throughout so that `nces-benchmark --help`
  does not load `torch`.
- **Artifact** — any generated file: JSON, CSV, log, or saved model state.
- **Report** — the structured JSON result for one benchmark run.
- **Summary** — an aggregate across multiple runs.
- **Checkpoint** — saved model weights reloaded for evaluation.
- **Postprocessing** — transformations after generation, chiefly the
  IRI-to-local-name reduction required by NCES.

---

## 5. Learning-problem generation

Learning problems are generated by
`generate_learning_problems` in **`src/data/lp.py`**, which wraps
`ontolearn.lp_generator.LPGen`.

`LPGen` performs a refinement-based search over the knowledge base, writes its
raw output to `LPs.json`, and the project then normalizes that output into the
canonical schema below — expanding local names back to full IRIs and rejecting
degenerate problems.


### Settings

Generation is controlled by `input/data_generation_settings.json`:

| Field | Meaning | Upstream name |
| --- | --- | --- |
| `kbs` | Comma-separated knowledge-base names. | — (project-level) |
| `num_rand_samples` | Learning problems to retain. | `max_num_lps` |
| `depth` | Refinement-search depth. | `depth` |
| `max_child_len` | Max length of a refinement child. | `max_child_length` |
| `refinement_expressivity` | Fraction of the refinement space explored → `ExpressRefinement.expressivity`. | `refinement_expressivity` |
| `beyond_alc` | `false` → ALC; `true` → ALCHIQ(D), enabling inverses, cardinality restrictions, and datatype constructors. | `beyond_alc` |
| `downsample_refinements` | → `ExpressRefinement.downsample`. **Must be `true` whenever `refinement_expressivity < 1.0`.** | `downsample_refinements` |
| `sample_fillers_count` | Fillers sampled per restriction. | `sample_fillers_count` |
| `num_sub_roots` | Sub-root concepts the generator expands from. | `num_sub_roots` |
| `min_num_pos_examples` | Minimum positive examples per target concept. | `min_num_pos_examples` |

The last two rows are the ones to watch. Upstream has no named-operator
parameter — it exposes a boolean `beyond_alc` — so `src/config.py` maps
`ALCHIQD → True` and `ALC → False`. Likewise `max_child_len` is spelled
`max_child_length` upstream and `num_rand_samples` is `max_num_lps`. These
translations live in `DataGenerationSettings.lpgen_kwargs()`.

On the name rho. In KB2Data.__init__, rho is the local variable holding the constructed ExpressRefinement operator — the conventional DL symbol \(\rho\) for a refinement operator.
It is not a settable parameter. Three fields shape it: beyond_alc toggles the five use_* constructor flags as a group,
refinement_expressivity becomes expressivity, and downsample_refinements becomes downsample.

### Learning-problem schema

Every generated learning problem is serialized as:

```json
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
```
> The four hardness fields are null immediately after generation and are populated by the hardness annotation stage,
> which runs once per knowledge base after ontology parsing. Structural fields are always present.

- `id` — stable identifier within one benchmark run.
- `target_concept` — DL-syntax string.
- `pos_example` / `neg_example` — sorted lists of **full IRIs**.
- `complexity` — Multi-dimensional component that aims to quantify how difficult a problem is.
- `num_pos` / `num_neg` — example counts, derived automatically.

The full set is persisted **grouped by complexity level** in
`learning_problems.json`; the splits are written as flat lists.

### Splitting

Learning problems are split at the **problem** level, not the example level.
Splitting a single problem's examples across train and test would leak the
target concept into evaluation, and the resulting scores would be
meaningless. The splits are disjoint by construction and deterministic in the
seed.
Further, the splits are stratified by a given value of `complexity`. Per
default this value is `dl_length`.
---

## 6. Metrics and result fields

### How scoring works

For each held-out learning problem, NCES produces a hypothesis. Both the
hypothesis and the target concept are parsed back into OWL class expressions
and handed to the reasoner, which returns their extensions. The metrics
compare those two sets.

This matters: scoring against **extensions over all individuals**, rather than
against the sampled example sets, prevents a hypothesis that merely happens to
separate the handful of provided examples from scoring perfectly.

### Metric definitions

Let \( P \) be the hypothesis extension, \( T \) the target extension, and
\( U \) all individuals in the knowledge base.

| Metric | Definition |
| --- | --- |
| **Precision** | \( \lvert P \cap T \rvert / \lvert P \rvert \) — of the individuals the hypothesis selects, how many belong to the target. |
| **Recall** | \( \lvert P \cap T \rvert / \lvert T \rvert \) — of the target's individuals, how many the hypothesis recovers. |
| **F1** | Harmonic mean of precision and recall. |
| **Accuracy** | Fraction of all of \( U \) classified correctly, counting true negatives. |
| **Jaccard** | \( \lvert P \cap T \rvert / \lvert P \cup T \rvert \). |
| **Semantic equivalence** | \( P = T \) exactly. |

All ratios return `0.0` on an empty denominator rather than raising.

### Ranking metrics (DICE only)

- **MRR** — mean reciprocal rank. The **selection metric** for choosing the
  best hyperparameter trial: validation MRR, falling back to test MRR.
- **H@10 / Hits@10** — whether the correct entity appears in the top ten
  predictions. Reported but not used for selection.
- **KvsAll** — the DICE scoring mode that evaluates a query against all
  entities.

### Result fields

Per-problem fields in a report's `results` array:

> Note: The final result returned does not look like this anymore. While the
> fields and there content are still relevant this section needs to be updated.

| Field | Contents |
| --- | --- |
| `id`, `target_concept`, `complexity` | Copied from the learning problem. |
| `hypotheses` | The serialized NCES hypothesis, in DL syntax. |
| `num_pos` / `num_neg` | Counts of the *sampled* examples. |
| `target_positive_count` / `target_negative_count` | Counts over the *full* target extension — not the sampled examples. |
| `target_extension_size` | Object with `positive`, `negative`, `total`. |
| `accuracy`, `precision`, `recall`, `f1`, `jaccard`, `semantic_equivalence` | The metrics above. |
| `runtime_seconds` | Wall-clock time for this problem. |
| `error` | Present only if NCES raised; the problem is then excluded from aggregates. |
| `complexity`	| The full complexity object, copied from the learning problem.
| `lift` | f1 minus atomic_baseline_f1. Negative when the hypothesis underperforms the best atomic class.
| `mcc` | Split into mean MCC and pooled MCC. Divergence between them indicates the embedding effect is concentrated in
problems of a particular size.|

Run-level and embedding-level fields:

| Field | Contents |
| --- | --- |
| `complexity_summary` | Aggregates over multiple bucketings — by_dl_length, by_depth, by_expressivity, by_extension_ratio. Each bucket carries count, mean F1, mean accuracy, mean precision, mean recall, mean Jaccard, mean lift, and semantic-equivalence rate. |
| `best_embedding_config` | The selected DICE configuration with its score and metrics. |
| `search_trials` | Every attempted hyperparameter trial, including failures. |
| `validation_error` | Explanatory text when validation MRR was unavailable and a fallback was used. |

A failed trial or a failed learning problem is **recorded, not fatal**. The
benchmark is designed to complete and report partial results rather than
abort.

---

## 7. Evaluation and inference

Sections §6 and §7 describe two different things and the distinction is the
whole point of this section.
Metrics (§6) are descriptive: they score one hypothesis against one
target extension, per problem, per condition. Evaluation is
inferential: it takes those scores across every problem and seed and
produces an estimate, an interval, and a verdict about the embedding
condition. Metrics answer "how good was this hypothesis"; evaluation answers
"does dice beat random, and by how much".
Evaluation is implemented in src/benchmarking/inference.py and consumes
SingleRunResult objects — the same reports written under nces/. It reads
artifacts; it never re-runs the learner.

> Naming note. The stage is called evaluation, and the module is
> inference.py. "Evaluation" names the pipeline stage and its artifact;
> "inference" names the statistical machinery inside it. Do not call this
> stage "analysis", "statistics", or "significance testing", and do not
> confuse it with NCES evaluation (§4, stage 7), which is per-problem
> scoring. When ambiguity is possible, write inferential evaluation or
> NCES evaluation explicitly.


### 7.1 Unit of analysis

  - ``Atomic observation`` — one (learning problem, seed) pair for one
  outcome. This is the unit of analysis for the entire evaluation stage.
  Never "sample" or "data point".
  - ``Paired difference`` — the response variable, always
  \(d_{ij} = m^{\text{dice}}_{ij} - m^{\text{random}}_{ij}\) for problem
  \(i\), seed \(j\), and outcome \(m\). Differencing cancels the target
  concept, examples, split, knowledge base, NCES architecture, and seed
  shared by the two conditions, which is exactly the isolation the
  benchmark's design claims. Always "paired difference", never "delta",
  "gap", or "improvement".
- ``Paired observation`` — the serialized record of one atomic observation:
  both conditions' values plus the complexity fields needed to bucket or
  regress it. The PairedObservation dataclass.
- ``Paired hypotheses`` — both conditions' hypothesis strings for one
  (problem, seed) pair, stored side by side. The PairedHypotheses
  dataclass. Enables the zero-difference count and per-concept inspection
  without re-running anything.
- ``Paired design`` — every paired observation for one knowledge base, across
  all outcomes, plus the paired hypotheses, the seeds and problems that
  survived pairing, and the failure counts. The PairedDesign dataclass.
- ``Join key`` — the learning problem's id. Learning-problem generation and
  splitting are seed-independent (§5), so a problem's id identifies the same
  target concept in every seed, which is what makes the per-problem random
  intercept estimable.
- ``Pairing`` — matching a dice result to a random result by join key
  within one seed. A problem contributes an observation for a given outcome
  only when both conditions produced a usable value for it.
- ``Unpaired problem`` — a problem present in one condition but not the other.
  Recorded in unpaired_problem_ids; excluded from every estimate.

## 7.2 Outcome hierarchy

The set of outcomes is pre-specified — fixed before the numbers are seen.
This is what keeps the multiplicity problem small, and it is why the layer
names are part of the vocabulary rather than an implementation detail.

| Layer | Outcome(s) | Role |
| --- | --- | --- |
| **Primary outcome** | `lift` | Confirmatory. The single headline claim. |
| **Confirmatory secondary** | `lift` trend in `dl_length` | Confirmatory. Does the advantage grow with complexity? |
| **Mechanism outcomes** | `precision`, `recall`, `hypothesis_extension_size` | Descriptive. Never tested confirmatorily. |
| **Robustness outcome** | `lift`, nonparametrically | Agreement check against the primary. Not a second primary. |
| **Exploratory outcomes** | `mcc`, `accuracy`, `jaccard`, `semantic_equivalence`, crossed with the bucketings | Screening only. BH-screened, labeled exploratory. |

- ``Confirmatory`` — pre-specified, tested, and permitted to support a claim.
  Exactly two coefficients per knowledge base carry this status: \(\beta_0\)
  and \(\beta_1\).
- ``Descriptive`` — reported without a p-value, to interpret the confirmatory
  result. The mechanism layer deliberately carries no test.
- ``Exploratory`` — screened, not claimed. Every exploratory finding is
  labeled "role": "exploratory" in the artifact.
  Why lift and not f1 is primary: lift is ``F1`` minus ``atomic baseline F1``, so
  it is the only outcome that distinguishes learning from recovering a good
  atomic class. Why the others are demoted:
- ``Jaccard`` is not a separate outcome. \(J = F1 / (2 - F1)\) for a fixed
  pair of sets, so testing both tests one thing twice. It stays in the
  exploratory layer.
- ``Accuracy`` is not a headline. Scored over all of \(U\) including true
  negatives, it is dominated by true negatives whenever extension_ratio is
  small. Reported, never claimed on.
- ``Semantic equivalence`` is a rate, not a score. It is handled as a
  proportion via paired McNemar, not with Wilcoxon.

### 7.3 Primary analysis — the crossed mixed model

The primary estimate comes from an intercept-only crossed random-effects
model on paired differences, fit per knowledge base:
\(d_{ij} = \beta_0 + u_{\text{seed}(j)} + u_{\text{problem}(i)} + \varepsilon_{ij}\)

| Term | Name | Meaning |
| --- | --- | --- |
| \(\beta_0\) | **mean embedding effect** | The headline estimate. Always reported with an interval. |
| \(u_{\text{seed}(j)}\) | **seed random intercept** | Absorbs DICE/SMAC/NCES training variance shared by every problem in a run. Omitting it treats \(n_{\text{seeds}} \times \lvert\text{test}\rvert\) observations as independent and yields anticonservative p-values. |
| \(u_{\text{problem}(i)}\) | **problem random intercept** | Absorbs concept-level heterogeneity persisting across seeds. Estimable only because the join key is seed-invariant. |
| \(\varepsilon_{ij}\) | residual | — |

- ``Crossed`` — seeds and problems are crossed, not nested: every problem
  appears under every seed. Never "nested".
- ``Variance decomposition`` — the triple
  (var_seed, var_problem, var_residual). This is a result, not
  diagnostics: var_problem ≫ var_seed means the advantage varies more
  across concepts than across training runs, which tells a reader whether a
  single run is trustworthy.
- ``REML`` — restricted maximum likelihood, the fitting method.
  _profile_reml optimizes the profiled REML criterion over the two variance
  ratios \((\sigma^2_{\text{seed}}/\sigma^2_\varepsilon,\;
  \sigma^2_{\text{problem}}/\sigma^2_\varepsilon)\) by Nelder–Mead from four
  starts, then back-solves the generalized least-squares estimate.
- ``Marginal covariance`` — \(V = \sigma^2_{\text{seed}} Z_s Z_s^\top +
  \sigma^2_{\text{problem}} Z_p Z_p^\top + \sigma^2_\varepsilon I\). Dense
  but tiny at benchmark scale, so it is formed and factorized directly rather
  than exploiting sparsity.
- ``Denominator degrees of freedom`` — deliberately conservative:
  \(\min(n_{\text{seeds}} - 1,\, n_{\text{problems}} - 1)\), reported as
  df_method: "satterthwaite-conservative". With five seeds the seed
  variance is poorly estimated and the naive \(n - 1\) is far too optimistic.
- ``Reduced fit`` — when the design cannot identify two variance components
  (fewer than two seeds or two problems), the model degrades to a paired
  t-interval and says so via df_method: "naive-t" and a note. This is
  reported, not raised.

### 7.4 Interval agreement

Because lift differences are bounded and spike at zero, the model-based
interval is not trusted on its own.
- ``Model-based interval (ci95)`` — the t-interval from \(\beta_0\) and its
  standard error, on the conservative degrees of freedom.
- ``Cluster bootstrap interval (bootstrap_ci95)`` — a percentile bootstrap
  in which the seed is the resampling unit. Whole seeds are resampled with
  replacement, preserving the within-run dependence that the seed random
  intercept exists to absorb. Never resample individual observations.
- ``Agreement`` — whether the two intervals reach the same
  zero-exclusion verdict. One of agree, disagree-trust-bootstrap, or
  bootstrap-unavailable. The name encodes the rule: on disagreement, the
  bootstrap wins, and the conjunction verdict (§7.9) uses it.

### 7.5 Confirmatory secondary — the complexity trend
\(d_{ij} = \beta_0 + \beta_1\,(\text{dl\_length}_i - \overline{\text{dl\_length}}) + \varepsilon_{ij}\)

- ``Trend predictor`` — dl_length, always centered. Centering is
  required, not cosmetic: uncentered, \(\beta_0\) is the effect at length
  zero, which does not exist. The artifact names the predictor
  dl_length_centered for exactly this reason.

- ``Complexity trend`` (\(\beta_1\)) — change in the embedding effect per
  additional DL token. \(\beta_1 > 0\): embeddings help more on complex
  concepts. \(\beta_1 < 0\): more on simple ones. \(\approx 0\): flat.
- ``Contiguous predictor`` — dl_length is never binned. Binning discards
  information, costs power, makes the boundaries an arbitrary forking path,
  and reintroduces the very multiplicity the trend test exists to remove.
- ``Cluster-robust standard error`` — the trend is fit by OLS with a CR1
  cluster-robust covariance, clustered on seed. A pragmatic stand-in for the
  full crossed model with a slope: the point estimate coincides with the
  mixed-model fixed effect under balance, and clustering keeps the standard
  error honest about within-run dependence. df = n_clusters - 1.
- ``Trend covariate`` — extension_ratio, added in a second fit. This
  addresses a real confound: longer concepts tend to have smaller extensions,
  and small extensions destabilize F1.
- ``Survives covariate adjustment`` — true when the adjusted interval still
  excludes zero and the adjusted slope keeps the sign of the unadjusted
  one. Both conditions, not just the p-value.
- ``Unidentified slope`` — when dl_length is constant across paired
  problems, the slope does not exist; the result carries NaN and a note
  rather than a number.

### 7.6 Robustness layer

The nonparametric agreement check. Two steps, in order:
- ``Collapse over seeds`` — average each problem's paired difference across
  its seeds, yielding one value per problem. Reduces per-problem noise by
  roughly \(\sqrt{n_{\text{seeds}}}\) and makes the across-problem
  independence assumption defensible. Always "collapse over seeds", never
  "pool".
- ``Wilcoxon signed-rank with Pratt zeros`` — zero_method="pratt". Ties at
  exactly zero are common here because NCES frequently synthesizes the
  identical hypothesis under both conditions. Wilcoxon's original rule
  discards zeros before ranking and thereby inflates the remaining ranks;
  Pratt ranks the zeros, then drops them from the sum.
- ``Sign-flip null`` — the p-value comes from flipping the signs of the
  observed differences, not from the asymptotic normal approximation, which is
  unreliable with a large zero mass and few nonzero observations. Enumerated
  exactly at \(n \le 20\) (exact-signflip), Monte Carlo above
  (monte-carlo-signflip).
- ``Hodges–Lehmann estimator`` — the median of pairwise Walsh averages; the
  effect size that matches the Wilcoxon test. Reported alongside the mean
  difference, because a significant Wilcoxon with a near-zero mean is possible
  when the zero mass is large and the nonzero tails are asymmetric.
- ``Sign test`` — an assumption-light binomial check on the nonzero
  differences.

- ``Win / loss / tie triple`` — the immediately interpretable summary, e.g.
  "dice wins 84, loses 41, ties 120".
- ``Zero fraction (n_zero / n_total)`` — a result, not diagnostics. It
  is the fraction of problems where the embedding condition changed nothing
  about the score. When it is large, "embeddings rarely alter NCES's output,
  but when they do it is usually an improvement" is a sharper finding than any
  p-value.
- ``Identical-hypothesis fraction`` — the fraction of paired hypotheses whose
  strings are byte-identical. The stronger sibling of the zero fraction:
  identical scores can arise from different expressions, identical strings
  cannot.
- ``Degenerate contrast`` — every paired difference exactly zero. Reported
  with null: "degenerate" and \(p = 1\); no test is meaningful.

### 7.7 Mechanism layer

Descriptive characterization of how embeddings alter hypotheses. It exists
to interpret the primary result — for instance to test the "broader
hypotheses" reading of a recall-heavy, precision-flat pattern — not to
generate additional claims. It deliberately carries no p-value.

- ``Mechanism summary`` — per outcome: mean under each condition, mean paired
  difference, Hodges–Lehmann estimate, and the win/loss/tie triple.
- ``Extension size summary`` — \(\lvert P \rvert\) against \(\lvert T \rvert\),
  the direct test of the breadth reading. Both a ratio of means
  (dice_over_target_ratio) and a mean of per-problem ratios
  (mean_per_problem_dice_ratio) are reported: extension sizes are skewed, so
  the former is dominated by the largest target extensions while the latter
  weights every problem equally. They answer different questions.
- ``Empty-hypothesis rate`` — how often \(\lvert P \rvert = 0\). A clean
  failure mode; a reduction under dice is a result.
- ``Classification summary`` — per condition, the pooled confusion matrix
  reconstructed from the per-problem result fields, with two MCC figures:
	- ``mean MCC`` — weights every learning problem equally; the quantity the
    paired analysis operates on.
	- ``pooled MCC`` — computed from the summed confusion matrix; dominated by
    problems with large extensions.
  Divergence between them indicates the embedding effect is concentrated in
  problems of a particular size.
- ``Confusion matrix`` — reconstructed per problem as
  \(TP = \lvert P \cap T \rvert\), \(FP = \lvert P \rvert - TP\),
  \(FN = \lvert T \rvert - TP\), \(TN = \lvert U \rvert - TP - FP - FN\), with
  \(\lvert U \rvert\) taken from target_extension.total. A matrix with any
  negative cell is inconsistent: it is logged and skipped, not clamped.
- ``Matthews correlation coefficient (mcc)`` — the skew-robust
  classification metric, in \([-1, 1]\):
  \(\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}\)
  Computed in log space, because the denominator is a product of four
  marginals each as large as the knowledge base and that product overflows
  float64 on vicodi well before the individual counts get large. Returns
  0.0 when any marginal is zero — the conventional degenerate-case
  definition, matching scikit-learn. Note that mcc sits in the
  exploratory layer as an outcome, while the classification summary that
  reports it is mechanism.

### 7.8 Exploratory grid and multiplicity

- ``Exploratory grid`` — the cross product of exploratory outcomes ×
  bucketings × buckets, plus one unbucketed cell per outcome. Each cell is one
  exploratory finding.
- ``Bucketing`` — a complexity field used to partition observations:
  depth, expressivity, or extension_ratio. The extension_ratio
  bucketing is the only legitimate binning in the evaluation, because
  there it is a degeneracy indicator rather than a contiguous predictor being
  tested for trend. Its edges are fixed at
  \[0.00,0.05\) \[0.05,0.25\) \[0.25,0.75\) \[0.75,0.95\) \[0.95,1.00\].
- ``Minimum cell size`` — a bucket with fewer than five collapsed problems is
  skipped, not reported with a meaningless interval.
- ``Benjamini–Hochberg screening`` — step-up FDR control at \(q = 0.10\) over
  the whole exploratory family. FDR rather than family-wise error, because
  family-wise control over a grid this size is self-defeating when the goal is
  screening.
- ``Discovery`` — an exploratory finding whose BH-adjusted p-value is at or
  below \(q\). A discovery is a candidate for follow-up, never a claim.
  The adjustment policy, in full:

| Comparison set | Adjustment |
| --- | --- |
| Primary \(\beta_0\), across knowledge bases | **None.** The claim is a conjunction, not a disjunction; requiring simultaneous rejection everywhere is already conservative, and correcting a conjunction makes it needlessly weaker. |
| Confirmatory \(\beta_1\) | **None.** One pre-specified coefficient per knowledge base. |
| Exploratory grid | Benjamini–Hochberg at \(q = 0.10\). |

> ``Holm`` appears nowhere in the confirmatory path. That is the point of the
> hierarchy: replacing \(k\) per-bucket tests with one trend coefficient
> removes the multiplicity rather than correcting for it.

### 7.9 Verdicts and artifacts

- ``Evaluation result`` — the inferential result for one knowledge base:
  the primary estimate, the trend, the robustness layer, the mechanism and
  classification summaries, the extension sizes, the exploratory findings, and
  optionally the paired design itself.
- ``Suite evaluation`` — evaluation results across every knowledge base, plus
  the conjunction verdict.
- ``Conjunction claim`` — the confirmatory claim is "dice beats random on
  every knowledge base". It is a conjunction, which is why no correction
  is applied across knowledge bases.
- ``Conjunction verdict (conjunction_holds)`` — true only when every
  knowledge base's chosen interval lies strictly above zero. The chosen
  interval is the bootstrap one when agreement == "disagree-trust-bootstrap",
  otherwise the model-based one. A knowledge base with no estimable primary
  counts as a failure, not as a skip.
- ``Conjunction statement`` — the human-readable form, e.g. "dice > random on
  3/4 knowledge bases (95% interval excluding zero). Conjunction does not
  hold." All results are reported regardless of outcome.
- ``Outcome unavailable`` — an outcome the design could not supply at all. The
  canonical case: lift requires atomic_baseline_f1 from the hardness
  annotation stage, and without it the primary does not run. The evaluation
  then falls back to reporting the robustness layer in its place and records
  the substitution in notes.
- ``Notes`` — free text recording every layer that could not be estimated.
  Every layer is guarded independently: a failed layer is noted and the
  remaining layers still run, consistent with the suite's design to finish and
  report rather than abort.
- ``Evaluation artifact`` — the JSON emitted by write_evaluation, sitting
  next to the descriptive summaries so the analysis is auditable and the
  family of tests is explicit.
- ``Paired-observations artifact`` — the optional second file holding every
  paired observation and every paired-hypothesis record. Worth persisting
  separately: it is the evaluation's input, and it enables per-concept
  inspection without re-running anything.

### 7.10 Failure semantics

The evaluation stage distinguishes three kinds of shortfall, and the
vocabulary keeps them apart:

| Situation | Handling | Recorded as |
| --- | --- | --- |
| No runs at all, or no seed carrying both conditions | `InferenceError` — the paired design cannot be assembled | exception |
| A layer cannot be estimated (unidentified slope, too few observations, optimizer failure) | The layer is skipped; the rest still runs | `notes`, `note`, `outcome_unavailable` |
| A single problem failed under NCES | Excluded from every aggregate | `error_counts` |

> ``InferenceError`` is raised only when the paired design itself is
> impossible. Everything downstream degrades and reports.

### 7.11 Evaluation naming rules

Extends §10.

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
| Pratt zeros | zero handling | The specific rule. |
| Hodges–Lehmann estimator | median difference | It is not the median of the differences. |
| mean MCC / pooled MCC | MCC | They differ and the difference is informative. |

---

## 8. Output layout

```text
Output/
└── <benchmark_name>/                      e.g. benchmark1
    ├── benchmark_summary.json             aggregate across all knowledge bases
    ├── evaluation.json                    aggregate across seeds, one KB
    └── <knowledge_base>/
        ├── embeddings_data/               splits
        ├── nces_data/                     learning problems and splits, data settings hash
        ├── ontology_parse_data/           triples and all individuals of a KB
        └── seed<N>/
            ├── embeddings/
            │   ├── <Model>.csv            trained DICE entity embeddings
            │   ├── <Model>_random.csv     random embedding baseline
            │   ├── embedding_report.json  trial metrics and triple counts
            │   ├── data/                  train.txt, valid.txt, test.txt
            │   └── trial_NN_<Model>/      per-trial DICE run directory
            ├── nces/
            │   ├── nces_report.json       the benchmark run report
            │   ├── results/               Results that are collected
            │   │                           within a single run
            │   └── trained_models/
            │       ├── dice/              weights, dice condition
            │       └── random/            weights, random condition
            └── logs/
                └── <knowledge_base>_<seed>.log
```

- **`embeddings/data/`** holds the RDF-triple split for DICE.
- **`/data/`** holds `learning_problems.json`,
  the two `*_problems.json` split files, and `nces_train_data.json`.
- **`trained_models/<condition>/`** is separated per condition so the two
  conditions never share weights.

---

## 9. Reference parameter values

### NCES

Defaults in `input/nces_settings.json`, consumed by `src/models/nces.py`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `learner_name` | `GRU` | Learner architecture: `LSTM`, `GRU`, or `SetTransformer`. Passed upstream as the single-element list `learner_names`. |
| `embedding_dim` | `64` | Entity-embedding width NCES expects; must match the embedding CSV. |
| `epochs` | `300` | Training epochs. |
| `batch_size` | `32` | Learning problems per training step. |
| `proj_dim` | `40` | Projection-representation dimension. |
| `rnn_n_layers` | `1` | Recurrent layers. |
| `drop_prob` | `0.0` | Dropout probability; `0.0` disables dropout. |
| `num_heads` | `2` | Attention heads. |
| `num_seeds` | `1` | Internal learner seeds. |
| `num_workers` | `0` | Data-loader workers; `0` keeps loading in the main process. |

Fixed by the project rather than configured:

| Parameter | Value | Rationale |
| --- | --- | --- |
| `auto_train` | `False` | Training must be explicit and driven by the project's own train split. |
| `load_pretrained` | `False` then `True` | Train from scratch, then reload the saved weights for evaluation. |
| `sorted_examples` | `True` | Stable example ordering. |
| `path_of_embeddings` | condition-specific CSV | The only thing that differs between conditions. |
| `path_of_trained_models` | `trained_models/<condition>/` | Keeps the conditions' weights separate. |

### DICE

Defaults in `input/embedding_settings.json`, consumed by
`src/models/dice.py`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model_name` | `Keci` | DICE architecture. |
| `embedding_dim` | `64` | Base embedding width. |
| `epochs` | `50` | Training epochs. |
| `batch_size` | `64` | Base batch size. |
| `scoring_technique` | `KvsAll` | Scoring mode. |
| `trainer` | `torchCPUTrainer` | CPU training backend. |
| `eval_model` | `train_val_test` | Evaluate on all three splits. |
| `num_core` | `0` | CPU-core setting. |
| `random_seed` | benchmark seed | Passed through per run. |
| `hpo-backend` | `smac` | which hyperparameter optimization to use |

**Hyperparameter grid.** `search_best_embedding_setting` evaluates the
cross product of {base dimension, doubled dimension} × {base batch size,
halved batch size} — four trials at the defaults. The winner is chosen by
validation MRR, then test MRR. This is only relevant when the backend
does not use `smac`

### Project

`input/project_settings.json`:

| Field | Default | Meaning |
| --- | --- | --- |
| `seeds` | `[1, 2, 3, 4, 5]` | Seeds to run. |
| `benchmark_name` | `benchmark1` | Names the output subtree. |
| `embedding_conditions` | `["dice", "random"]` | Conditions to compare. |

### CLI flags

Every flag overrides the corresponding settings file.

| Flag | Overrides |
| --- | --- |
| `--input-dir` | Location of the settings files. |
| `--output-dir` | Root of the output tree. |
| `--benchmark-name` | `project.benchmark_name` |
| `--datasets` | `data_generation.kbs` |
| `--seeds` | `project.seeds` |
| `--embedding-conditions` | `project.embedding_conditions` |
| `--embedding-model` | `embedding.model_name` |
| `--num-problems` | `data_generation.num_rand_samples` |
| `--dice-epochs` | `embedding.epochs` |
| `--nces-epochs` | `nces.epochs` |
| `--hpo-backend` | `embedding.hpo_backend` |
| `--log-level` | Logging verbosity. |

---

## 10. Naming rules

Apply these in code, identifiers, log messages, JSON keys, and prose.

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
| benchmark summary | results file | Aggregate across runs. |
| configuration | config dict, settings blob | The resolved settings. |
| parameter | — | A named function or CLI input. |
| artifact | result file, output file | Any generated file. |
| report | result file | The structured per-run JSON. |
| metric / score | — | A numeric outcome. |
| seed | — | Never a synonym for run or dataset. |
| complexity | difficulty, level | The multi-dimensional object. |
| DL length	| complexity, length, concept length | The scalar token count; one field within complexity. |
| hardness	| semantic difficulty, hardness score	| The reasoner-derived fields, collectively. Not a single number. |
| nesting depth	| depth	| depth alone is the LPGen search parameter. |
| lift	| improvement, delta, gain |	F1 over atomic baseline. |
| atomic baseline F1 |	baseline, trivial score	| Fully qualified; there are other baselines in this project.
| target / hypothesis extension | extension | Always qualified. |
| number of learning problems | problem_count | Clearer in CLI help. |
| property / role | relation | Ontology terminology. |
| output directory | output dir | Full words in prose. |

---

## 11. Quick reference

| Term | One-line definition |
| --- | --- |
| Knowledge base | The OWL ontology being learned from. |
| Learning problem | One target concept plus its positive and negative examples. |
| Target concept | The concept the learner must recover. |
| Hypothesis | The concept expression NCES proposes. |
| Extension | The individuals satisfying an expression, per the reasoner. |
| Semantic equivalence | Target and hypothesis extensions are identical. |
| Complexity | DL-expression length of a target concept. |
| Embedding model / DICE model | The generic / specific entity-embedding model. |
| NCES model | The Ontolearn concept learner under evaluation. |
| Embedding condition | `dice` or `random` — the benchmark's independent variable. |
| Hyperparameter trial | One attempted DICE configuration. |
| Selection metric | Validation MRR, then test MRR. |
| Benchmark run | One (knowledge base, seed) execution. |
| Benchmark suite | All runs in one invocation. |
| Seed | The reproducibility handle for all stochastic behavior. |
| Configuration | The resolved settings for one invocation. |
| Artifact | Any generated file. |
| Report / Summary | Per-run result / cross-run aggregate. |

## 12. Schema versions
Learning-problem schema v1 — `complexity` is an integer, the DL-expression length.
v2 — `complexity` is an object; the v1 integer survives as complexity.dl_length.
v3 — no backwards compatibility for `complexity`. It is a pure multi-dimensional object.
v4 — Now contains inference evaluation section
