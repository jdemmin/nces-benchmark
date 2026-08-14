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
- **Complexity** — the DL-expression length of a generated target concept.
  Higher values mean a syntactically larger concept and, generally, a harder
  learning problem. Used to bucket results in the `complexity_summary`.

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

> **Naming note.** The third-party library is `dicee` (lowercase, as
> installed from PyPI). This project's module wrapping it is
> `src/models/dice.py`. Referring to the library, write `dicee`; referring to
> our module or the model family, write DICE.

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
- **Split ratios** — the train/validation/test proportions, `[0.8, 0.1, 0.1]`.
  Applied independently to RDF triples (for DICE) and to learning problems
  (for NCES).

---

## 4. Code and workflow vocabulary

### Modules

| Module | Responsibility |
| --- | --- |
| `src/config.py` | Typed settings dataclasses; translates project JSON field names into upstream keyword arguments. |
| `src/paths.py` | The canonical output directory layout and knowledge-base resolution. |
| `src/logging_utils.py` | Console and per-run file logging. |
| `src/data/ontology.py` | OWL parsing, RDF-triple extraction, individual enumeration, extension computation. |
| `src/data/lp.py` | Learning-problem generation, the canonical schema, and splitting. |
| `src/models/dice.py` | DICE dataset preparation, embedding training and search, entity-embedding export, random baseline. |
| `src/models/nces.py` | NCES training-data preparation, training, and hypothesis evaluation. |
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
3. **Learning-problem splitting** — partition problems into disjoint
   train/validation/test sets.
4. **Embedding stage** — write DICE triple splits, run the hyperparameter
   search, export the winning entity embeddings, and generate the random
   baseline.
5. **NCES training** — train the learner on the train split, once per
   embedding condition.
6. **NCES evaluation** — for each held-out problem, synthesize a hypothesis,
   compute its extension, and score it against the target extension.
7. **Result aggregation** — summarize per run, per knowledge base, and per
   suite.

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

> **Why `LPGen` and not `LearningProblemGenerator`?** The lower-level
> `LearningProblemGenerator.get_examples()` is broken upstream; see
> `KNOWN_ISSUES.md`. Going through `LPGen` avoids the defect entirely, because
> `LPs.json` already contains materialized example sets.

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

On the name rho. In KB2Data.__init__, rho is the local variable holding the constructed ExpressRefinement operator — the conventional DL symbol \(\rho\) for a refinement operator. It is not a settable parameter. Three fields shape it: beyond_alc toggles the five use_* constructor flags as a group, refinement_expressivity becomes expressivity, and downsample_refinements becomes downsample

### Learning-problem schema

Every generated learning problem is serialized as:

```json
{
  "id": "lp_0000",
  "target_concept": "male ⊓ ∃ hasChild.person",
  "pos_example": ["http://example.com/father#stefan"],
  "neg_example": ["http://example.com/father#anna"],
  "complexity": 4,
  "num_pos": 1,
  "num_neg": 1
}
```

- `id` — stable identifier within one benchmark run.
- `target_concept` — DL-syntax string.
- `pos_example` / `neg_example` — sorted lists of **full IRIs**.
- `complexity` — DL-expression length.
- `num_pos` / `num_neg` — example counts, derived automatically.

The full set is persisted **grouped by complexity level** in
`learning_problems.json`; the splits are written as flat lists.

### Splitting

Learning problems are split at the **problem** level, not the example level.
Splitting a single problem's examples across train and test would leak the
target concept into evaluation, and the resulting scores would be
meaningless. The splits are disjoint by construction and deterministic in the
seed.

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

| Field | Contents |
| --- | --- |
| `id`, `target_concept`, `complexity` | Copied from the learning problem. |
| `hypotheses` | The serialized NCES hypothesis, in DL syntax. |
| `num_pos` / `num_neg` | Counts of the *sampled* examples. |
| `target_positive_count` / `target_negative_count` | Counts over the *full* target extension — not the sampled examples. |
| `target_extension_size` | Object with `positive`, `negative`, `total`. |
| `target_extension_overlap` | Object with `intersection`, `union`, `jaccard`, `precision`, `recall`. |
| `accuracy`, `precision`, `recall`, `f1`, `jaccard`, `semantic_equivalence` | The metrics above. |
| `runtime_seconds` | Wall-clock time for this problem. |
| `error` | Present only if NCES raised; the problem is then excluded from aggregates. |

Run-level and embedding-level fields:

| Field | Contents |
| --- | --- |
| `complexity_summary` | Per-complexity aggregates: count, mean F1, mean accuracy, mean precision, mean recall, mean Jaccard, semantic-equivalence rate. |
| `best_embedding_config` | The selected DICE configuration with its score and metrics. |
| `search_trials` | Every attempted hyperparameter trial, including failures. |
| `validation_error` | Explanatory text when validation MRR was unavailable and a fallback was used. |

A failed trial or a failed learning problem is **recorded, not fatal**. The
benchmark is designed to complete and report partial results rather than
abort.

---

## 7. Output layout

```text
Output/
└── <benchmark_name>/                      e.g. benchmark1
    ├── benchmark_summary.json             aggregate across all knowledge bases
    ├── <knowledge_base>_summary.json      aggregate across seeds, one KB
    └── seed<N>/
        └── <knowledge_base>/
            ├── embeddings/
            │   ├── <Model>.csv            trained DICE entity embeddings
            │   ├── <Model>_random.csv     random embedding baseline
            │   ├── embedding_report.json  trial metrics and triple counts
            │   ├── data/                  train.txt, valid.txt, test.txt
            │   └── trial_NN_<Model>/      per-trial DICE run directory
            ├── nces/
            │   ├── nces_report.json       the benchmark run report
            │   ├── data/                  learning problems and splits
            │   └── trained_models/
            │       ├── dice/              weights, dice condition
            │       └── random/            weights, random condition
            └── logs/
                └── <knowledge_base>.log
```

- **`embeddings/data/`** holds the RDF-triple split for DICE.
- **`nces/data/`** holds `learning_problems.json` (grouped by complexity),
  the three `*_problems.json` split files, and `nces_train_data.json`.
- **`trained_models/<condition>/`** is separated per condition so the two
  conditions never share weights.

---

## 8. Reference parameter values

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

**Hyperparameter grid.** `search_best_embedding_setting` evaluates the
cross product of {base dimension, doubled dimension} × {base batch size,
halved batch size} — four trials at the defaults. The winner is chosen by
validation MRR, then test MRR.

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
| `--log-level` | Logging verbosity. |

---

## 9. Naming rules

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
| complexity | difficulty, level | DL-expression length. |
| target / hypothesis extension | extension | Always qualified. |
| number of learning problems | problem_count | Clearer in CLI help. |
| property / role | relation | Ontology terminology. |
| output directory | output dir | Full words in prose. |

---

## 10. Quick reference

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
