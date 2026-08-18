# NCES Benchmark

A benchmark suite measuring how **entity-embedding quality** affects neural
concept learning on OWL knowledge bases.

The suite trains [DICE](https://github.com/dice-group/dice-embeddings)
knowledge-graph embeddings, feeds them to
[Ontolearn's](https://github.com/dice-group/Ontolearn) NCES concept learner,
and scores the resulting hypotheses against a deterministic random-embedding
baseline. Because both conditions share the same knowledge base, learning
problems, splits, learner architecture, and seed, any difference in the
metrics is attributable to the embeddings alone.

---

## The question

Given a knowledge base \( K \), a target concept \( T \), positive examples
\( E^+ \), and negative examples \( E^- \), a concept learner must find a
concept expression \( C \) such that, for \( K' = K \cup \{T \equiv C\} \):

$$ \forall e^+ \in E^+ : K' \models C(e^+) \qquad\text{and}\qquad \forall e^- \in E^- : K' \not\models C(e^-) $$

NCES approaches this as a neural sequence-synthesis task over entity
embeddings. This suite quantifies how much those embeddings actually
contribute — by replacing them with random vectors of identical shape and
re-running everything.

Results are reported as extension-based precision, recall, F1, accuracy,
Jaccard similarity, and exact semantic equivalence, bucketed by target-concept
complexity.

---

## Quick start

The benchmark requires **Linux**: `python-sat`, the DICE CPU trainer, and NCES
data-loader forking all depend on POSIX behavior. Use Docker.

```bash
# 1. Build the image.
docker compose build

# 2. Add a knowledge base.
cp /path/to/semantic_bible.owl datasets/

# 3. Smoke test — one seed, tiny epoch counts, a couple of minutes.
docker compose run --rm benchmark \
  --datasets semantic_bible --seeds 1 \
  --num-problems 8 --dice-epochs 2 --nces-epochs 5

# 4. Full run.
docker compose run --rm benchmark --datasets semantic_bible --seeds 1 2 3 4 5
```

Results land in `Output/benchmark1/`. Start with
`Output/benchmark1/benchmark_summary.json`.

---

## How it works

Each **benchmark run** is one (knowledge base, seed) pair and executes seven
stages:

```text
  OWL file
     │
     ├─► ontology parsing ──────────► RDF triples + individuals
     │
     ├─► learning-problem generation ──► target concepts + example sets
     │        (ontolearn LPGen)
     │
     ├─► learning-problem splitting ───► train / test
     │
     ├─► embedding stage
     │     ├── triple split ─────────► train.txt / valid.txt / test.txt
     │     ├── hyperparameter search ─► best DICE config by validation MRR
     │     ├── export ───────────────► <Model>.csv            (dice)
     │     └── random baseline ──────► <Model>_random.csv     (random)
     │
     ├─► NCES training ──────────────► weights, once per condition
     │
     ├─► NCES evaluation ────────────► hypotheses → extensions → metrics
     │
     └─► aggregation ────────────────► run report, KB summary, suite summary
```

Two design decisions are worth knowing up front:

**Learning problems are split at the problem level, not the example level.**
Splitting one problem's examples across train and test would leak its target
concept into evaluation, making the scores meaningless. Each problem belongs
to exactly one split.

**Hypotheses are scored over extensions, not over sampled examples.** Both the
hypothesis and the target concept are parsed back into OWL and handed to the
reasoner. Scoring against every individual in the knowledge base — true
negatives included — prevents a hypothesis that merely separates the provided
handful of examples from scoring perfectly.

---

## Project layout

```text
.
├── datasets/            OWL knowledge bases (.owl)
├── input/               the four settings files
│   ├── project_settings.json
│   ├── data_generation_settings.json
│   ├── embedding_settings.json
│   └── nces_settings.json
├── Output/              all generated artifacts
├── src/
│   ├── __main__.py           nces-benchmark CLI entry point
│   ├── config.py             typed settings; maps project JSON → upstream kwargs
│   ├── paths.py              canonical output layout, knowledge-base resolution
│   ├── logging_utils.py      console + per-run file logging
│   ├── data/
│   │   ├── ontology.py       OWL parsing, triples, individuals, extensions
│   │   └── lp.py             learning-problem generation, schema, splitting
│   ├── models/
│   │   ├── dice.py           DICE datasets, training, search, export, baseline
│   │   └── nces.py           NCES data prep, training, evaluation
│   └── benchmarking/
│       ├── metrics.py        extension metrics, complexity aggregation
│       └── runner.py         the orchestrator
├── tests/
├── Dockerfile
├── docker-compose.yml
├── environment.yml
├── requirements.txt
└── pyproject.toml
```

> **`dice.py` vs. `dicee`.** The PyPI library is `dicee`; this project's
> wrapper module is `src/models/dice.py`. Every import of the library is
> absolute (`from dicee.executer import Execute`) and
> `src/models/__init__.py` performs no imports, so the names do not collide.

---

## Output layout

```text
Output/benchmark1/
├── benchmark_summary.json          across all knowledge bases
├── semantic_bible_summary.json         across seeds, for `semantic_bible`
└── seed1/
    └── semantic_bible/
        ├── embeddings/
        │   ├── Keci.csv                trained DICE entity embeddings
        │   ├── Keci_random.csv         random embedding baseline
        │   ├── embedding_report.json   trial metrics, triple counts
        │   ├── data/                   train.txt, valid.txt, test.txt
        │   └── trial_00_Keci/ …        per-trial DICE run directories
        ├── nces/
        │   ├── nces_report.json        the benchmark run report
        │   ├── data/
        │   │   ├── learning_problems.json    all problems, by complexity
        │   │   ├── train_problems.json
        │   │   ├── test_problems.json
        │   │   ├── nces_train_data.json      local-name form for NCES
        │   │   └── LPs.json                  raw ontolearn output
        │   └── trained_models/
        │       ├── dice/               weights, dice condition
        │       └── random/             weights, random condition
        └── logs/
            └── semantic_bible.log
```

### Reading the results

`benchmark_summary.json` gives the headline comparison:

```json
{
  "benchmark_name": "benchmark1",
  "per_knowledge_base": {
    "semantic_bible": {
      "num_runs": 5,
      "embedding_conditions": {
        "dice":   { "mean_f1": 0.83, "semantic_equivalence_rate": 0.41 },
        "random": { "mean_f1": 0.52, "semantic_equivalence_rate": 0.09 }
      }
    }
  }
}
```

The gap between conditions is the result the benchmark exists to produce.

For per-problem detail, open a run's `nces/nces_report.json`. Each entry in
`results` carries the target concept, the hypothesis, all six metrics, the
full target-extension counts, and the runtime. The `complexity_summary`
breaks the same metrics down by DL-expression length, which is usually where
the interesting pattern lives: embedding quality tends to matter more as
target concepts grow.

---

## Configuration

Four JSON files in `input/`. Every field has a CLI override.

### `project_settings.json`

```json
{
  "seeds": [1, 2, 3, 4, 5],
  "benchmark_name": "benchmark1",
  "embedding_conditions": ["dice", "random"],
  "stratify_by": "dl_length"
}
```

### `data_generation_settings.json`

```json
{
  "kbs": "semantic_bible",
  "num_rand_samples": 150,
  "depth": 2,
  "max_child_len": 10,
  "refinement_expressivity": 0.2,
  "rho": "ALC",
  "min_num_pos_examples": 1
}
```

`rho` selects refinement-operator expressivity: `ALC` or `ALCHIQD`. Upstream
`LPGen` has no named-operator parameter — it takes a boolean `beyond_alc` — so
`src/config.py` translates. It likewise maps `num_rand_samples → max_num_lps`
and `max_child_len → max_child_length`. All translation lives in
`DataGenerationSettings.lpgen_kwargs()`.

### `embedding_settings.json`

```json
{
  "model_name": "Keci",
  "embedding_dim": 64,
  "epochs": 50,
  "batch_size": 64
}
```

`model_name` is one of `Decal`, `Keci`, `DualE`, `ComplEx`, `QMult`, `OMult`,
`ConvQ`, `ConvO`, `ConEx`, `TransE`, `DistMult`, `Shallom`.

The values above seed a four-point hyperparameter grid — {base, doubled}
dimension × {base, halved} batch size — scored by validation MRR with a
fallback to test MRR. A trial that raises is recorded in `search_trials` and
skipped rather than aborting the run.

### `nces_settings.json`

```json
{
  "learner_name": "GRU",
  "embedding_dim": 64,
  "epochs": 300,
  "batch_size": 32,
  "proj_dim": 40,
  "rnn_n_layers": 1,
  "drop_prob": 0.0,
  "num_heads": 2,
  "num_seeds": 1,
  "num_workers": 0
}
```

`learner_name` is `LSTM`, `GRU`, or `SetTransformer`. Keep `embedding_dim`
equal to the DICE `embedding_dim` — it must match the exported CSV width.
NCES receives no hyperparameter search: it is held fixed so that the
embedding condition remains the only variable.

---

## CLI

```bash
nces-benchmark [options]        # installed entry point
python -m src [options]         # equivalent
```

| Flag | Effect |
| --- | --- |
| `--datasets NAME [NAME …]` | Knowledge bases to evaluate. |
| `--seeds N [N …]` | Seeds to run. |
| `--benchmark-name NAME` | Names the output subtree. |
| `--embedding-conditions {dice,random}` | Conditions to compare. |
| `--embedding-model NAME` | DICE architecture. |
| `--num-problems N` | Learning problems per knowledge base. |
| `--dice-epochs N` | DICE training epochs. |
| `--nces-epochs N` | NCES training epochs. |
| `--input-dir PATH` | Settings-file location. |
| `--output-dir PATH` | Output-tree root. |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Verbosity. |

Examples:

```bash
# Baseline only — no DICE training, so this is fast.
docker compose run --rm benchmark --embedding-conditions random

# Compare two DICE architectures into separate output subtrees.
docker compose run --rm benchmark \
  --embedding-model TransE --benchmark-name transe_run
docker compose run --rm benchmark \
  --embedding-model ComplEx --benchmark-name complex_run

# Several knowledge bases, verbose.
docker compose run --rm benchmark \
  --datasets semantic_bible vicodi mutagenesis --seeds 1 2 3 --log-level DEBUG
```

`--help` is dependency-free: heavy imports are lazy, so it does not load
`torch`.

---

## Adding a knowledge base

1. Drop the `.owl` file into `datasets/`. Either `datasets/<name>.owl` or
   `datasets/<name>/<name>.owl` resolves.
2. Run it: `--datasets <name>`.
3. If learning-problem generation yields too few problems, raise
   `refinement_expressivity` or `depth`, or lower `min_num_pos_examples`.

Larger ontologies mainly cost time in the DICE hyperparameter search
(four trials) and in reasoner extension computation during evaluation.

---

## Environment

Docker is the supported path. `Dockerfile` builds a conda environment on
Python 3.11 with a headless JRE, which `owlapy` needs for its OWLAPI
synchronization layer.

For a local Linux install:

```bash
conda env create -f environment.yml
conda activate nces-benchmark
pip install --no-deps -e .
```

`OMP_NUM_THREADS=1` is set in the image. Unbounded thread counts make CPU
training results drift between otherwise identical runs.

---

## Tests

```bash
docker compose run --rm --entrypoint pytest benchmark tests -v
```

The suite covers configuration translation (including the `rho` → `beyond_alc`
and `max_child_len` → `max_child_length` mappings), the learning-problem
schema and split disjointness, DICE dataset writing and MRR selection
fallbacks, random-baseline determinism, metric edge cases such as empty
hypotheses, and orchestration with the heavy stages stubbed out. Tests
requiring `ontolearn` are guarded by `importorskip`, so the fast subset runs
anywhere.

---

## Failure handling

The suite is built to finish and report, not to abort:

| Failure | Behavior |
| --- | --- |
| A DICE hyperparameter trial raises | Recorded in `search_trials`; search continues. |
| Validation MRR unavailable | Falls back to test MRR; notes it in `validation_error`. |
| NCES raises on one learning problem | Recorded with an `error` field; excluded from aggregates. |
| A target concept fails to parse | Falls back to the sampled positive examples as its extension. |
| An entire benchmark run raises | Recorded in the suite summary's `failures`; remaining runs proceed. |

Every run also writes `logs/<knowledge_base>.log` alongside its report.

---

## Further reading

- `GLOSSARY.md` — canonical terminology and naming rules.
- `KNOWN_ISSUES.md` — upstream defects and the workarounds in use, notably why
  learning-problem generation goes through `LPGen` rather than
  `LearningProblemGenerator.get_examples()`.
- [Ontolearn](https://github.com/dice-group/Ontolearn) — NCES and the OWL
  learner framework.
- [dice-embeddings](https://github.com/dice-group/dice-embeddings) — the
  knowledge-graph embedding library.
