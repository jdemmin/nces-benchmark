# Methodology — Compressed Reference

## Learning problem (formal)
Given KB `K`, target `T`, positive examples `E+`, negative `E-`: find concept `C` s.t. for `K'=K∪{T≡C}`, `K'⊨C(e+) ∀e+∈E+` and `K'⊭C(e-) ∀e-∈E-`.
**Scoring is NOT done on this eq. directly** — the benchmark scores the *extension* of the hypothesis vs. the extension of the target over ALL individuals (not the sampled examples).

## Research questions & design
- RQ1: effect of KGE model choice on quality/learnability of DL class expressions.
- RQ2: effect of KGE hyperparameters on performance.
- Design: **paired, within-problem**. Sole IV = *embedding condition*. Held fixed within a comparison: KB, learning problems (LPs), example samples, train/test split, NCES architecture/hyperparams, seed.

## Embedding conditions (13 total)
- `random` — **control/reference level**, all effects differenced against it. Matrix `|U|×128`, entries iid Uniform[-1,1), deterministic in benchmark seed.
- 12 KGE architectures (trained via DICE, hyperparams from search): `Keci, DeCaL, DualE, ComplEx, QMult, OMult, ConvQ, ConvO, ConEx, TransE, DistMult, Shallom`.
  Families: translational(TransE), bilinear(DistMult,ComplEx), hypercomplex(QMult,OMult,DualE), Clifford(Keci,DeCaL), convolutional(ConvQ,ConvO,ConEx), shallow(Shallom).

## Estimand
`d^c_ij = m^c_ij − m^random_ij` (paired diff; condition c, problem i, seed j, outcome m).
`Δ^c = E[d^c_ij]` = "embedding effect" of c, per KB.
Embedding dim fixed = 128 for every condition ⇒ pair differs ONLY in embedding-matrix content.

## Knowledge bases (4, never pooled)
`semantic_bible`, `vicodi` (historical/cultural) · `carcinogenesis`, `mutagenesis` (biomedical/chemical). Chosen for domain + structural heterogeneity.

## Ontology parsing
- Only IRI→IRI triples embedded; literal-object triples dropped (equal for all conditions → unbiased, but bounds `dice` ceiling; literal info unavailable to any condition).
- Entities indexed by **local name** (fragment after final `#`/`/`); must be unique. Colliding IRIs (≥2 share local name) are **all dropped**, never merged (symmetric ⇒ unbiased). Collision count logged/reported per KB.
- Derived artifacts: RDF triple set (for embedding training), universe `U` = all named individuals.

## LP generation (refinement-based search, Kouagou 2022)
| field | upstream name | value |
|---|---|---|
| num_rand_samples | max_num_lps | 1500 / KB |
| max_child_len | max_child_length | 25 |
| depth | depth | 10 (search depth ≠ concept nesting depth) |
| beyond_alc | beyond_alc | `ALC`→False (toggles 5 constructor flags as group) |
| refinement_expressivity | expressivity | semantic_bible .9, vicodi .5, carcinogenesis .6, mutagenesis .8 |

`refinement_expressivity` varies per KB (fixed within KB/across conditions&seeds ⇒ no bias to paired contrast) but makes LP populations non-comparable across KBs ⇒ reason for never pooling.

**Degeneracy rejection**: LP needs ≥1 positive AND ≥1 negative example, else rejected at construction.

## Complexity annotation (per target concept)
- Structural (expression-only): `dl_length` (token count), `depth` (max quantifier nesting), `constructors` (multiset), `num_atomic_classes`, `num_roles`, `expressivity` (EL/ALC/ALCHIQD — smallest fragment containing expr, from constructors actually present).
- Hardness (reasoner-derived, KB-dependent, computed once & cached): `extension_size` = |T|, `extension_ratio` = |T|/|U|, `atomic_baseline_f1` = best F1 by any single atomic class, `redundant` = bool(target ext == some atomic class ext).

## LP schema (JSON)
`id` (hash of target_concept+example lists; stable across seeds = join key for pairing), `target_concept`, `pos_example[]`, `neg_example[]` (SAMPLED, not full extension), `complexity{…}`, `num_pos`, `num_neg` (sample counts, ≠ extension_size).

## Splitting
- Split at **problem level** (never example level) — avoids target leakage.
- Deterministic hash of data-gen settings; stratified by `depth` (structural, not hardness — avoids coupling split to reasoner stage).
- 1500 LPs/KB, 0.8/0.2 split → ~300 test problems/KB → ≤1500 paired obs per (condition, KB, outcome).
- Generation + annotation + splitting: seed-independent, fixed seed=0, computed once/KB, reused across all 5 benchmark seeds (parsing needs no seed) — this is what keeps LP `id` stable across seeds and makes pairing possible.
- Benchmark seeds ∈ {1,2,3,4,5} govern: HPO search, DICE training, random baseline, NCES train/eval.

## Treatment: `dice` condition
Fixed DICE config: `model_name=Keci`(default name of condition table), `embedding_dim=128`, `scoring_technique=KvsAll`, `trainer=torchCPUTrainer`, `eval_model=train_val_test`, `random_seed=`benchmark seed.
HPO-overridable defaults: `epochs=150`, `batch_size=64`, `learning_rate=0.1`.

### Hyperparameter search
Treatment = **output of a search**, not a single model (models practitioner-realistic quality under fixed budget).
SMAC3, random-forest surrogate, **32 trials**, single worker (determinism), `trial_walltime_limit=null` (avoids pynisher subprocess discarding per-trial records).
Search space: `batch_size∈{32,64,128,256,512}`(tuned), `learning_rate∈[0.001,0.3]` log-scale(tuned), `epochs∈[25,100]`(tuned); `embedding_dim=128` and `scoring_technique=KvsAll` fixed/not tuned.
Selection metric: validation MRR, fallback test MRR if unavailable (logged as `validation_error`). Failed trials recorded, not fatal. **Every** trial (incl. failures) persisted — raw material for RQ2.

### `random` condition (control)
`|U|×128` matrix, iid Uniform[-1,1), deterministic in benchmark seed. Strong baseline: zero-mean symmetric support, matched dimensionality/capacity, independent of graph (isolates graph-derived info content). Null result ⇒ bounds how much learner *exploits* structure via embedding input, not "embeddings uninformative."

### Dimensionality
Fixed `d=128` for all conditions, excluded from search space ⇒ matched capacity per pair, no architecture advantaged by width, no seed-to-seed width variance. Cost: no architecture evaluated at its own optimal width (limitation).

## Concept learner: NCES, GRU variant only
`learner_name=GRU` (sole architecture — no architecture factor in design), `embedding_dim=128`, `epochs=300`, `batch_size=128`, `proj_dim=128`, `rnn_n_layers=1`, `drop_prob=0.0` (deterministic given seed), `num_heads=2`(inactive for GRU), `num_seeds=1`(NCES-internal, ≠ benchmark seed), `max_length=48` (max synthesized expr length; targets exceeding it are unrecoverable by EITHER condition — symmetric ceiling), `num_predictions=5`, `learning_rate=0.0001`, `auto_train=False`, `load_pretrained=False→True` (train then reload for eval), `sorted_examples=True`.
Trained once per embedding condition, on TRAIN split only; weights in condition-specific directories (no parameter sharing).

## Outcome measurement
Reasoner: OWLAPY `SyncReasoner` wrapping **HermiT** (sound+complete for SROIQ ⇒ complete for EL/ALC/ALCHIQD). Completeness required for validity of `semantic_equivalence` and `atomic_baseline_f1`.
Scoring on **extensions over all individuals**, never on sampled examples (example-based scoring is trivially gameable).

### Metrics
`P`=hypothesis extension, `T`=target extension, `U`=all individuals. All ratios → `0.0` on empty denominator (never raise).
- Precision = |P∩T|/|P|; Recall = |P∩T|/|T|; F1 = harmonic mean.
- Accuracy = fraction of U correctly classified (incl. TN); dominated by TN when extension_ratio small.
- Semantic equivalence = bool(P==T); aggregated as a rate, never averaged as score.
- **ABL** (primary outcome) = F1(H) − b, where b = max over admissible atomic classes A (all named classes excl. owl:Thing/owl:Nothing) of F1(A). `b` is optimistically biased (chosen with knowledge of target) but is identical across conditions per problem ⇒ cancels exactly in paired diff.
- **ABL_norm** (secondary) = (F1(H)−b)/(1−b), computed only if `b≤0.95` (undefined at b=1, unstable near it); reported as median + count of excluded problems.
- Jaccard NOT reported — monotone reparam of F1 (`J=F1/(2−F1)`), no independent info.

### Confusion matrix (per problem)
`TP=|P∩T|`, `FP=|P|−TP`, `FN=|T|−TP`, `TN=|U|−TP−FP−FN`. Any negative cell ⇒ logged & **skipped**, never clamped.

### Ranking metrics
MRR, Hits@10 on held-out triples, per trained KGE model. MRR = HPO selection metric AND independent embedding-quality measure regressed against downstream ABL (central RQ1 analysis).

### Per-problem record
id, target_concept, complexity, serialized hypothesis, sample counts, target extension counts+total, all metrics, wall-clock runtime, `error` field (present only if learner raised → excluded from all aggregates; run continues, reports partial).

## Analysis strategy
Consumes persisted artifacts only, never re-executes the learner (reproducible, non-perturbing). Deliberately lean (13 conditions×4 KB×5 seeds = 52 contrasts): one estimator, one interval, one non-parametric test, plus diagnostics — equal prominence.

### Pairing
Atomic observation = (condition, LP, seed) triple for one outcome. Pair `dice`↔`random` via join key (LP id) within same seed; a problem contributes only if BOTH conditions produced a usable value, else "unpaired" (excluded, id recorded).
Paired-design record (per condition, KB) stores: surviving seeds/problems, atomic-obs count, unpaired ids, per-condition failure counts, both conditions' hypothesis strings side-by-side (enables identical-hypothesis diagnostic + per-concept inspection w/o rerun).

### Collapsing over seeds
Per-problem paired diffs averaged over 5 seeds → 1 value/problem ("collapse"). Reduces residual-noise SD by ~√5; supports the cross-problem-independence assumption needed by Wilcoxon. Run-to-run variability not discarded — reported separately as seed spread.

### Point estimate + interval
`Δ̂^c = mean_i( mean_j(d^c_ij) )` (mean of collapsed per-problem diffs).
Interval: percentile **cluster bootstrap**, 10,000 resamples, **seed is the resampling unit** (whole seeds resampled w/ replacement; never resample individual obs — would destroy within-run dependence, giving too-narrow interval).

### Wilcoxon signed-rank (robustness check, not a separate analysis)
On collapsed diffs. **Zero handling = Pratt's rule** (fixed, not configurable): ranks zeros then drops them from signed sum (vs. Wilcoxon's discard-then-rank, which inflates non-zero magnitudes) — appropriate when zero mass is large (learner often synthesizes identical hypothesis under both conditions).
p-value: exact sign-flip null enumeration for n≤20, else Monte Carlo (not asymptotic normal).
Effect size: **Hodges-Lehmann estimator** (median of pairwise Walsh averages) — NOT median of diffs; reported alongside Δ̂^c (a significant Wilcoxon w/ near-zero mean is possible when zero mass large + tails asymmetric).
Sign test on nonzero diffs = extra assumption-light cross-check; win/loss/tie triple given equal prominence.
Degenerate contrast (all diffs = 0): report p=1, triple=(0,0,n).

### Diagnostics (per contrast, equal prominence to interval)
- Zero fraction — share of problems where condition changed nothing.
- Identical-hypothesis fraction — share of byte-identical serialized hypothesis strings (stronger evidence of no-effect than zero fraction).
- Win/loss/tie triple.
- Empty-hypothesis rate (|P|=0), per condition.

### Seed-level spread
5 per-seed mean diffs reported individually + SD. Compare to collapsed per-problem spread: per-problem spread ≫ seed-level ⇒ single run trustworthy, concept-level heterogeneity is the interesting variation; reverse pattern ⇒ few-seed conclusions fragile.

## Reporting structure
Always **per knowledge base**, never pooled (KBs differ in domain/size/structure/LP-population character). Cross-KB consistency assessed by **agreement in sign and ordering**, not averaging.

## RQ1 — answered via 3 artifacts (increasing interpretive weight)
1. **Conditional table**: per KB, 12 conditions × [Δ̂^c, bootstrap interval, Wilcoxon p, Hodges-Lehmann, win/loss/tie].
2. **Cross-KB ranking**: conditions ranked by Δ̂^c within each KB; agreement via pairwise Kendall's τ across the 4 KBs. High concordance ⇒ architecture ordering transfers across domains; low ⇒ best embedding is a property of the ontology, not the architecture (answers RQ1 negatively).
3. **[Primary evidence] Link-plot**: per (condition, KB, seed), plot MRR (link-prediction quality) vs. mean ABL over test split (downstream). Summarized via Spearman's ρ computed **within each KB** (avoids cross-ontology scale confound). Pre-registered interpretation:
   - Positive monotone relationship ⇒ MRR is a usable proxy for embedding utility in NCES.
   - Flat/absent relationship (+ small effect vs. random control) ⇒ NCES uses embedding mainly as an identifier, not a structure carrier.

## RQ2 — answered in 2 parts
1. **Observational (trial record)**: all HPO trials (12 arch × 4 KB × 5 seed × 32 trials) with config + validation MRR. Reports: marginal hyperparameter↔validation-MRR relationship, and **selection stability** (does search converge to similar configs across seeds for a given arch/KB?). Relates HPs → *embedding* quality only (not downstream — only the selected config ever reaches the concept learner).
2. **Experimental (configuration sub-study)**: 1 KB + 1 architecture fixed (pre-registered selection: architecture = widest observed validation-MRR spread in its trial record; KB = largest main-suite effect vs. control). 4 configs drawn from that architecture's own trial record: best-MRR, worst-MRR, median-MRR, extreme-learning-rate. Each trained/exported, passed to NCES under the same 5 seeds, all else unchanged. Adds 20 runs.

## Complexity trend (descriptive only, not confirmatory)
Mean paired diff vs. `depth`, per-seed points overlaid; OLS slope on collapsed diffs + seed-cluster bootstrap interval reported as plot description (few distinct depth levels ⇒ not a confirmatory test). Also plotted vs. `extension_ratio` to separate a genuine depth effect from an extension-size artifact (greater depth → smaller extensions → unstable F1/ABL); trend called "complexity-related" only if the two plots dissociate.

## Descriptive breakdowns (no p-values, no multiplicity correction)
Primary outcome broken down by: `expressivity` (EL, ALC); `extension_ratio` bands `[0,.05) [.05,.25) [.25,.75) [.75,.95) [.95,1]` (fixed edges, class-balance indicator not a trend predictor). Reported as mean / win-loss-tie tables with per-cell N; cells with n<5 flagged, never omitted.

## Mechanism layer (interpretive only, no p-values, no new claims)
Per `precision`, `recall`, `hypothesis_extension_size`: both conditions' means, mean diff, Hodges-Lehmann estimate, win/loss/tie. Tests the "breadth hypothesis" (recall-heavy + precision-flat ⇒ one condition synthesizes broader, not better, hypotheses).
Extension-size check: `|P|` vs `|T|`, two ratio families — ratio-of-means (dominated by largest targets) and mean-of-per-problem-ratios (equal weight per problem); denominator = # problems with available target extension size.
Caution: `hypothesis_extension_size` differenced on **raw count scale**, not comparable across problems/KBs (|U| varies) — read within-KB only.

## Precision / status of conclusions
Precision governed by **seed count (5)**, not problem count (~300/KB already well-determined). 5-cluster bootstrap ⇒ coarse intervals, tail-sensitive to extreme seeds. Non-rejection = "failure to establish effect at attainable precision," **not** evidence of absence. No formal power calculation (would require assuming the seed-level variance this work measures).

## Failure semantics (3 tiers)
1. No runs at all / no seed with both condition & control present → **exception raised** (paired design impossible).
2. An analysis is uncomputable (too few obs, unidentified slope, missing annotation) → that analysis skipped, others still run; per-block note + unavailable-outcome list.
3. Single problem fails under the learner → excluded from every aggregate; per-condition error counts kept.
Special case: `atomic_baseline_f1` missing → ABL unavailable → same estimator applied to raw F1 instead, substitution explicitly recorded (never silently presented as ABL).

## Reproducibility / scale
- Main suite: 4 KB × 13 conditions × 5 seeds = **260** concept-learner runs.
- 12 KGE conditions require 4×12×5 = **240** HPO searches × 32 trials each.
- Configuration sub-study: +**20** runs.
- Seeding regimes: (a) materials-construction (parsing, LP gen, annotation, splitting) — seed-independent, fixed seed=0, computed once per KB, reused across all 5 benchmark seeds (parsing needs no seed) — enables stable LP `id` across seeds. (b) run-constituting stages (HPO search, DICE training, random baseline, NCES train/eval) — governed by benchmark seed ∈ {1,2,3,4,5}.

## Persisted artifacts
Benchmark summary · Suite analysis (per condition×KB: estimate/interval/test/diagnostics/seed-spread/mechanism/ext-size/breakdowns + cross-KB ranking concordance + MRR-vs-ABL summary) · Paired observations (every paired obs + paired-hypothesis record, no re-execution needed) · Ontology parse data (triples, individual enumeration, collision report) · LP data (problem set by DL length, 2 split files, learner training data, data-settings hash) · Embedding artifacts (triple splits, exported matrices/condition, selected-trial metrics [MRR, Hits@10], EVERY attempted trial incl. failures, per-trial run dirs) · Extension cache (reasoner extensions keyed by expression string) · Run reports (per-problem results, selected config+score, full trial record, validation-fallback note) · Trained weights (per condition, separate dirs, no sharing) · Logs (one per KB×condition×seed).
Compression: each seed directory compressed at run end; whole benchmark directory compressed at suite end.

## Limitations
1. Literal-value assertions unavailable to treatment (matters most for carcinogenesis/mutagenesis numeric attrs) — bounds ceiling, doesn't bias paired contrast (neither condition gets it).
2. Local-name collisions shrink universe U (symmetric/unbiased but reduces N; counts reported per KB).
3. Random embeddings are a strong control — null result bounds "exploitation of structure," not "embeddings uninformative in general."
4. Fixed d=128 — no architecture at its individually optimal width; reported ordering is "at 128 dimensions."
5. Fixed 32-trial search budget — conflates model class with hyperparameter-sensitivity; not a claim about best attainable quality per architecture.
6. Only the GRU variant of NCES evaluated (not set-transformer/LSTM) — conclusions are GRU-specific.
7. 5 seeds — coarse bootstrap intervals; non-rejection ≠ evidence of absence.
8. `refinement_expressivity` varies per KB — LP populations not comparable across KBs — reason results are never pooled.
9. `max_length=48` decoding cap — targets exceeding it can't be recovered exactly by either condition (symmetric, lowers ceiling on semantic_equivalence independent of embedding quality).
10. Mechanism (extension-size) differences are raw-scale — read within a KB only.
11. RQ2's experimental part covers only 1 KB × 1 architecture (not generalized); its observational part relates hyperparameters to embedding quality, not to downstream performance.

## Key citations
Kouagou et al. 2022 (LP refinement search) · Glimm et al. 2014 (HermiT reasoner) · Lindauer et al., JMLR v23:21-0888 (SMAC3) · Hodges & Lehmann 2011 (Hodges-Lehmann estimator) · Pratt 1959 (zero-handling rule) · Woolson 2007 (Wilcoxon signed-rank) · Kendall 1938 (Kendall's τ) · Spearman 1961 (Spearman's ρ)
