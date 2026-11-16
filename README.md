# TCHG

TCHG models dynamic trust with entity, behavior, and context evidence, masked temporal memory, conditioned heterogeneous propagation, and uncertainty-aware prediction. This repository combines the TCHG algorithm release with the existing public Epinions, Ciao, and CiaoDVD datasets. Training configurations cover timestamped Epinions and the bundled random-link Ciao/CiaoDVD splits. Raw-data regeneration currently covers Epinions.

Public repository: <https://github.com/pandasir108/TCHG>.

## Latest experimental version

The default `main` branch contains the latest experimental implementation,
including causal message propagation, differentiable memory writing, explicit
Observed-user/Unobserved-user evaluation, memory rates expressed per day,
and training-only static trust features. This release starts from development
commit `b5df9d754fefde5c8591ed23575fe8e31f53f277`.

| Dataset | Current configuration | Evaluation |
| --- | --- | --- |
| Epinions | `configs/epinions_70_15_15_memory_days.yaml` | Timestamped trust; separate Observed-user and Unobserved-user reports. |
| Ciao | `configs/ciao_70_15_15_static_refined.yaml` | Fixed training trust graph; all-user random-link split. |
| CiaoDVD | `configs/ciaodvd_70_15_15_diffusion.yaml` | Fixed training trust graph with directed diffusion; all-user random-link split. |

The development results described below are single-seed experiments. They
should not be interpreted as the paper's five-seed aggregate results.

```bash
git clone https://github.com/pandasir108/TCHG.git
cd TCHG
```

## Setup

Python 3.10 or 3.11 is recommended. Create either a Conda environment:

```bash
conda env create -f environment.yml
conda activate tchg
```

or a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The repository contains environment specifications, not a generated environment directory. Do not commit `.env`, `.venv`, `venv`, or Conda environment folders.

## Data

Raw and processed data are already included under `data/`; see [the dataset inventory](data/README.md). To regenerate an Epinions split, the required raw trust file is:

```text
data/raw/epinions/user_user_time.txt
```

Then prepare either split:

```bash
python runners/prepare_data.py --config configs/epinions_70_15_15.yaml
python runners/prepare_data.py --config configs/epinions_80_10_10.yaml
```

## Main experiment

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_main.py --config configs/epinions_70_15_15_memory_days.yaml --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_main.py --config configs/ciao_70_15_15_static_refined.yaml --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_main.py --config configs/ciaodvd_70_15_15_diffusion.yaml --device cuda
```

Use `--device cuda`, `--epochs N`, or `--output PATH` to override the defaults. Checkpoints and metrics are written under `outputs/`.

## Observed-user and Unobserved-user evaluation

The two paper tasks refer to **user history**, independently of event
missingness. The fixed reference users are endpoints of genuine positive
training relations (`split == 0`, `sample_type == 0`, `label > 0.5`). For a
candidate directed pair `(src, dst)`:

| Field | Meaning |
| --- | --- |
| `observed_user_mask == 1` | Both endpoints belong to the training reference users: **Observed-user**. |
| `unobserved_user_mask == 1` | At least one endpoint does not: **Unobserved-user**. |
| `observed_mask == 1` | This event's evidence is available for eligible historical updates. This is an event mask. |

Every candidate belongs to exactly one user regime. A sampled training
negative does not add its endpoints to the reference users. Validation/test
relations and changes to `observed_mask` do not change this reference set.
Thus an Unobserved-user test pair may have an available event mask, and an
Observed-user pair may have a missing event mask. The `p_missing` configuration
controls event evidence missingness. All labeled training candidates still
participate in the training loss.

The definition is implemented once in `tchg/data/user_regimes.py`. Newly
prepared data store both explicit masks. When loading older included files,
the loader derives them in memory and validates the legacy `is_new_node`
field against the training-only definition. Inconsistent stored grouping
metadata raises an error. Loading leaves the dataset file intact.

Each seed trains one model and reports both user regimes from the same
checkpoint. The default monitor is `val_overall_mrr`; calibration and threshold
tuning also use overall validation. This policy is recorded explicitly and
does not use test scores for selection. To use a different validation monitor,
configure `val_observed_user_mrr` or `val_unobserved_user_mrr` before running and
report that policy consistently.

MRR is computed **within each user regime**, using 100 negative scores sampled
uniformly with replacement from that regime's own negative-score pool
(sampling seed 42). Optimistic and pessimistic tied ranks are averaged. AP and
AUC also use the corresponding subset. The overall candidate set is sampled
at the configured positive/negative ratio; its two user subsets retain their
actual class counts and need not individually be 1:1.

The main command displays separate Observed-user and Unobserved-user rows,
with the combined result labeled `Overall (auxiliary)`, and writes:

- `test_user_regimes.csv`: separate rows with candidate/positive/negative
  counts, MRR, AP, AUC, F1, accuracy, and global MRR.
- `metrics.json`: named `test_by_user_regime` results plus flat
  `observed_user_*`, `unobserved_user_*`, and `overall_*` fields under `test`.
- `evaluation_protocol.json`: definitions, dataset hash, per-split group and
  class counts, event-mask cross counts, seed, ranking rules, and selection policy.
- `history.csv`: validation metrics for each user regime at every epoch.

For compatibility, `old_*` aliases `observed_user_*`, `new_*` aliases
`unobserved_user_*`, and unprefixed metrics alias overall results.
`is_new_node` aliases `unobserved_user_mask` for candidate pairs. Report the
explicit task names when comparing with the paper. Overall MRR uses its own
pooled negative scores and is not a weighted mean of the two task MRRs.

Use `--seed N` to override the configuration seed and a distinct output folder
for each run. Each invocation is a single-seed result; a multi-seed mean
requires completing and aggregating the corresponding runs. Prepare candidate
data once and reuse the same `events.npz` across training seeds; compare the
recorded dataset hashes to confirm the shared candidate set. The ablation and
robustness runners also write `summary_user_regimes.csv` with explicit task rows.

## Ablation experiment

```bash
python runners/run_ablation.py --config configs/epinions_70_15_15.yaml
```

Run selected variants by passing their exact names:

```bash
python runners/run_ablation.py --config configs/epinions_70_15_15.yaml --variants "Full TCHG" "w/o Masked Memory"
```

## Robustness experiment

Train the main model first, then evaluate the same checkpoint under missing observations and trust- or graph-oriented perturbations:

```bash
python runners/run_robustness.py --config configs/epinions_70_15_15.yaml
```

Rates and scenarios can be overridden:

```bash
python runners/run_robustness.py --config configs/epinions_70_15_15.yaml --rates 0.1 0.2 0.3 --scenarios missing trust_oriented gnn_oriented
```

## Tests

```bash
pip install -e ".[test]"
pytest -q
```

## Development provenance and experiments

The code and data were consolidated on 2026-11-15 without changing algorithm,
metric, preprocessing, or training logic. The original algorithm files are
recorded in `docs/consolidation.json`; existing tracked dataset files retain
their pre-consolidation hashes.

The pre-fix baseline, intermediate branches and historical checkpoints are
retained in the private development archive. This public repository starts
with the latest experimental snapshot; the dated sections below document its
development history. Historical tags mentioned in the consolidation metadata
are not part of this public snapshot. Use a separate output directory for each
run, for example:

```bash
python runners/run_main.py --config configs/epinions_70_15_15_memory_days.yaml --device cuda --output outputs/experiment_001
```

Checkpoints and outputs are generated locally under `outputs/`. Passing the
unit tests does not establish that published aggregate scores have been
reproduced.

## Causal graph and memory correction (2026-11-15)

The experiment branch replaces the query-batch graph with a chronological
message graph. Each propagation layer aggregates historical event-time
messages whose timestamps are strictly earlier than the query. Sampled
negative pairs are prediction queries and never become historical edges.
Unobserved events do not supply graph messages. Events with the same timestamp
read the same snapshot, and timestamp groups stay together across batches.

Memory writes retain their gradient graph within an optimization chunk.
Detachment happens in the trainer after backward and the optimizer step.
The fast memory implementation now vectorizes within timestamp groups;
simultaneous eligible writes to the same key are averaged. This preserves
chronology while allowing the GRU, input projection, update gate, and reconnect
gate to learn from later predictions.

Validation replays observed training history with the current model weights.
Test replays training and observed validation history. Held-out prediction
queries do not write into memory or the historical graph. The existing
processed features, negative-sampling protocol, metrics, and hyperparameters
are unchanged; this correction does not claim to resolve every issue in
preprocessing or the evaluation protocol.

Run the corrected Epinions 70/15/15 experiment with:

```bash
python runners/run_main.py --config configs/epinions_70_15_15_causal_fix.yaml --device cuda --epochs 50
```

Artifacts and the matched seed-42 baseline are kept separately under
`outputs/causal_fix_20261115/`. The original code remains available at
`baseline/20261115-before-code-fixes`.

## Experimental evidence and conditioning improvements

`configs/epinions_70_15_15_causal_improved.yaml` enables optional changes on top
of the causal correction. Earlier configurations retain the causal-fix
behavior; the improved candidate has its own configuration.

- Transform the first two entity-history proportions to scaled log odds,
  retaining distinctions between values close to one.
- Append historical reciprocal-edge existence and recency to behavior evidence,
  and two-hop counts and incoming/outgoing neighbor Jaccard similarities to
  context evidence. Training queries read strictly earlier timestamp groups;
  validation reads the training graph; test reads training plus observed
  validation positives. Sampled negatives and test outcomes never update this
  feature graph.
- Omit the existing global-horizon clock columns from neural predictors while
  retaining actual event timestamps and relative gaps for state updates.
- Standardize encoder and edge inputs using training examples only. Means and
  scales are saved in the model checkpoint and stay fixed during inference.
- Scale memory contributions to the trust representations and prediction head
  by `memory_strength`. The candidate uses `0.1`; all memory writers remain
  trainable through later events.

The processed dataset is left intact. Structural evidence is appended in
memory, and its optional cache is fingerprinted by queries and admissible
historical writes. The existing sampled-ranking metric and negative-sampling
policy are retained. These changes add model inputs and alter conditioning;
they should be reported as an experimental model variant.

Run the ten-epoch candidate with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_main.py --config configs/epinions_70_15_15_causal_improved.yaml --device cuda
```

Validation is used for checkpoint selection, calibration, and threshold tuning.
The main runner then evaluates the selected checkpoint on test. Use the
`Trainer.fit` interface for validation-only comparisons before selecting a
recipe. Numerical experiment outputs remain local under the ignored
`outputs/` directory.

## Refinement with native past-rating evidence

`configs/epinions_70_15_15_causal_refined.yaml` is a further experimental
variant. It adds evidence from the included native `user_item_time.txt`:
rating activity, mean/deviation and helpfulness summaries for each endpoint;
shared-item overlap and rating agreement; and rating-category similarity.
Review text is ignored. Raw user IDs must match the trust preprocessing map.
Newly prepared files store `node_to_raw`; older included files are checked
against the raw trust endpoints and timestamps before rating evidence is used.

With `rating_history_policy: pre_event`, every rating used for a candidate at
time `t` must have timestamp strictly less than `t`. Thus earlier ratings in
the validation/test period can be available as external behavioral history.
Trust labels and held-out trust query outcomes do not construct these features.
The alternative `split_frozen` policy freezes rating history at the training
boundary for validation and the validation boundary for test.

All admissible rating records are observed auxiliary evidence in this variant;
the trust-event `observed_mask` and `p_missing` still apply to trust events.
This expands the model inputs. The candidate pairs, user-regime definitions,
and sampled MRR/AP/AUC rules are retained. Feature normalization is fitted on
training queries only, and rating/structure caches are fingerprinted.

The refinement also supports a pointwise `evidence_residual` logit path whose
output layer starts at zero. With `ema_decay > 0`, training maintains an
exponential moving average after optimizer steps; validation and saved best
checkpoints use those averaged model weights, while optimization continues
from the raw weights. The selected artifact is still a single model.

`val_macro_user_mrr` is an auxiliary checkpoint-selection score: the arithmetic
mean of the separately computed Observed-user and Unobserved-user validation
MRRs. It is distinct from pooled `val_overall_mrr`. Calibration and threshold
selection continue to use overall validation; the paper task results remain
separate. `resolved_config.json` records the effective run configuration.

Run the refinement on CUDA with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_main.py --config configs/epinions_70_15_15_causal_refined.yaml --device cuda
```

An optional `training.warm_start_checkpoint` initializes model weights before
training with a new optimizer. It does not resume optimizer or EMA state.
Use the corresponding seed's checkpoint when performing repeated runs.

## Recent activity and joint user-task selection

`configs/epinions_70_15_15_causal_recent.yaml` extends the rating variant with
each endpoint's rating counts over the previous 30, 180, and 365 days, the
fraction of ratings in the previous year, and time since its last rating.
The rating timestamp must be strictly earlier than the query timestamp.
These features distinguish current activity from activity accumulated years
earlier. `stationary_time_features` removes the original horizon-normalized
node ages and replaces horizon-normalized pair gaps with a fixed log-days
transform. Actual timestamps and memory gaps keep their original units.

The optional directed user embeddings learn source and target roles only for
users appearing in genuine positive training relations. Unknown users have
zero lookup contributions, including zero biases. Training randomly hides
known identities with `user_identity_dropout`; this does not change the
Observed-user/Unobserved-user masks. The reference users and embedding
parameters are saved in the checkpoint. This adds a learned identity path
to the model, so results must be identified as this experimental variant.

The rating builder also supports optional recent item/category profiles.
`rating_profile_dim` uses a fixed seeded item-hash basis plus category counts
over `rating_profile_window_days`; no basis is fitted on future interactions.
`use_rating_profiles` enables their directed logit path. Both are disabled in
the recent-activity configuration above.

`val_robust_target_ratio` selects one shared checkpoint for both user tasks.
It takes the minimum of each task's MRR divided by its configured target
over whole validation and its earlier/later chronological halves. Tied
timestamps stay together. Each diagnostic window has its own task-specific
negative pool; the original whole-validation and test MRR calculations keep
their original candidate pools and 100-negative sampling rule. A ratio at
least one means both targets were reached in all validation windows. This
is a stability check within validation; both halves have been used during
model development and are not independent holdouts. Test results from earlier
development have also already been inspected.

For validation comparisons without computing test results, run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_main.py --config configs/epinions_70_15_15_causal_recent.yaml --device cuda --validation-only
```

This writes `validation_user_regimes.csv`, `metrics.json`, the effective
configuration, and the evaluation protocol. Once a recipe is fixed, omit
`--validation-only` to report test for the selected checkpoint. Both user
tasks use the same model, validation calibrator, and threshold. All graph
layers and the three differentiable memory writers remain enabled in this
configuration. Numerical targets are selection settings, not guarantees
about unseen data or different seeds.

## Memory time units and untimestamped trust datasets

`model.memory_time_scale_seconds` converts native second-valued gaps before
memory decay, uncertainty discounting, and the update/reconnect gates. The
default `1.0` preserves older checkpoints' behavior. The corrected Epinions
configurations explicitly use `86400.0`, so rates are per day. With rates
`0.015 / 0.045 / 0.09`, the memory half-lives are approximately
46.2 / 15.4 / 7.7 days. Encoder time transforms do not change raw timestamps.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_main.py --config configs/epinions_70_15_15_memory_days.yaml --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_main.py --config configs/ciao_70_15_15_memory_days.yaml --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_main.py --config configs/ciaodvd_70_15_15_memory_days.yaml --device cuda
```

The Ciao configurations use their existing 70/15/15 random trust-link splits
and report all-user metrics. Native trust timestamps and gaps remain zero;
no temporal ordering is inferred from file rows or rating timestamps. Their
trust features are rebuilt in memory from training relations only, with the
current training edge removed from its own trust features. Existing static
rating context, candidates, splits, and node tables are retained. Both held-out
splits use the same training graph and memories; validation links do not enter
test history. Training-graph priming is differentiable and is rebuilt after
each optimizer step, so memory writers can learn without query writes.

Changing the memory time unit has no numerical effect when all gaps are zero.
These runs assess the restored static training path; they do not establish a
temporal-decay improvement on Ciao or CiaoDVD. The Epinions time-unit comparison
keeps the previous features and evaluation protocol, including the separately
identified same-timestamp issue in legacy training features.

## Static trust refinement for Ciao and CiaoDVD

The `*_static_refined.yaml` configurations add directed evidence from the
training trust graph: reciprocal relations, two-hop paths, common outgoing
and incoming neighbors, and unsaturated role-specific neighbor counts.
For a training query, its own relation is removed before these features are
computed, including self-loop handling. Held-out outcomes never build this
graph. The original static rating context is retained.

`static_bidirectional` uses separate operator banks for messages from incoming
and outgoing neighbors. The two aggregations remain separate in each node
update, preserving the direction of trust. This option requires the static
training-graph mode; it does not alter Epinions temporal propagation.
Both propagation directions use the existing evidence gates and context
mixture, and the memory writers remain differentiable.

The refinement also increases dropout and weight decay and reduces the user
lookup learning-rate multiplier. These settings were selected by all-user
validation MRR. Candidates, random splits, negative pools and the sampled
MRR/AP/AUC definitions are retained. Earlier test scores were already examined
during development, so new single-seed scores are exploratory results rather
than an independent confirmation or a five-seed mean.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_main.py --config configs/ciao_70_15_15_static_refined.yaml --device cuda
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_main.py --config configs/ciaodvd_70_15_15_static_refined.yaml --device cuda
```

Use `--validation-only` when comparing variants. The new graph evidence and
propagation path are model changes; paper descriptions and matched baseline
ablations must reflect them if these results are used in a manuscript.

## CiaoDVD directed diffusion experiment

`ciaodvd_70_15_15_diffusion.yaml` retains the released candidates, 70/15/15
split and 100-negative MRR calculation. Its candidate users remain the 4,658
users that participate in trust relations. Epinions and Ciao have 9,163 and
2,342 such users, respectively, covering their complete released node sets.
CiaoDVD's other 14,875 nodes are rating-only users; adding them to its negative
pool would define a different comparison. The accepted Ciao configuration is
unchanged.

The optional diffusion features sum directed random walks on the observed
training graph, separately for incoming and outgoing transitions. For each
training positive, an exact inverse update deletes its own relation and
renormalizes the affected transition row before reading its features.
Validation and test outcomes never construct these features. The predictor
also receives the product and absolute difference of the two graph
representations, alongside the original ordered source/target representations.

During training, each minibatch's candidate relations are excluded from the
adjacency used for graph priming, and 20% of the remaining historical
relations are randomly hidden. This is training regularization. Validation
and test both read the complete fixed training graph, without this dropout
or query writes. Graph priming and memory writers stay differentiable.
Query-edge exclusion requires `static_train` with `event_stream` minibatches.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python runners/run_main.py --config configs/ciaodvd_70_15_15_diffusion.yaml --device cuda --validation-only
```

Variants are compared by validation MRR before evaluating the selected
checkpoint on test. A requested MRR target is not a measured result. These
optional model changes do not modify the temporal Epinions configuration or
the accepted Ciao model.

The seed-42 development comparison selected a model at validation MRR 0.7465.
Its test MRR was 0.7323, compared with 0.7202 for the previous configuration;
test AP/AUC changed from 0.9755/0.9749 to 0.9797/0.9804. Observed-user MRR
changed from 0.7260 to 0.7357, while unobserved-user MRR changed from 0.3803
to 0.3731. The requested overall MRR of 0.90 was not achieved. The accepted
Ciao test MRR remains 0.6548. This is a single-seed exploratory comparison,
with earlier test results already known during development.
