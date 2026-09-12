# ml/ — the training pipeline

Statistical downscaling: take a coarse weather field (HRRR, 3 km) plus fine-scale
covariates, and predict temperature at the resolution of the label data
(geohash-8, ~19 × 33 m).

This document has **three levels**. Read the one that matches what you are doing.

| Level | For | Answers |
|---|---|---|
| **[1. High](#level-1--high-what-this-is)** | anyone | What is this, what does it produce, why is it built this way |
| **[2. Medium](#level-2--medium-how-to-run-one)** | operators, and AI assistants helping them | How do I run an experiment, read the results, and change something |
| **[3. Low](#level-3--low-the-complete-reference)** | building a config from scratch | Every section, every field, every allowed value, mandatory vs optional |

**If you are an AI assistant helping someone write a config: go to
[Level 3](#level-3--low-the-complete-reference).** It is exhaustive and
authoritative — every value listed there was read out of the code, not remembered.

---

# Level 1 — High: what this is

## The problem

Vehicle-telematics data (Geotab) gives ambient temperature readings along road
networks, aggregated to geohash-8 cells per hour. HRRR gives a 3 km gridded forecast
of the same variable. HRRR is smooth and complete; the telematics data is sharp,
sparse, and road-biased.

The model learns to turn the coarse field plus local covariates — building height,
vegetation, elevation, surface temperature, solar geometry — into a fine-scale
temperature estimate.

## What a run produces

One command produces all of this:

```bash
python -m ml.pipeline train ml/experiments/ltdm_houston/lightgbm_spatial.yaml --name you
```

- **Metrics in absolute units** (°C), per split part, with a pass/fail gate
- **A model card** recording exactly what was trained, on what, with which transforms
- **The fitted model** plus every scaler and imputer needed to use it
- **Diagnostic figures** — where the error is, by hour, by covariate, in space
- **Heat maps** — the temperature field the model produces, beside its 3 km input
- **Everything published to MLflow as each stage finishes**, not batched at the end

## Five ideas that shape everything

**1. A config is the experiment.** Data, split, processing, model, and scoring are all
declared in one YAML file. Two configs sharing a base differ only where they say they
differ, so a metric difference between them is the thing you changed.

**2. Fitted state comes from `train` only.** Every mean, median, quantile and bin edge
is computed on the training part. This is enforced in code, not by convention — a
transform that tries to fit on validation data raises `LeakageError`.

**3. The split is the honest part.** A random row split on this data scores the model
on cells it trained on. Since neighbouring 33 m cells are nearly identical, that
measures interpolation and reports it as generalisation. The pipeline makes spatial
and temporal hold-outs first-class, and refuses to guess a default.

**4. Structure survives to the model.** Features stay in named groups (`static`,
`dynamic`) rather than being flattened into one matrix, so a two-tower network can
address them by name while a tree model flattens them itself.

**5. Nothing is silently dropped.** A split that would discard rows fails. An empty
part fails. A processing step that would lose more rows than declared fails. The
failures this pipeline is built to prevent are the ones that look like success.

---

# Level 2 — Medium: how to run one

## The eight stages

Every run goes through these in order. `prepare_data` owns 1–4; `run` owns 5–8.

```
1. READ       parquet -> a Frame (data + feature groups + target + group key)
2. FILTER     partition-level predicates, pushed down so unread files stay unread
3. SPLIT      carve into train / val / test by declared strategy
4. PROCESS    impute, scale, clip, encode — each fitted on `train` alone
5. BUILD      construct the model from `model.arch` or a custom file
6. FIT        train, streaming per-epoch metrics as they happen
7. SCORE      metrics per part, in absolute units, plus the gate
8. EXPORT     model card, weights, transforms, figures, heat maps
```

Two facts worth internalising:

- **Split runs before processing.** A processing step cannot influence the split. That
  is why `split:` appears above `processing:` in every config.
- **Everything up to stage 4 stays lazy on Dask.** The graph executes once, at a
  single materialisation point, with filtering and splitting already folded in.

## Vocabulary

| Term | Meaning |
|---|---|
| **Frame** | Data plus its target, feature groups, group key, and fitted state |
| **part** | A named subset of rows: `train`, `val`, `test`, or anything you declare |
| **splitter** | One strategy for claiming rows into a part (`temporal`, `spatial_blocks`, …) |
| **step** | One processing operation (`impute`, `scale`, `clip`, …) |
| **group** | A named set of feature columns a model can address (`static`, `dynamic`) |
| **group_key** | The column(s) identifying the unit a split must not cut through — `geohash` |
| **gate** | Declared thresholds; a run that misses them exits non-zero |

### What `group_key` actually does

It names the thing that must not appear in two parts at once. With
`group_key: [geohash]`, the pipeline can tell you whether a cell has rows in both
train and test — and whether that is a leak or the point.

The distinction matters and the pipeline knows it:

- A **temporal** split shares cells between parts *by design* — that is what makes it
  a test of generalising across time.
- A **ratio** split sharing cells is a **leak**: the same cell at 14:00 and 15:00 in
  two different parts measures memorisation.

So each splitter declares whether it cuts through groups, and only the ones that
shouldn't are checked.

## Running one

```bash
# Validate without touching data — always do this first
python -m ml.pipeline check ml/experiments/ltdm_houston/lightgbm_spatial.yaml

# See what the config resolves to after `extends`
python -m ml.pipeline describe ml/experiments/ltdm_houston/lightgbm_spatial.yaml

# Train. Without --name nothing is logged to MLflow; stages still go to disk.
python -m ml.pipeline train <config> --name you --output runs/my_run

# Override any field from the command line
python -m ml.pipeline train <config> --name you --set training.max_epochs=5

# Sweep: any list-valued training/model param becomes an axis
python -m ml.pipeline sweep <config> --name you
```

Exit codes: **0** pass, **1** error, **2** ran but failed the gate.

## Reading the results

While the run is still going, MLflow fills in stage by stage:

| Stage | When | What to look at |
|---|---|---|
| `data` | after the split | `split.md` — are the part fractions what you asked for? `drift.md` — is the test set *different* or just *harder*? |
| `target` | after the split | Do train and test have the same target distribution? If not, the score was never comparable |
| `nulls` | after processing | Any feature mostly null is being fed as a constant |
| `model` | before training | Parameter count, input widths, the architecture as built |
| `fit` | after training | The loss curve, and feature importance |
| `metrics` | after scoring | Everything, plus the gate and the heat maps |

**The three numbers that matter most on this problem:**

- **`test_mae`** — error in °C. The headline.
- **`test_sd_ratio`** — predicted spread ÷ actual spread. Below ~0.8 means the model
  is over-smoothing: it is winning on RMSE by predicting the mean. **RMSE hides this
  completely.**
- **train vs test gap** — a spatial hold-out on this data typically shows a large gap.
  Check `drift.md` before blaming the model: a spatial split is *supposed* to cover
  different ground.

## Changing something

| I want to… | Change |
|---|---|
| use less data | add `{op: sample, fraction: 0.01}` as the **first** processing step |
| hold out differently | rewrite `split.parts` — see [3.7](#37-split) |
| add a feature | add the column to a group under `data.groups` |
| swap the model | change `model.kind` and its block — see [3.10](#310-model) |
| draw the temperature field | add `testing.heatmap` with an `aoi` — see [3.12](#312-testing) |
| train longer | `training.max_epochs`, `training.early_stopping` |

## Inheritance

```yaml
extends: _base.yaml          # resolved relative to this file
```

Dicts merge key by key; **lists are replaced, not concatenated**. So overriding
`processing.steps` means *these steps*, not "the base's plus these" — which is right,
because step order is significant.

> **A trap worth knowing.** Because dicts merge, a child's
> `split: {parts: {test: {spatial_blocks: ...}}}` lands *beside* a parent's
> `test: {temporal: ...}` rather than replacing it, and a part with two strategy keys
> is refused. When you change a split strategy, write the config standalone. Both
> `lightgbm_spatial.yaml` and `two_tower_spatial.yaml` do this, and say so at the top.

---

# Level 3 — Low: the complete reference

Everything below was extracted from `ml/pipeline/config/schema.py`, the registry, and
each component's constructor. If this document and the code disagree, the code wins —
but they were checked against each other when this was written.

## 3.1 How a config is validated

Three things happen at load:

1. **Unknown keys are refused**, with the allowed set named. There is no silent
   `.get(key, default)` anywhere in config handling.
2. **Renamed keys are refused with their replacement named.** `mlflow:` tells you to
   use `tracking:`; `evaluation:` tells you `testing:`; `train_split.feature_cols`
   tells you `data.groups`.
3. **Path-valued keys are resolved against the declaring file** — see [3.2](#32-paths).

`python -m ml.pipeline check <config>` runs all of this and reads no data.

### Top-level sections

| Section | Required | Purpose |
|---|---|---|
| `experiment` | recommended | Experiment name; falls back for `tracking.experiment_name` |
| `extends` | optional | Path to a parent config |
| `run` | optional | Seed, name, notes, tags |
| `tracking` | optional | Where results go |
| `data` | **yes** | Source, target, groups, engine |
| `filter` | optional | Partition-level predicates |
| `split` | **yes** | How rows become parts |
| `processing` | optional | Ordered transform steps |
| `model` | **yes** | Which model, and how it is shaped |
| `training` | torch only | Batch size, epochs, optimiser |
| `testing` | optional | Tolerances, gate, strata, figures, heat maps |

### The four mandatory keys

```
data.source.path
data.target
split.parts
model.kind
```

Nothing else is required. Everything with a defensible default has one; anything
whose default would be a guess about the science is mandatory instead.

## 3.2 Paths

These keys are resolved **relative to the config file that declares them**:

```
file          model.arch.custom.file       — a custom model .py
aoi           testing.heatmap.aoi          — the heat-map polygon
geometry      split.parts.*.geometry       — a GeoJSON hold-out
model_dir, script_path
```

`data.source.path` is **not** in that list: it resolves against the **repo root**,
because a dataset location (`projects/houston-gh8/data/merged/final`) is a
repo-relative fact shared by every config that reads it.

Absolute paths pass through unchanged. `~` is expanded.

## 3.3 `run`

| Field | Type | Default | Notes |
|---|---|---|---|
| `seed` | int | `42` | Seeds numpy, torch, and every splitter that doesn't override it |
| `name` | str | — | MLflow run name; the CLI's `--run-name` wins |
| `notes` | str | — | Becomes the run's description in MLflow |
| `tags` | map | — | Arbitrary MLflow tags |

## 3.4 `tracking`

| Field | Type | Default | Allowed |
|---|---|---|---|
| `experiment_name` | str | falls back to `experiment` | Required to log at all |
| `tracking_uri` | str | absent | see table below |
| `backend` | str | `mlflow` | `mlflow`, or `none`/`off`/`false` to disable |
| `register_model` | bool | `false` | Register the export as a model version |
| `registered_name` | str | experiment name | Only with `register_model` |
| `register_alias` | str | — | e.g. `champion` |
| `enable_system_metrics` | bool | `true` | CPU/GPU/memory sampling |

### `tracking_uri` values

| Value | Behaviour |
|---|---|
| absent, or `local` | SQLite at `mlruns/mlflow.db`. No server, no network. |
| `localhost` | Starts a local `mlflow server` on 127.0.0.1:5000 (~9 s) and reuses one already running |
| `http://host:5000` | That server |
| `sqlite:///path.db` | That SQLite file |
| a directory path | A file store there (with MLflow's own deprecation warning) |

`MLFLOW_TRACKING_URI` in the environment overrides the config — the server is a
property of the environment, not of the experiment.

## 3.5 `data`

### `data.source` (required)

| Field | Type | Required | Notes |
|---|---|---|---|
| `path` | list[str] or str | **yes** | Directories or globs, resolved against the repo root. `s3://` works. |
| `format` | str | no (`parquet`) | `parquet` |
| `partitioning` | map | no | `{scheme: hive, keys: [year, month, day]}` |

Declaring `partitioning` matters: with it, `filter` predicates on those keys prune
whole files before any bytes are read.

> **Hive keys are strings.** A directory named `year=2023` yields the *string*
> `"2023"`. So `{column: year, in: ["2023"]}` matches and `in: [2023]` does not. The
> integer equivalents in the table are usually named `year_n`, `month_n`, `day_n`.

### `data.dataset` (optional but recommended)

```yaml
dataset: {name: us-gh8, version: 1}
```

Pins a spec in `data/datasets/<name>/dataset.yaml`. With it, group entries may name
feature groups or sources (`worldcover`) and get expanded to columns. The pinned
`version` is what makes a run reproducible.

### `data.target` (required)

A single column name. The label. It is excluded from every feature group
automatically — a target that is also a feature is a model predicting its own input.

### `data.groups` (required in practice)

```yaml
groups:
  static:  [longitude, latitude, built_fraction, elevation]
  dynamic: [t2m_2m, u10_10m, hour_sin, hour_cos]
```

Each entry may be:

- a literal column name;
- a **feature-group name** from the dataset spec (`worldcover`, `hrrr`);
- a **source name**, expanded to that source's declared columns.

Group names are free-form, with two conventions the built-in architectures use:
`static` (constant per cell) and `dynamic` (varies with time).

> A group may name a column that a **processing step creates** — `hour_sin` does not
> exist in any dataset; a `cyclical` step makes it. The validator knows which columns
> the configured steps produce and does not reject those.

### `data.group_key` (optional)

```yaml
group_key: [geohash]
```

The unit a split must not cut through. Required for leakage checks and for
`group_by:` on a splitter. Read from the source automatically even when it is not a
feature.

### `data.columns` (optional)

Extra columns to read that are **not** model inputs — needed by a processing step, a
split predicate, or a heat map.

```yaml
columns: [geohash, year_n, month_n, day_n, hour, t2m_2m]
```

### `data.engine` (optional)

| Field | Type | Default | Allowed / notes |
|---|---|---|---|
| `name` | str | `auto` | `auto`, `dask`, `pandas`. `auto` picks Dask past the thresholds below |
| `cluster` | str | `none` | `none` (threads), `local` (processes), or a scheduler address |
| `workers` | int or `auto` | `auto` | Processes, when `cluster: local` |
| `threads_per_worker` | int | `2` | |
| `partition_size` | str | `128MB` | Target partition size |
| `blocksize` | str | `256MB` | Read block size |
| `memory_limit` | str | `auto` | Per worker |
| `dask_threshold_rows` | int | `8000000` | Above this, `auto` picks Dask |
| `dask_threshold_bytes` | int | — | Alternative threshold |
| `spill_to_disk` | bool | `true` | |
| `shuffle` | str | `tasks` | `tasks` or `disk` |

**Threads or processes.** Threads are the default because parquet decoding releases
the GIL. Choose `local` (processes) when per-row Python work dominates — string
handling, `map_partitions` with a Python function. On the 1.5 M-row Houston table
processes measured 52 s against 41 s for threads, so processes are not free.

### `data.window` (optional — sequence models only)

| Field | Type | Default | Notes |
|---|---|---|---|
| `lookback` | int | `0` | Timesteps of history per sample. `0` disables windowing |
| `horizon` | int | `0` | Steps ahead to predict. `0` = predict the last input row |
| `order_by` | str or list | — | Column(s) defining time order. **Required** when `lookback > 0` |
| `stride` | int | `1` | Step between window starts |
| `allow_gaps` | bool | `false` | When false, a window with a missing timestep is dropped |

Windows are built **per group** (`group_key`), so a window never spans two cells.

## 3.6 `filter`

A list of predicates applied at read time. Pushed down to the parquet reader where
possible, which means matching files are never opened.

```yaml
filter:
  - {column: year, in: ["2023"]}
  - {column: month, in: ["04", "05", "06"]}
  - {column: built_fraction, gte: 0.1}
```

**Operators:** `equals`, `in`, `not_in`, `gte`, `lte`, `gt`, `lt`.

Random strategies (`ratio`, `spatial_blocks`) are **refused** here — a filter runs
before the split, so a random filter would change what the split sees run to run.

## 3.7 `split`

```yaml
split:
  parts:
    test:  {spatial_blocks: {size_km: 25, fraction: 0.15, seed: 42}}
    val:   {spatial_blocks: {size_km: 25, fraction: 0.15, seed: 1337}}
    train: {remainder: true}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `parts` | map | **required** | Ordered: parts are carved in declared order |
| `report` | bool | `true` | Print the split summary |
| `require_full_coverage` | bool | `true` | Every row must land somewhere, or the run fails |
| `require_group_disjoint` | bool | `false` | Check *every* part for shared groups |

**Sequential carving.** Parts are claimed in order, each from what is left. So `test`
declared first gets first refusal, and `remainder: true` — which must be **last** —
takes the rest.

There is **no default split**. A missing `split.parts` is an error.

### Splitters

Each part names exactly **one** strategy key.

#### `spatial_blocks` — whole geographic blocks

```yaml
test: {spatial_blocks: {size_km: 25, fraction: 0.15, seed: 42}}
```

| Field | Required | Notes |
|---|---|---|
| `size_km` | **yes** | Block edge in km. No default: this is the correlation length of your target |
| `fraction` | **yes** | Share of blocks to hold out, in (0, 1] |
| `seed` | no | Falls back to `run.seed` |
| `lon`, `lat` | no | Default `longitude`, `latitude` |

Blocks are corrected for latitude: a degree of longitude is 111.32 km × cos(lat), so
across CONUS a fixed-degree grid would make northern blocks ~28% narrower and the
hold-out systematically easier up north.

**Also exports GeoJSON** to `stages/data/<part>_blocks.geojson` — a dissolved
MultiPolygon plus one polygon per block, for dropping onto a map.

#### `temporal` — a named period

```yaml
test: {temporal: {column: month, in: [12]}}
test: {temporal: {column: date, from: "2024-06-01"}}
test: {temporal: {column: date, between: ["2024-06-01", "2024-08-31"]}}
```

Predicates: `in`, `before`, `from`, `between`. Shares groups between parts by
design, so no leakage check is applied.

#### `ratio` — a random share

```yaml
val: {ratio: 0.15, group_by: [geohash], seed: 42}
```

| Field | Required | Notes |
|---|---|---|
| `ratio` (or `fraction`) | **yes** | Share of rows, or of groups when `group_by` is set |
| `group_by` | no | Sample **whole groups**. Without it the same cell can land in two parts |
| `seed` | no | Falls back to `run.seed` |
| `allow_group_split` | no | `true` to say a shared group is deliberate |

#### `geometry` — a polygon from a file

```yaml
test: {geometry: "assets/east.geojson", mode: within}
test: {geometry: ["assets/a.geojson", "assets/b.geojson"]}
```

`mode`: `within` (default) or `intersects`. Needs shapely ≥ 2.0.

#### `column` — a predicate on any column

```yaml
test: {column: {name: city, equals: houston}}
```

Operators as in [3.6](#36-filter).

#### `all_of` / `any_of` — combinators

```yaml
test:
  all_of:
    - {temporal: {column: month, in: [7, 8]}}
    - {geometry: "assets/east.geojson"}
```

#### `remainder` — everything left

```yaml
train: {remainder: true}
```

Must be declared last. `remainder: false` is refused — omit the key instead.

## 3.8 `processing`

An **ordered** list of steps. Order is significant and steps are not reordered.

```yaml
processing:
  steps:
    - {op: sample, fraction: 0.001, seed: 42}
    - {op: cyclical, columns: [hour], period: 24}
    - {op: impute, columns: [lst], strategy: median, add_indicator: true}
    - {op: clip, columns: [temperature], lower_quantile: 0.001, upper_quantile: 0.999}
    - {op: drop_nulls, max_fraction: 0.35}
    - {op: scale, columns: {group: static}, method: robust}
```

**Every fitted statistic comes from the `train` part alone.** Attempting otherwise
raises `LeakageError`.

### Selecting columns

Steps taking `columns` accept:

```yaml
columns: [a, b, c]           # explicit
columns: {group: static}     # every column in one group
columns: {groups: [a, b]}    # several groups
columns: target              # the target column
```

### The eleven steps

#### `sample` — take a subset of rows

| Field | Default | Notes |
|---|---|---|
| `fraction` | — | In (0, 1]. Exactly one of `fraction` / `n` |
| `n` | — | Exact row count. Needs a materialised frame |
| `stratify_by` | — | Covariate columns; draws equally per stratum |
| `bins` | `5` | Bins for numeric strata |
| `bin_method` | `uniform` | `uniform` or `quantile` |
| `seed` | `42` | |
| `parts` | `[train]` | Which parts to sample |

**`bin_method` must default to `uniform`.** Equal-*frequency* (`quantile`) bins each
hold the same number of rows, so drawing equally per stratum reproduces the input
distribution exactly and stratification does nothing — a silent no-op.

**Put `sample` first** when using it for scale. With `fraction` and no `stratify_by`
it stays lazy and folds into the Dask graph, so the full data is never assembled.

#### `impute` — fill nulls

| Field | Default | Allowed |
|---|---|---|
| `columns` | **required** | |
| `strategy` | `median` | `median`, `mean`, `constant`, `most_frequent` |
| `value` | — | **Required** when `strategy: constant` |
| `add_indicator` | `false` | Emits `<col>_missing` (1 where the source was null) |

`add_indicator` is how "unknown" and "zero" stay distinguishable. The indicator
column can be named in a group and used as a `stratify_by` stratum.

#### `scale` — normalise

| Field | Default | Allowed |
|---|---|---|
| `columns` | **required** | |
| `method` | `standard` | `standard`, `minmax`, `robust`, `quantile` |
| `sample_rows` | `500000` | Rows used to fit |

Use `robust` for heavy-tailed columns (building volume, height) — with `standard`, one
downtown cell sets the variance for the whole country. Trees do not need scaling;
gradient descent does.

#### `clip` — bound values

| Field | Notes |
|---|---|
| `columns` | **required** |
| `lower`, `upper` | Absolute bounds |
| `lower_quantile`, `upper_quantile` | Quantile bounds, fitted on `train` |

#### `drop_nulls` — remove rows still carrying nulls

| Field | Default | Notes |
|---|---|---|
| `columns` | all features | |
| `max_fraction` | `0.5` | Fails the run if more than this share would be dropped |

State `max_fraction` deliberately: it is what makes a newly-sparse covariate fail
loudly instead of quietly shrinking the training set.

#### `cyclical` — periodic encoding

| Field | Default | Notes |
|---|---|---|
| `columns` | **required** | |
| `period` | **required** | e.g. `24` for hour, `365` for day-of-year, or `day_of_year` |
| `date_column` | — | For `period: day_of_year` from a real date column |
| `prefix` | column name | Output naming |

Emits `<col>_sin` and `<col>_cos`, so 23:00 and 00:00 are adjacent rather than 23
apart.

#### `derive` — a new column from an expression

```yaml
- {op: derive, name: day_of_year, expr: "(month_n - 1) * 30.44 + day_n"}
- {op: derive, name: dz, expr: "orog - elevation", group: static}
```

`expr` goes to `DataFrame.eval`: column names and arithmetic, not arbitrary Python.
`group` adds the new column to a feature group; omit it to create the column without
exposing it as a feature.

#### `residual` — model a difference

```yaml
- {op: residual, target: temperature, baseline: t2m_2m}
```

| Field | Default | Notes |
|---|---|---|
| `target` | **required** | |
| `baseline` | **required** | The coarse field to subtract |
| `name` | `<target>_residual` | |
| `keep_baseline` | `false` | Keep the baseline as a feature too |

Metrics automatically add the baseline back, so scores stay in absolute °C. This is a
modelling choice: the residual keeps the target range small and degrades to
interpolation rather than extrapolating during an unprecedented event.

#### `one_hot` — indicator columns

| Field | Default | Allowed |
|---|---|---|
| `columns` | **required** | |
| `categories` | fitted | |
| `drop_source` | `true` | |
| `unseen` | `zero` | `zero`, `error`, `other` |

#### `ordinal` — integer codes

| Field | Default | Notes |
|---|---|---|
| `columns` | **required** | |
| `unseen` | `-1` | Code for a category not seen in training |
| `categories` | fitted | |

#### `weight` — per-row sample weights

| Field | Default | Allowed |
|---|---|---|
| `column` | — | Exactly one of `column` / `expr` |
| `expr` | — | An expression |
| `normalize` | `mean` | `mean`, `sum`, or `null` |
| `clip_max` | — | Bound the largest weight |

Needs `training.sample_weight: true` for torch models.

## 3.9 Order of the sections in a config

Write them in execution order. This is convention, not enforcement, but it prevents a
real misreading:

```yaml
experiment / run / tracking     # metadata
data:                           # what is read
filter:                         # what is kept
split:                          # how it is divided     <- BEFORE processing
processing:                     # how it is transformed
model:                          # what is fitted
training:                       # how it is fitted
testing:                        # how it is judged
```

## 3.10 `model`

| Field | Applies to | Notes |
|---|---|---|
| `kind` | all | **Required.** `torch`, `sklearn`, `xgboost` |
| `params` | all | Passed to the estimator or architecture |
| `estimator` | `sklearn` | **Required.** Dotted path, e.g. `lightgbm.LGBMRegressor` |
| `arch` | `torch` | **Required.** Architecture — see below |
| `loss` | `torch` | `{name: ..., params: {...}}` |
| `num_boost_round` | trees | |
| `early_stopping_rounds` | trees | |
| `streaming` | `xgboost` | `true` (default) feeds the matrix in chunks |
| `max_rows` | all | Cap on training rows |
| `cache_prefix` | all | Cache location for prepared matrices |

### `kind: sklearn`

Wraps any estimator with `fit`/`predict`.

```yaml
model:
  kind: sklearn
  estimator: lightgbm.LGBMRegressor
  early_stopping_rounds: 50
  params:
    n_estimators: 2000
    learning_rate: 0.05
    num_leaves: 127
    verbose: -1
```

LightGBM gets an eval set automatically when a `val` part exists, and its
per-iteration scores become the loss curve.

### `kind: xgboost`

Native XGBoost, with `DataIter` streaming so a second full copy of the matrix is
never held beside the frame.

```yaml
model:
  kind: xgboost
  streaming: true
  num_boost_round: 2000
  early_stopping_rounds: 50
  params:
    objective: reg:squarederror
    max_depth: 10
    tree_method: hist
```

### `kind: torch` — declared towers

```yaml
model:
  kind: torch
  arch:
    towers:
      site:
        inputs: {group: static}
        embed: {numeric: piecewise_linear, bins: 48}
        blocks: [{mlp: {dims: [256, 128], activation: gelu, dropout: 0.1}}]
        output_dim: 64
      conditions:
        inputs: {groups: [dynamic, position]}
        embed: {numeric: periodic, k: 8}
        blocks: [{mlp: {dims: [128, 64], activation: gelu}}]
        output_dim: 32
    combine: {op: gated}
    head: {type: gaussian, hidden: [64]}
  loss:
    name: gaussian_nll
    params: {warmup_steps: 500}
```

**Tower fields**

| Field | Notes |
|---|---|
| `inputs` | `{group: name}` or `{groups: [a, b]}` |
| `embed` | Numeric embedding — see below |
| `blocks` | List of `{mlp: {dims, activation, dropout}}`. Default `[{mlp: {dims: [128]}}]` |
| `output_dim` | Bottleneck width. For a static tower this is what gets cached per cell |
| `lookup` | Optional learned table keyed by `group_key` |

**`embed.numeric` values**

| Value | Params | Use when |
|---|---|---|
| `linear` (default) | `d_embedding` (8) | Baseline |
| `piecewise_linear` | `bins` (48), `d_embedding` (8) | Skewed or saturating features. Bin edges are **fitted on train** |
| `periodic` | `k` (8), `sigma` (0.1) | Cyclic inputs (hour, solar geometry) |
| `passthrough` / `none` | — | Raw scalars |

**`combine.op`:** `gated` (A(conditions) × P(site) + I(both), both terms bounded) or
`concat`.

**`head.type`**

| Type | Output | Pair with loss |
|---|---|---|
| `point` (default) | `{mean}` | `mse`, `mae`, `huber` |
| `gaussian` | `{mean, log_variance}` | `gaussian_nll` |
| `quantile` | one output per quantile | `pinball` |

**Losses:** `mse`, `mae`, `huber` (`delta`, 1.0), `gaussian_nll` (`warmup_steps`,
`full`), `pinball` (`quantiles`).

> With `gaussian_nll`, set `warmup_steps`. A variance head fitted before the mean is
> any good learns to explain the mean's error as noise and never recovers.

### `kind: torch` — a custom model file

```yaml
model:
  kind: torch
  arch:
    custom:
      file: models/my_net.py      # relative to THIS config
      name: MyNet
      params: {hidden: 256}
    head: {type: gaussian}        # optional; omit if the module returns a dict
```

Your class receives keyword arguments and must accept `**kwargs`:

```python
class MyNet(nn.Module):
    def __init__(self, *, widths, seq_len, groups, target, n_keys, seed,
                 hidden=256, **kwargs):
        # widths:  {"static": 20, "dynamic": 23}   columns per group
        # seq_len: timesteps per sample (1 when unwindowed)
        # groups:  {"static": ["built_fraction", ...], ...} ordered column names
        super().__init__()

    def forward(self, inputs, keys=None):
        # inputs: {group: tensor}. (batch, features) or (batch, seq_len, features)
        # return a tensor (the configured head consumes it)
        # or a dict {"mean": ...} / {"mean": ..., "log_variance": ...}
```

`**kwargs` is required: a newly injected key must not break existing files. Declare
`self.output_dim` if you return a tensor, so the head can be sized without probing.

## 3.11 `training` (torch only)

| Field | Default | Notes |
|---|---|---|
| `batch_size` | — | |
| `max_epochs` | — | |
| `early_stopping` | — | Epochs of no improvement before stopping |
| `optimizer` | — | `{name: AdamW, params: {lr: 0.001, weight_decay: 0.01}}`. `Adam`, `AdamW`, `SGD`, `RMSprop` |
| `scheduler` | — | `{name: ReduceLROnPlateau, params: {...}}`. Also `CosineAnnealingLR`, `OneCycleLR`, `StepLR` |
| `precision` | `32` | `32`, `16-mixed`, `bf16-mixed` |
| `gradient_clip_val` | — | |
| `accumulate_grad_batches` | `1` | |
| `num_workers` | `0` | DataLoader workers. Keep at 0 on Windows |
| `prefetch_factor` | — | |
| `sample_weight` | `false` | Use weights from a `weight` step |
| `accelerator` | `auto` | `auto`, `gpu`, `cpu` |
| `devices` | `auto` | |
| `checkpoint_every` | — | |

Setting any of these on a non-torch `kind` is refused rather than ignored.

## 3.12 `testing`

| Field | Default | Notes |
|---|---|---|
| `accuracy_tolerances` | — | e.g. `[0.5, 1.0, 2.0]` → `within_0.5` etc., as percentages |
| `thresholds` | — | The gate: `{test_mae: 2.0}` |
| `require_test_split` | `true` | A declared threshold with no test part **fails** |
| `stratify_by` | — | Per-stratum breakdown |
| `metrics` | all | Restrict which are computed |
| `figures` | `true` | Render diagnostic figures |
| `heatmap` | absent (off) | The gridded temperature field |

### `stratify_by`

```yaml
stratify_by:
  hour: hour                                      # categorical, as-is
  built: {column: built_fraction, bins: 4}        # numeric, binned
  lst_missing: lst_missing                        # an impute indicator
```

Published as a **table** in `stages/metrics/metrics.md` and in `metrics.json` — not
as metrics. The cross product (parts × strata × bands × stats) came to ~800 scalar
entries and buried the 30 real metrics; a per-hour curve is a shape, and it is drawn
in `error_by_hour.png`.

### `heatmap`

```yaml
heatmap:
  aoi: ../../assets/houston.json     # REQUIRED, resolved against this config
  part: test                          # default `test`
  when: {day_n: 1, hour: 14}          # omit for the best-covered hour
  raster_size: 256                    # SSIM raster only; the map stays polygons
```

**`aoi` is mandatory** when `heatmap` is present. Without it the drawn boundary would
be "wherever the split's rows happened to fall", which reads as a claim about the area
rather than about the data. Cells are clipped to the polygon, so the map takes the
AOI's own silhouette.

Produces:

- **`heatmap_predicted.png`** — the model's field, one filled polygon per cell
- **`heatmap_comparison.png`** — HRRR at its native 3 km beside the model's output, on
  **one shared colour scale**

And four metrics: `heatmap_r2`, `heatmap_mae`, `heatmap_cells`, `heatmap_ssim`.

> **`heatmap_r2` is not `test_r2`.** It covers only the cells drawn at one timestamp.
>
> **`heatmap_ssim`** is structural similarity between the model's field and its HRRR
> input. 1.0 means the output reproduced the coarse field at finer spacing; lower
> means added fine-scale structure. **Noise also lowers it**, so read it beside
> `test_sd_ratio`.
>
> **Off by default.** At geohash-8, a CONUS-wide AOI is ~10⁸ polygons.

## 3.13 Metrics reference

Per part, in the target's absolute units:

| Metric | Meaning |
|---|---|
| `mae` | Mean absolute error |
| `rmse` | Root mean squared error |
| `bias` | Mean signed error. Non-zero means systematic offset |
| `r2` | Coefficient of determination |
| `sd_ratio` | **predicted spread ÷ actual spread.** Below ~0.8 = over-smoothing |
| `median_abs_error` | Robust central error |
| `p95_abs_error` | Tail error |
| `within_<t>` | Percentage within tolerance `t`, from `accuracy_tolerances` |
| `coverage_95` | Share inside the 95% interval (Gaussian head only) |

Per-epoch, streamed live: `epoch_index`, `epoch_train_loss`, `epoch_val_loss`,
`epoch_lr`, `epoch_best_so_far`, `epoch_improved`, `epoch_stalled`.

## 3.14 What a run writes

```
runs/<name>/
├── model/               weights or model.joblib, features.json
├── transforms/          every fitted scaler / imputer / encoder
├── figures/             scatter_density, error_map, residual_vs_covariate,
│                        error_by_hour, history
├── model_card.yaml      what was trained, on what, with which transforms
├── metrics.json         every metric, including all per-stratum breakdowns
├── config.yaml          the resolved config
└── stages/              per-stage reports, published to MLflow as each finishes
    ├── data/            split.md, columns.json, processing.md, drift.md,
    │                    <part>_blocks.geojson, split_composition.png,
    │                    covariate_drift.png, correlation_heatmap.png,
    │                    row_density.png
    ├── target/          target.md, target_by_part.png
    ├── nulls/           nulls.md, null_shares.png
    ├── model/           model.md, model_config.json, training_config.json
    ├── fit/             history.json, importances.md, history.png,
    │                    feature_importance.png
    └── metrics/         metrics.md, metrics_by_part.png, error_vs_prediction.png,
                         heatmap_predicted.png, heatmap_comparison.png
```

## 3.15 Common failures and what they mean

| Message | Cause |
|---|---|
| `a split part needs exactly one strategy key, got [...]` | `extends` merged your strategy beside the parent's. Write the config standalone |
| `part(s) [...] claimed no rows` | A predicate matched nothing. Check values actually present — hive keys are strings |
| `N rows were claimed by no part` | Declare a `remainder` part or set `require_full_coverage: false` |
| `groups appear in more than one part` | Add `group_by: [geohash]`, or `allow_group_split: true` if deliberate |
| `LeakageError` | A transform tried to fit on a non-train part |
| `drop_nulls would drop X > max_fraction` | A sparse covariate. Impute it or raise the bound deliberately |
| `is not a recognised key` | Typo, or a renamed key — the message names the replacement |
| `testing.heatmap needs 'aoi'` | Add the polygon path |
| `model.estimator is required for kind: sklearn` | Add the dotted estimator path |
| `model.arch.custom.file ... does not exist` | Path is relative to the declaring config |

## 3.16 A complete minimal config

```yaml
experiment: my_experiment

run: {seed: 42}

tracking:
  experiment_name: my_experiment
  tracking_uri: local

data:
  dataset: {name: us-gh8, version: 1}
  source:
    path: ["s3://fg-train-data/gh8_data_sources/v2/us"]
    format: parquet
    partitioning: {scheme: hive, keys: [year, month, day]}
  target: temperature
  groups:
    static:  [longitude, latitude, built_fraction, elevation]
    dynamic: [t2m_2m, u10_10m, hour_sin, hour_cos]
  group_key: [geohash]
  columns: [geohash, month_n, day_n, hour]
  engine: {name: dask, cluster: none, partition_size: 256MB}

filter:
  - {column: year, in: ["2023"]}
  - {column: month, in: ["04"]}

split:
  parts:
    test:  {spatial_blocks: {size_km: 25, fraction: 0.15, seed: 42}}
    val:   {spatial_blocks: {size_km: 25, fraction: 0.15, seed: 1337}}
    train: {remainder: true}

processing:
  steps:
    - {op: sample, fraction: 0.001, seed: 42, parts: [train, val, test]}
    - {op: cyclical, columns: [hour], period: 24}
    - {op: drop_nulls, max_fraction: 0.35}

model:
  kind: sklearn
  estimator: lightgbm.LGBMRegressor
  early_stopping_rounds: 50
  params: {n_estimators: 1000, learning_rate: 0.05, verbose: -1}

testing:
  accuracy_tolerances: [0.5, 1.0, 2.0]
  thresholds: {test_mae: 2.5}
  stratify_by: {hour: hour}
```

## 3.17 Extending the pipeline

Every extension point is a registry. Register a class and it is usable from a config
immediately; an unknown name produces an error listing what is available.

```python
from ml.pipeline.registry import register

@register("step", "my_transform")
class MyTransform(Step):
    op = "my_transform"
    def apply(self, frame): ...

@register("splitter", "my_split")
class MySplit:
    def claim(self, frame, available): ...
```

Kinds: `step`, `splitter`, `model`, `loss`.

## 3.18 Setup, layout, tests

```bash
conda activate ml-lab
pip install -e ./common
python -m pytest ml/tests/pipeline -q        # 355 tests
```

Keep `torch` and `torchvision` versions in lockstep — a mismatched `torchvision`
breaks `import lightning` with an unrelated-looking error.

```
ml/
├── pipeline/            the pipeline
│   ├── config/          loading, validation, schema
│   ├── data/            frame, source, filters, engine, split/, steps/, window
│   ├── models/          sklearn_, xgboost_, torch_/, custom
│   ├── testing/         figures, heatmap, stage_reports
│   ├── metrics.py       absolute-unit metrics and the gate
│   ├── reporting.py     per-stage MLflow publishing
│   ├── run.py           the eight stages
│   └── cli.py           train / check / describe / sweep
├── experiments/         one directory per experiment
│   ├── ltdm_houston/    single city, HRRR residual target
│   └── ltdm_gh8/        CONUS, absolute temperature target
└── tests/pipeline/      the test suite
```
