# Aether HVAC Transfer

**Team Aether** · NeurIPS 2026 Smart Buildings Challenge, Round One

Code for the paper *Diagnosing the Transfer Gap of a Physics Prior Planner for HVAC Control*.

The repository contains the controller, the evaluation harness and the seeds of every
evaluation grid used in the paper. At run time the controller needs only NumPy and the Python
standard library. The weather oracle used in one ablation also reads YAML.

## Contents

| Path | Description |
|---|---|
| `aether/` | The controller: observation parser, per zone thermal prior, rule layer, planner, per zone offset, and a one step inversion (`inv.py`) used as a comparison |
| `aether/prior.json` | The frozen population prior (ten parameters per zone) |
| `harness/` | Rollouts on the starter kit simulator, scenario generation, tuning, sweeps, oracle analyses and paired bootstrap reporting |
| `pool/` | Seeds and split definitions of the generated buildings (held out set, stress grid and fresh grid) |
| `submission/` | Parameter files and the builder of the Kaggle submission notebook |
| `tests/`, `demo/data/` | Unit and integration tests, and three short replay recordings they use |

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/smart-buildings-2026/starter-kit external/starter-kit
git -C external/starter-kit checkout 95cea5b
uv sync
```

The simulator and scenario generator of the starter kit are used unmodified.

## Generated Buildings

The generated buildings are not stored in the repository. They are rebuilt deterministically
from the seeds in `pool/`.

```bash
uv run python -m harness.gen_pool --workers 8      # training and held out buildings
uv run python -m harness.gen_pool --stress         # stress grid, 48 scenarios
uv run python -m harness.gen_pool --rl_big         # fresh grid
```

## Running the Controller

Evaluate a configuration on a split or on scenario directories:

```bash
uv run python -m harness.run_parallel \
    --policies aether:mpc_off pid_water \
    --scenarios pool_test \
    --params submission/params_zc.json \
    --out results/eval --workers 16
```

| Parameter file | Configuration |
|---|---|
| `submission/params_zc.json` | Final planner configuration (arm `mpc_off`) |
| `submission/params_inv.json` | One step inversion (arm `inv`) |
| `submission/params_inv2.json`, `params_inv3.json` | Inversion with later refinements (arm `inv`) |

## Reproducing the Paper

In distribution evaluation on the held out and shipped sets:

```bash
uv run python -m harness.report run --splits pool_test shipped
uv run python -m harness.report tables
```

Stress grid comparisons over seven days (2016 steps):

```bash
SWEEP_ARM=mpc_off SWEEP_BASE=submission/params_zc.json SWEEP_SPLITS=stress \
    uv run python -m harness.sweep run 2016 final
SWEEP_ARM=inv SWEEP_BASE=submission/params_inv.json SWEEP_SPLITS=stress \
    uv run python -m harness.sweep run 2016 inversion
SWEEP_SPLITS=stress uv run python -m harness.sweep score 2016
```

Use 8640 steps for the thirty day runs. Variants are written as `name=key:value,key:value`
on top of the base parameter file. Setting `AETHER_BILLING_FIX=1` bills boiler gas at the
actual tank temperature, for analysis only.

Intervals are paired percentile bootstrap intervals over scenarios, and percent changes are
ratios of mean scores.

## Building the Submission Notebook

```bash
uv run python -m submission.notebook --params submission/params_zc.json --arm mpc_off \
    --out submission/aether_submission.ipynb
```

## Tests

The tests use the generated buildings, so run `harness.gen_pool --workers 8` first.

```bash
uv run pytest -m "not slow"
```

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
