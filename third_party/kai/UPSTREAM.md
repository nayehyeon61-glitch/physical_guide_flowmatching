# Vendored: KAI-Atmos

- Source: `KAI-Atmos-main.zip` (project files), upstream commit `b1fb02d619949d4d12e771adfc2bc706d262b97d`
- License: BSD 3-Clause, Copyright (c) 2025 MetaEarth-Lab (see `LICENSE`)
- PG-CFM code imports `src/utils.py` by path (`pgcfm/data.py`).

## Local changes
- `src/inference.py` `_rollout_one_ic`: upstream stored day 2 at `lead_time=1` and skipped
  day 1 (each lead was one day ahead of its label). Now lead `i` stores the day-`i`
  forecast. `src/evaluator.py` uses a one-behind pointer and was already correct, so the
  RMSE/ACC numbers it reports are unaffected. Covered by `tests/test_kai_vendored.py`.
