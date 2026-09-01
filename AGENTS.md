# DetGeo collaboration rules

GitHub is the shared project state between Codex and ChatGPT web. The repository must explain both the current code and how each reported result was produced.

## Experiment workflow

- Treat one meaningful, completed experiment as one Git commit.
- After running an experiment, update `EXPERIMENTS.md` in the same commit as its code or configuration changes.
- Record the date, goal, base/version, changed files, dataset and split, checkpoint, exact command, important parameters, results, baseline comparison, and conclusion.
- For ranking experiments, always report `Acc@0.25`, `Acc@0.5`, `Rescue@0.5`, `Degradation@0.5`, and net gain when available.
- State explicitly whether the test split was used. Prefer model selection on validation data and leave test untouched until the direction is fixed.
- Use concise commit messages such as `exp: residual cross-attention top5`.
- Push a completed experiment after its checks pass so GitHub remains the canonical shared state.

## Repository safety

- Never commit datasets, checkpoints, generated outputs, logs, credentials, SSH configuration, or machine-specific editor profiles.
- Keep paths and commands reproducible, but redact passwords, access tokens, and temporary cloud endpoints when they are not required to reproduce the experiment.
- Do not rewrite or invent historical results. Mark experiments completed before Git initialization as pre-history snapshots.
- Preserve the frozen DetGeo baseline unless an experiment explicitly says the detector is being fine-tuned.
