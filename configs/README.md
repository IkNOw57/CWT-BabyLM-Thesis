# Configs

Add your model/training configuration JSON files here. Each training script
reads one via `--config path/to/file.json`.

This snapshot of the repository does not ship example configs (they lived
alongside the private `engine` package). At minimum, a config should specify:

- **Preprocessing** (`preprocess.py`): `dataset_name`, `dataset_config`,
  `hf_tokenizer`, `max_seq_len`, `output`.
- **Pretraining** (`mlm_headless.py`, `gpt_headless.py`): training objective
  (headless/CWT vs. vanilla), optimizer settings, learning rate schedule, and
  any objective-specific hyperparameters (e.g. contrastive temperature for
  CWT) expected by `engine.tasks.pretraining`.
- **Fine-tuning** (`ft_gpt_headless.py`): learning rate, schedule, and number
  of steps for adapting a headless checkpoint's new output head.

Refer to `docs/thesis.pdf` (Chapter 3, Methodology) for the exact
hyperparameters used in the reported experiments.
