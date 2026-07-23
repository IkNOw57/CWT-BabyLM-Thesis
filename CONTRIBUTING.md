# Contributing

This repository accompanies a master's thesis and is primarily shared for
transparency and reproducibility. Contributions are still welcome — bug
reports, documentation fixes, and small improvements are appreciated.

## Getting started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Note that the training scripts depend on a private `engine` package that
   is not distributed in this repository (see the main [README](README.md)).
   Contributions that don't require running full training jobs (docs, configs,
   preprocessing, tooling) can be developed and tested without it.

## Making changes

1. Create a topic branch: `git checkout -b fix/short-description`.
2. Keep changes focused — one logical change per pull request.
3. Match the existing code style (docstrings on scripts, `argparse` help text
   on every CLI flag).
4. Update the README if you change a script's CLI interface or add a new one.

## Submitting a pull request

- Describe what the change does and why.
- Link any related issue.
- Make sure shell scripts remain executable (`chmod +x`) and SLURM templates
  keep their placeholder-style documentation.

## Reporting issues

Please open a GitHub issue with:
- What you expected to happen.
- What actually happened (include the full error/traceback if applicable).
- Your environment (Python version, CUDA version, GPU model).

## Code of conduct

Be respectful and constructive. This project follows the spirit of the
[Contributor Covenant](https://www.contributor-covenant.org/).
