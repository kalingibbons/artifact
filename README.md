# artifact

Add a short description here!


## Description

A longer description of your project goes here...

## Installation

In order to set up the necessary environment:


1. create an environment `artifact` with the help of [conda],

   ``` none
   conda env create -f environment.yaml
   ```


2. activate the new environment with

   ``` none
   conda activate artifact
   ```


3. install `artifact` with:

   ``` none
   python setup.py install # or `develop`
   ```

    > Note: The folowing is optional and needed only once after `git clone`:


4. install several [pre-commit] git hooks with:

   ``` none
   pre-commit install
   ```

   and checkout the configuration under `.pre-commit-config.yaml`. The `-n,
   --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks
   temporarily.

5. install [nbstripout] git hooks to remove the output cells of committed
   notebooks with:

   ``` none
   nbstripout --install --attributes notebooks/.gitattributes
   ```

   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.


Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in
   `environment.yaml` and eventually in `setup.cfg` if you want to ship and
   install your package via `pip` later on.

2. Create concrete dependencies as `environment.lck.yml` for the exact
   reproduction of your environment with:

   ``` none
    conda env export -n artifact -f environment.lck.yml
   ```

   For multi-OS development, consider using `--no-builds` during the export.


3. Update your current environment with respect to a new `environment.lck.yml`
   using:

   ``` none
   conda env update -f environment.lck.yml --prune
   ```

## Project Organization

``` none
├── AUTHORS.rst             <- List of developers and maintainers.
├── CHANGELOG.rst           <- Changelog to keep track of new features and fixes.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── preprocessed        <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yaml        <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `python setup.py develop` to install for development or
|                              or create a distribution with `python setup.py bdist_wheel`.
├── src
│   └── dsproject_demo      <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

## Note

This project has been set up using PyScaffold 3.2.3 and the [dsproject
extension] 0.4. For details and usage information on PyScaffold see
https://pyscaffold.org/.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject