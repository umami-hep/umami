# Development

If you wan to get started with umami you can pick an issue and work on it, well suited
are the issues labeled with [`good-first-issue`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/issues?label_name%5B%5D=good-first-issue) which can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/issues?scope=all&state=opened&label_name[]=good-first-issue). Please tell us if you are working on an issue and create already in the beginning a merge request marked as `Draft`. This helps us to see who is working at what.


Please follow the good coding practices which are also summarised [here](good_practices_code.md).

## Test suite

Umami's development uses unit tests to ensure it is working. The unit tests are located in [`umami/tests`](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/tree/master/umami/tests).
You can run the unit tests with the command:
```bash
pytest ./umami/tests/ -v
```

If you want to only run unit tests, this can be done via

```bash
pytest ./umami/tests/unit/ -v
```

and the integration test similarly via

```bash
pytest ./umami/tests/integration/ -v
```

If you want to see the output of the code which is being tested you can also add the `-s` flag.



## Tools for uniform coding style

In order to run the code style checker `flake8` use the following command:
```bash
flake8 ./umami
```

In order to run the yaml linter `yamllint` use the following command:

```bash
yamllint -d "{extends: relaxed, rules: {line-length: disable}}" .
```

In order to format the code using [`black`](https://github.com/psf/black) use the following command:

```bash
black ./umami
```

Checking doc strings (more infos below)
```bash
darglint * -s numpy -z full  --log-level INFO
```


## Commiting changes and pre-commit hooks

If you want to commit changes it is recommended to install the [pre-commit hooks](https://githooks.com) by doing the following:

```bash
pre-commit install
```

This will run `isort`, `black`, `flake8` and `yamllint` on staged python files when commiting.

## Excluding files from git only locally

If you want to exclude certain files only locally for you and not changing the `.gitignore` file from `umami` you can
find some solution on [stack overflow](https://stackoverflow.com/questions/5724455/can-i-make-a-user-specific-gitignore-file).

You can e.g. add the patterns for the files which should be excluded to the file in your git folder `.git/info/exclude`.





## Global Configuration

There is a [global configuration](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configs/global_config.yaml) file which allows to set different global settings.

| Option | Description |
|--------|-------------|
| `pTvariable`       |    Setting the name of the $p_T$ variable which is used in several places.         |
|  `etavariable`      |      Setting the name of the `absolute eta` variable which is used in several places.        |
|  `DebugLevel`      |      Defines the debug level. Possible values:  `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`      |
|  `TFDebugLevel`      |      Defines the debug level of tensorflow, it takes integer values [0,1,2,3], where 0 prints all messages.      |


## Updating CI files

In certain cases it is necessary to update the CI files located e.g. in `/eos/user/u/umamibot/www/ci/preprocessing`.

To prepare these preprocessing files you can use the provided config in `.gitlab/workflow/ci-preprocessing.yaml` and create the files then via

```bash
preprocessing.py -c .gitlab/workflow/ci-preprocessing.yaml --prepare
```

This will give you 5 different files

- `ci_ttbar_basefile.h5`
- `ci_ttbar_testing.h5`
- `ci_zpext_basefile.h5`
- `ci_zpext_testing.h5`

To copy them to the `eos` area, please ask one of the umami responsibles.


## Generate `requirements.txt` Lock File
The `requirements.txt` file in the repository serves for umami also as a lock file, pinning down the exact packages and their versions installed in the environment in which umami "lives". This file is created using the [`uv`](https://pypi.org/project/uv/) package.

To recreate the `requirements.txt` file (needed for when you changeda version of a package in the `setup.cfg` for example) you need to install `uv` via

```bash
python -m pip install uv
```

If you run umami in a containered way, you might want to set the extra `--prefix python_install` option, where `python_install` is the path where umami is installed, to install uv in the same python version as used in the container. The full command would look like this:

```bash
python -m pip install --prefix python_install uv
```