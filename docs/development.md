## Development

### Test suite

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

### Tools for uniform coding style

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


### Commiting changes and pre-commit hooks

If you want to commit changes it is recommended to install the [pre-commit hooks](https://githooks.com) by doing the following:

```bash
pre-commit install
```

This will run `isort`, `black` and `flake8` on staged python files when commiting.

## Global Configuration

There is a [global configuration](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configs/global_config.yaml) file which allows to set different global settings.

| Option | Description |
|--------|-------------|
| `pTvariable`       |    Setting the name of the `pT` variable which is used in several places.         |
|  `etavariable`      |      Setting the name of the `absolute eta` variable which is used in several places.        |
|  `DebugLevel`      |      Defines the debug level. Possible values:  `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`      |
|  `TFDebugLevel`      |      Defines the debug level of tensorflow, it takes integer values [0,1,2,3], where 0 prints all messages.      |


## Logging
The umami framework has a custom logging module defined in [umami/configuration/Configuration.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configuration/Configuration.py). To make use of the module you need to import it via 
```python
from umami.configuration import logger
```
and then it can be used e.g. via
```python
logger.info(f"Loading config file {config_file}.")
logger.debug(f"Using variable {variable} in training.")
logger.warning(f"Not enough jets available in sample, using only {njets}")
```
All logging levels are defined in the following table

| Level    | Numeric value |
|----------|---------------|
| CRITICAL | 50            |
| ERROR    | 40            |
| WARNING  | 30            |
| INFO     | 20            |
| DEBUG    | 10            |
| NOTSET   | 0             |
