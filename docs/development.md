## Development

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


In order to run the code style checker flake8 use the following command:
```bash
flake8 ./umami
```

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
