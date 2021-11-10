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

## General good coding practices

In the following we are listing some good code practices, we are asking to follow when making merge requests to the repository.

## Commenting your Code
If you write new code for Umami, please keep in mind to comment your code properly. It is very hard to understand what you are doing something and why you are doing it. Please keep this in mind! This will make it easier to revise your code.

To make the framework modular, new code that is repeated should be written in a function. When defining a function, please provide a proper doc string and type of the input variables. For example:

```python
def LoadJetsFromFile(
    filepath: str,
    class_labels: list,
    nJets: int,
    variables: list = None,
    cut_vars_dict: dict = None,
    print_logger: bool = True,
):
    """
    Load jets from file. Only jets from classes in class_labels are returned.

    Input:
    - filepath: Path to the .h5 file with the jets.
    - class_labels: List of class labels which are used.
    - nJets: Number of jets to load.
    - variables: Variables which are loaded.
    - cut_vars_dict: Variable cuts that are applied when loading the jets.
    - print_logger: Decide if the number of jets loaded from the file is printed.

    Output:
    - Jets: The jets as numpy ndarray
    - Umami_labels: The internal class label for each jet. Corresponds with the
                    index of the class label in class_labels.
    """

```

### Unit/Integration Tests
If you contribute to Umami, please keep in mind that all code should be tested by unit- and integration tests. Normally, the integration test will cover small changes in the pipeline directly, but unit test should be added for all new functions added! Please make sure that all cases of the new functions are tested!

### Readability of numbers
To make large number better readable, please use a `_` to separate them (typically the thousand separator) which was introduced in python 3.6 [PEP515](https://www.python.org/dev/peps/pep-0515/#literal-grammar). 
For examle instead of `6728339` please use `6_728_339`.

### Usage of Generators
Often it is more useful to use a generator in the code than returning the values in the loop. You can find examples [here](https://wiki.python.org/moin/Generators) stating *The performance improvement from the use of generators is the result of the lazy (on demand) generation of values, which translates to lower memory usage* and a selection is given below.

```python
def first_n(n):
    '''Build and return a list'''
    num, nums = 0, []
    while num < n:
        nums.append(num)
        num += 1
    return nums


sum_of_first_n = sum(first_n(1_000_000))
```

```python
# a generator that yields items instead of returning a list
def firstn(n):
    num = 0
    while num < n:
        yield num
        num += 1

sum_of_first_n = sum(firstn(1_000_000))
```


In the same philosophy there is also list and dict comprehension, here such an example
```python
# list comprehension
doubles = [2 * n for n in range(50)]

# same as a generator
doubles = (2 * n for n in range(50))

# dictionary comprehension
dict_variable = {key:value for (key,value) in dictonary.items()}
```

### f-Strings
Since Python 3.6 the so-called f-strings were introduced providing a powerful syntax for string manipulation. Nice examples and explnations can be found [here](https://realpython.com/python-f-strings/). Try to avoid `str.format()` and `%-formatting` whenever possible, especially for a better readability of the code.

A simple example
```python
nJets = 2_300
jet_collection = "EMPFlow"
info_text = f"We are using the {jet_collection} jet collection and have {nJets} available."

## arbitrary operations
info_text_event = f"We are using the {jet_collection} jet collection and have {nJets * 4} available."
```

### Integer division

In Python 3 a dedicated integer division was introduced. 

```python
# standard division -> returns by default a flaot
nEvents = nJets / 4

# integer division -> returns an integer
nEvents = nJets // 4
```

### Type declaration in functions
For a better readablility it is often useful to declare the object type in a function as well as the return type of a function.

Instead of this function
```python
def GetNumberOfEvents(nJets, avg_nJets_per_event=4.3):
    return nJets * avg_nJets_per_event
```

it would look like this
```python
def GetNumberOfEvents(nJets: int, avg_nJets_per_event: float=4.3) -> float:
    return nJets * avg_nJets_per_event
```


### Logging
The umami framework has a custom logging module defined in [umami/configuration/Configuration.py](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configuration/Configuration.py). Do not use the `print()` function but rather the logging. To make use of the module you need to import it via 
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


## Global Configuration

There is a [global configuration](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/umami/configs/global_config.yaml) file which allows to set different global settings.

| Option | Description |
|--------|-------------|
| `pTvariable`       |    Setting the name of the `pT` variable which is used in several places.         |
|  `etavariable`      |      Setting the name of the `absolute eta` variable which is used in several places.        |
|  `DebugLevel`      |      Defines the debug level. Possible values:  `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`      |
|  `TFDebugLevel`      |      Defines the debug level of tensorflow, it takes integer values [0,1,2,3], where 0 prints all messages.      |


