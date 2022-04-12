# General good coding practices

In the following we are listing some good code practices, we are asking to follow when making merge requests to the repository. We would like to ask you to use British English for your contributions.

### Commenting your Code
If you write new code for Umami, please keep in mind to comment your code properly. It is very hard to understand what you are doing something and why you are doing it. Please keep this in mind! This will make it easier to revise your code.

To make the framework modular, new code that is repeated should be written in a function. When defining a function, please provide a proper doc string and type of the input variables. The type of the input variables of packages (like `numpy`) can also be set when the package is imported (`np.ndarray` for example). An example of this can be seen here:

```python
def example_function(
    y_pred: np.ndarray,
    class_labels: list,
    unique_identifier: str = None,
) -> dict:  # You can say what the function returns (this only works for one return value)
    """
    Add here a description of what the function does.

    Parameters
    ----------
    y_pred : numpy.ndarray
        Add here a description of the argument.
    class_labels : list
        Add here a description of the argument.
    unique_identifier: str
        Add here a description of the argument and also a
        "as default None" if a default value is given

    Returns
    -------
    Rejection_Dict : dict
        Add here a description of the returned element.

    Raises
    ------
    ValueError
        If you have raise statements in the function, list them here
        and add here (where this text stands) a description in which
        cases this error is called.
    """
```

### Doc strings
Each function and class should have a doc string describing its functionality.
The numpy style for doc strings is being used which is documented [here](https://numpydoc.readthedocs.io/en/latest/format.html)

In the section above is an example for a docstring given. In `Parameters`, all arguments of the function are listed. First is the name of the argument followed by whitespace, double point and again whitespace and then the argument type. The line below needs a indentation to signal that this is the explanation for this argument. Multiple lines can be written like that. `self` for class functions doesn't need to be added here.
The same rules are for the `Returns` part. If nothing is returned, add a `-> None` in the function definition (in the example there is `-> dict` currently).
If your function has a `raise` statement, you also need to add a section called `Raises`, where the Error is added and a line below, with indentation, a description why the error was raised. This needs to be done for all raise statements. So multiple `ValueError` can be in this section.

To check if your doc string is compatible with the recommended style you can use

```bash
darglint <path/to/your/file> -s numpy -z full --log-level INFO
```

You can choose for yourself whether it is necessary to also document the keys for dictionaries. There is no official recommendation in the doc strings docs or from the [community](https://stackoverflow.com/questions/62511086/how-to-document-kwargs-according-to-numpy-style-docstring). If you prefer to document also `dict` keys, here is an [example](https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html) from numpy.

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
# standard division -> returns by default a flaot (no rounding)
nEvents = nJets / 4

# integer division -> rounds to integer precision
nEvents = nJets // 4
```

### Type declaration in functions
For a better readablility it is often useful to declare the object type in a function as well as the return type of a function.

Instead of this function
```python
def get_number_of_events(n_jets, avg_n_jets_per_event=4.3):
    return n_jets * avg_n_jets_per_event
```

it would look like this
```python
def get_number_of_events(n_jets: int, avg_n_jets_per_event: float=4.3) -> float:
    return n_jets * avg_n_jets_per_event
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

