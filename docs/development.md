## Development

If you wan to get started with umami you can pick an issue and work on it, well suited
are the issues labeled with `good-first-issue` which can be found [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/issues?scope=all&state=opened&label_name[]=good-first-issue). Please tell us if you are working on an issue and create already in the beginning a merge request marked as `Draft`. This helps us to see who is working at what.


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

Checking doc strings (more infos below)
```bash
darglint * -s numpy -z full  --log-level INFO
```


### Commiting changes and pre-commit hooks

If you want to commit changes it is recommended to install the [pre-commit hooks](https://githooks.com) by doing the following:

```bash
pre-commit install
```

This will run `isort`, `black`, `flake8` and `yamllint` on staged python files when commiting.

### Excluding files from git only locally

If you want to exclude certain files only locally for you and not changing the `.gitignore` file from `umami` you can
find some solution on [stack overflow](https://stackoverflow.com/questions/5724455/can-i-make-a-user-specific-gitignore-file).

You can e.g. add the patterns for the files which should be excluded to the file in your git folder `.git/info/exclude`.


## General good coding practices

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



## Using Visual Studio Code
The editor Visual Studio Code (VSCode) provides very nice and helpful options for developing Umami. VSCode is also able to run a singularity image
with Umami and therefore has all the needed dependencies (Python interpreter, packages, etc.) at hand. A short explanation how to set this up
will be given here.

### Using a Singularity Image on a Remote Machine with Visual Studio Code
To use a Singularity image on a remote machine in VSCode, to use the Python interpreter etc., we need to set up some configs and get some
VSCode extensions. The extensions needed are:

| Extension | Mandatory | Explanation |
|-----------|-----------|-------------|
| [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) | Yes | The Remote - SSH extension lets you use any remote machine with a SSH server as your development environment. |
| [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) | Yes | The Remote - Containers extension lets you use a singularity container as a full-featured development environment. |
| [Remote - WSL](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl) | Yes (On Windows) | The Remote - WSL extension lets you use VS Code on Windows to build Linux applications that run on the Windows Subsystem for Linux (WSL). |
| [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) | Yes | The Remote Development extension pack allows you to open any folder in a container, on a remote machine, or in the Windows Subsystem for Linux (WSL) and take advantage of VS Code's full feature set. |

Now, to make everything working, you need to prepare two files. First is your ssh config (can be found in ~/.ssh/config). This file
needs to have the permission of only you are able to write/read it (`chmod 600`). In there, it can look like this for example:

```bash
Host login_node
    HostName <Login_Hostname>
    User <Login_Username>
    IdentityFile <path>/<to>/<private>/<key>

Host working_node tf2~working_node
    HostName <working_node_hostname>
    User <Username>
    ProxyJump login_node
```

The first entry is, for example, the login node of your cluster. The second is the working node. The login node is jumped (used as a bridge). The
second entry also has two names for the entry, one has a `tf2~` in front. This is *important* for the following part, so please add this here.
After adapting the config file, you need to tell VSCode where to find it. This can be set in the `settings.json` of VSCode. You can find/open it in
VSCode when pressing `Ctrl + Shift + P` and start typing `settings`. You will find the option `Preferences: Open Settings (JSON)`. When selecting this,
the config json file of VSCode is opened. There you need to add the following line with the path of your ssh config file added (if the config is in the default path `~/.ssh/config`, you don't need to add this).

```json
"remote.SSH.configFile": "<path>/<to>/<ssh_config>",
"remote.SSH.remoteServerListenOnSocket": false,
"remote.SSH.enableRemoteCommand": true,
```

The second option added here disables the `ListenOnSocket` function which blocks the running of the singularity images in some cases. The third option
will enable the remote command needed for singularity which is blocked when `ListenOnSocket` is `True`. Node: If this gives you errors, you need to switch
to the pre-release version of [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh). Just click on the extension in the extension tab and click `Switch to Pre-Release` at the top.

Next, you need to create a executable script, lets call it `singularity-ssh` here, which tells VSCode what to do when connecting. This file is the same
for Linux/Mac but looks a bit different for Windows. After creating this files, you need to make them executable (`chmod +x <file>`) and also add them
in the VSCode settings with:

```json
"remote.SSH.path": "<path>/<to>/<executable>",
```

Now restart VSCode and open the Remote Explorer tab. At the top switch to `SSH Targets` and right-click on the `tf2~` connection and click on
`Connect to Host in Current Window`. VSCode will now install a VSCode server on your ssh target to run on and will ask you to install your
extensions on the ssh target. This will improve the performance of VSCode. It will also ask you which path to open. After that, you can open
a python file and the Python extension will start and should show you at the bottom of VSCode the current Python Interpreter which is used.
If you now click on the errors and warnings right to it, the console will open where you can switch between Problems, Output, Debug Console, Terminal
and Ports. In terminal should be a fresh terminal with the singularity image running. If not, check out output and switch on the right from Tasks to
Remote - SSH to see the output of the ssh connection.

#### Singularity-SSH Linux/Mac
```bash
#!/bin/sh

# Get last command line argument, should be hostname/alias
for trghost; do true; done

# Parse host-aliases of form "venvname~hostname"
imagename=`echo "${trghost}" | sed 's/^\(\(.*\)~\)\?.*$/\2/'`

# Note: VS Code will override "-t" option with "-T".

if [[ "${imagename}" =~ tf2 ]]; then
    exec ssh -t "$@" "source /etc/profile && module load tools/singularity/3.8 && singularity shell --nv --contain -B /work -B /home -B /tmp docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase-plus:latest-gpu"
else
    exec ssh "$@"
fi
```

If somehow this is not working, you can try to extract the hostname directly with this:

```bash
#!/bin/sh

# Get last command line argument, should be hostname/alias
for trghost
do
    if [ "${trghost}" = "tf2~working_node" ]; then
        image="${trghost}"
    fi
done

# Parse host-aliases of form "venvname~hostname"
imagename=`echo "${image}" | sed 's/^\(\(.*\)~\)\?.*$/\2/'`

# Note: VS Code will override "-t" option with "-T".

if [ "${imagename}" = "tf2" ]; then
    exec ssh -t "$@" "source /etc/profile && module load tools/singularity/3.8 && singularity shell --nv --contain -B /work -B /home -B /tmp docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase-plus:latest-gpu"
else
    exec ssh "$@"
fi
```

#### Singularity-SSH Windows
This file needs to have the file ending `.cmd`!

```bat
@echo off

if NOT %1==-V (
    for /F "tokens=1,3 delims=~" %%a in ("%~4") do (
        if %%a==tf2 (
            ssh.exe -t %2 %3 %4 "source /etc/profile && module load tools/singularity/3.8 && singularity shell --nv --contain -B /work -B /home -B /tmp docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/umamibase-plus:latest-gpu"
        ) else if %%a==tf1 (
            echo "connect with another image"
        ) else (
            ssh.exe %*
        )
    )
) else (
    ssh.exe -V
)
```

### Useful Extensions
| Extension | Mandatory | Explanation |
|-----------|-----------|-------------|
| [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) | Yes | A Visual Studio Code extension with rich support for the Python language (for all actively supported versions of the language: >=3.6), including features such as IntelliSense (Pylance), linting, debugging, code navigation, code formatting, refactoring, variable explorer, test explorer, and more! |
| [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) | Yes (Will be installed with Python extension) | Pylance is an extension that works alongside Python in Visual Studio Code to provide performant language support. Under the hood, Pylance is powered by Pyright, Microsoft's static type checking tool. Using Pyright, Pylance has the ability to supercharge your Python IntelliSense experience with rich type information, helping you write better code faster. |
| [Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) | No | Automatically creates a new docstring with all arguments, their types and their default values (if defined in the function head). You just need to fill the descriptions. |

To make full use of VSCode, you can add the following lines to your `settings.json` of VSCode:

```json
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "autoDocstring.docstringFormat": "numpy",
```

The first entry here sets the automated python formatter to use. Like in Umami, you can set this to `black` to have your code auto-formatted. The second
entry enables auto-format on save. So everytime you save, `black` will format your code (style-wise). The third entry set the docstring style used in the
[Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring). Just press `Ctrl + Shift + 2` (in Linux) below
a function header and the generator will generate a fresh docstring with all arguments, their types and their default values (if defined in the function head) in the `numpy` docstring style (which is used in Umami).

### VSCode Debugger
There are plenty of tutorials and instructions for the VSCode debugger.
However, you might run into trouble when trying to debug a script which is using Umami, with the debugger telling you it can not locate the `umami` package.
In this case, try adding the directory where umami is located to the environment variables that are loaded with a new debugger terminal.

Click on `create a launch.json` as explained [here](https://code.visualstudio.com/docs/python/debugging), select the directory where you want to store it (in case you have multiple folders open) and select "Python File".
VSCode will create the default configuration file for you (located in `.vscode`). All you have to do is adding the following to the `configurations` section:
```json
            "env": {
                "PYTHONPATH": "<your_umami_dir>"
            }
```
Afterwards, the debugger should find the umami package.
