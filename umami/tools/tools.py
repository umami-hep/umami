"""Collection of tools used in different places in the project."""
import re

import yaml

# adding a custom yaml loader in order to be able to have nubers with
# scientific notation
# TODO: This should be replaced everywhere with the new YAML loader which
# also allows !include
yaml_loader = yaml.SafeLoader
yaml_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def replaceLineInFile(file, key, newLine, only_first=False):
    """Replace line in file

    Parameters
    ----------
    file : str
        file name
    key : str
        key which triggers the replacement of line
    newLine : str
        content of line replacement
    only_first : bool, optional
        if True only first line in which key found is replaced, by default False

    Raises
    ------
    AttributeError
        If no matching line could be found.
    AttributeError
        If no matching line could be found.
    """
    filedata = ""

    if only_first:
        replacedLine = False
        with open(file, "r") as f:
            for line in f:
                if key in line and not replacedLine:
                    line = newLine + "\n"
                    replacedLine = True
                filedata += line

            if replacedLine is False:
                raise AttributeError(f'No line could be found matching "{key}"')

        with open(file, "w") as f:
            f.write(filedata)

    else:
        replacedLine = False
        with open(file, "r") as f:
            for line in f:
                if key in line:
                    line = newLine + "\n"
                    replacedLine = True
                filedata += line

            if replacedLine is False:
                raise AttributeError(f'No line could be found matching "{key}"')

        with open(file, "w") as f:
            f.write(filedata)


def atoi(text):
    """
    Return string as int, if the given string is a int.

    Parameters
    ----------
    text : str
        String with int inside.

    Returns
    -------
    Int_string : int/str
        Returning the string if it is not a digit, otherwise
        return string as int.
    """

    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    Sorting strings by natural keys.

    Parameters
    ----------
    text : str
        String with int inside.

    Returns
    -------
    sorted_list : list
        List with the sorted strings inside.
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]
