"""Collection of tools used in different places in the project."""
import re

import yaml

from umami.configuration.Configuration import logger

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


def compare_leading_spaces(ref: str, comp: str):
    """Compares if leading spaces of 2 strings are the same.

    Parameters
    ----------
    ref : str
        reference string
    comp : str
        comparison string

    Returns
    -------
    int
        difference in leading spaces of ref and comp string
    """
    ref_spaces = len(ref) - len(ref.lstrip())
    comp_spaces = len(comp) - len(comp.lstrip())
    logger.debug(f"Leading spaces in {ref}: {ref_spaces}")
    logger.debug(f"Leading spaces in {comp}: {comp_spaces}")
    diff_spaces = ref_spaces - comp_spaces
    if diff_spaces != 0:
        logger.warning(
            f"Your strings `{ref}` and `{comp}` have a different amount of leading "
            f"spaces ({diff_spaces})."
        )

    return diff_spaces


def replaceLineInFile(file, key, new_line, only_first=False):
    """Replace line in file

    Parameters
    ----------
    file : str
        file name
    key : str
        key which triggers the replacement of line
    new_line : str
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

    replaced_line = False
    with open(file, "r") as f:
        for line in f:
            if key in line:
                if (only_first and not replaced_line) or not only_first:
                    compare_leading_spaces(line, new_line)
                    line = new_line + "\n"
                    replaced_line = True
            filedata += line

        if replaced_line is False:
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


def check_main_class_input(main_class) -> set:
    """
    Checks the given main class for type and returns
    a set with the main classes inside.

    Parameters
    ----------
    main_class : str or list
        Main class loaded from the yaml file.

    Returns
    -------
    main_class : set
        The main class(es) as a set.

    Raises
    ------
    TypeError
        If the given main_class is neither a string, list or a set.
    """

    # Check main class if string or list and covert it to a set
    if isinstance(main_class, str):
        main_class = set([main_class])

    elif isinstance(main_class, list):
        main_class = set(main_class)

    elif not isinstance(main_class, set):
        raise TypeError(
            f"Main class must either be a str or a list, not a {type(main_class)}"
        )

    return main_class
