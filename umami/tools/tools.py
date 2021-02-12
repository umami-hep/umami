import re

import yaml

# adding a custom yaml loader in order to be able to have nubers with
# scientific notation
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


def replaceLineInFile(file, key, newLine):
    filedata = ""
    with open(file, "r") as f:
        for line in f:
            if key in line:
                line = newLine + "\n"
            filedata += line

    with open(file, "w") as f:
        f.write(filedata)
