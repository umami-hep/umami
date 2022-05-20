"""Script that converts numpy docstrings to markdown tables.
These .md files can then be used in the documentation."""
import re

import puma
from npdoc_to_md import render_md_from_obj_docstring  # pylint: disable=import-error


def generate_parameters_table(
    obj: object,
    filename: str,
    exclude: list = None,
):
    """Creates a markdown file with a table that lists the parameters of a given object

    Parameters
    ----------
    obj : object
        Object you want to get the docstring of
    filename : str
        Filename where to save the markdown file
    exclude : list, optional
        List of parameters which should be excluded from the resulting table,
        by default None
    """

    # get markdown rendered docstring
    full_docstring_md = render_md_from_obj_docstring(obj=obj, obj_namespace="")
    parameters_md_table = "| Parameter | Type | Description |\n|---|---|---|\n"

    parameters_md = None
    for paragraph in full_docstring_md.split("\n\n"):
        if "Parameters" in paragraph and parameters_md is None:
            parameters_md = paragraph

    # modify the resulting markdown text such that the result is a table
    # the parameters are listed in a markdown list of the form
    # * <parameter_name> : <b><i><type></i></b>  <description>
    for line in parameters_md.split("\n* "):
        # skip the heading line
        if "##" in line:
            continue
        # join the lines (for cases where the description is multi-line)
        line = line.replace("\n", "")
        # extract parameter_name, type_hint and description
        parameter_search = re.search("(.*) : <b><i>", line)
        type_hint_search = re.search("<b><i>(.*)</i></b>", line)
        description_search = re.search("</i></b> (.*)", line)
        if (
            parameter_search is None
            or type_hint_search is None
            or description_search is None
        ):
            print(f"\x1b[1;33;40mSkipping this line: {line}\x1b[0m")
            continue
        parameter = parameter_search.group(1)
        type_hint = type_hint_search.group(1)
        description = description_search.group(1)
        if exclude is None or parameter not in exclude:
            parameters_md_table += (
                f"| `{parameter}` "
                f"| `{type_hint.split(', optional')[0]}`"
                f"{', optional' if 'optional' in type_hint else ''} "
                f"| {description} |\n"
            )
        else:
            print(f"\x1b[1;33;40mExcluding parameter `{parameter}`\x1b[0m")

    print(f"\nOriginal markdown:\n>>>\n{parameters_md}\n<<<")
    print(f"\nMarkdown table:\n{parameters_md_table}")
    print(100 * "-")

    with open(filename, "w") as docstring_md_table:
        docstring_md_table.write(parameters_md_table)


def main():
    """Main function which is called when the script is executed"""
    # define here the objects of which you want the parameters as markdown table
    objects_to_render = {
        "puma.PlotObject": {
            "obj": puma.PlotObject,
            "filename": "docstring_puma_PlotObject.md",
            "exclude": ["logy", "plotting_done"],
            # Excluded because:
            # logy -> has different default (False) in Histogram plot
            # plotting_done -> attribute that should not be modified by the user
        },
        "puma.HistogramPlot": {
            "obj": puma.HistogramPlot.__init__,
            "filename": "docstring_puma_HistogramPlot.md",
            "exclude": ["bins", "bins_range", "**kwargs"],
            # Excluded because:
            # bins -> are handled differently (defined in the input var plot config)
            # bins_range -> same here
            # **kwargs -> we specifically put the **kwargs from PlotObject in the docs
        },
    }
    for name, config in objects_to_render.items():
        print(f"\x1b[1;32;40mProcessing: {name}\x1b[0m")
        generate_parameters_table(**config)


if __name__ == "__main__":
    main()
