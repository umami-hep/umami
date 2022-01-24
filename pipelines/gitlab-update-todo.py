"""Checks repository in master and updates ToDo issue."""
import fnmatch
import os
from sys import stdout

import gitlab
import yaml
from pylint import epylint as lint


def pylint_fixmes():
    """
    Updates issue dedicated to ToDos with all todos found in the code.

    Returns
    -------
    list
        pylint_files with Todos
    list
        pylint_msgs with Todos
    """

    (pylint_stdout, _) = lint.py_run("umami/ --disable=all --enable=fixme ", True)
    pylint_stdout = pylint_stdout.read()
    pylint_files, pylint_msgs = [], []
    for line in pylint_stdout.splitlines():
        if "umami/" not in line:
            continue
        file_name, todo_message = line.split(" warning (W0511, fixme, ) ")
        if "TODO: " in todo_message:
            todo_message = todo_message.replace("TODO: ", "")
        elif "TODO " in todo_message:
            todo_message = todo_message.replace("TODO ", "")
        pylint_files.append(file_name[:-1])
        pylint_msgs.append(todo_message)
    return pylint_files, pylint_msgs


if __name__ == "__main__":
    todo_files, todo_msgs = pylint_fixmes()
    issue_description = (
        "This issue shows the TODOs specified in the code. "
        "It is updated each time the CI in the master branch is running."
        "(*Please do not modify the issue description - it will be overwritten*)\n\n"
    )

    for files, msgs in zip(todo_files, todo_msgs):
        issue_description += f"- [ ] {files} - *{msgs}*\n"
    print(issue_description)
    # connecting to the CERN gitlab API
    gl = gitlab.Gitlab(
        "https://gitlab.cern.ch",
        private_token=os.environ["API_UMAMIBOT_TOKEN"],
    )
    # specifying the project, in this case umami
    project = gl.projects.get("79534")

    # issue used to track changes
    issue = project.issues.get(120)
    # post new issue description
    issue.description = issue_description
    issue.save()
