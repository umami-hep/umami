"""Checks a gitlab MR and posts pylint report and labels to it."""
import fnmatch
import os

import gitlab
import yaml
from pylint.lint import Run


def get_labels(changed_files: list, labels_mr: list):
    """
    Depending on the changed files in a MR, different labels are associated to
    the MR.

    Parameters
    ----------
    changed_files : list
        Files which were changed in the merge request.
    labels_mr : list
        Already associated merge request labels.

    Returns
    -------
    labels_mr: list
    changed_files_in_docs : list
    """
    with open("pipelines/label_mapping.yaml", "r") as file:
        labels = yaml.load(file, yaml.FullLoader)

    changed_files_in_docs = 0
    for elem in changed_files:
        for label, files in labels.items():
            for entry in files:
                if entry in elem:
                    labels_mr.append(label)
                    if label == "Documentation":
                        changed_files_in_docs += 1

    return list(set(labels_mr)), changed_files_in_docs


def pylint_mr_message(changed_python_files: list):
    """
    Posts a message in the merge request for changed python files quoting the pylint
    score.

    Parameters
    ----------
    changed_python_files : list
        List containing all python files which were changed in the MR.

    Returns
    -------
    None
        if no python files were changed
    str
        otherwise returning the text to be posted in the MR.
    """
    if len(changed_python_files) == 0:
        return None

    pylint_text = (
        "The umami framework is currently changing to pylint (see #105).\n"
        "It is not yet enforced in the CI, but we recommend to follow the instructions "
        "from pylint. You can see if the files you edited are conform with pylint from "
        "the score behind each file (10 indicates that no issues were found, "
        "everything below needs adaptions). A detailed report can be found in the "
        "`pylint` MR CI job. The output of pylint only for the files you changed "
        "is available in the `checking_mr` CI job.\n\n"
    )
    for changed_file in changed_python_files:
        # avoiding here import errors since we don't use umami base image
        # in the CI
        if "__init__.py" in changed_file:
            continue
        results = Run(["--disable=import-error", changed_file], do_exit=False)
        pylint_score = results.linter.stats.global_note
        pylint_text += f"- `{changed_file}` ({pylint_score:.2f}/10)\n"
    return pylint_text


if __name__ == "__main__":
    # connecting to the CERN gitlab API
    gl = gitlab.Gitlab(
        "https://gitlab.cern.ch",
        private_token=os.environ["API_UMAMIBOT_TOKEN"],
    )
    # specifying the project, in this case umami
    project = gl.projects.get("79534")

    mr_id = os.environ["CI_MERGE_REQUEST_IID"]
    mr = project.mergerequests.get(mr_id)

    changed_files_mr = [elem["new_path"] for elem in mr.changes()["changes"]]
    mr_labels, changed_files_in_docs_mr = get_labels(changed_files_mr, mr.labels)

    mr.labels = mr_labels
    mr.save()
    print("Found following labels:", mr_labels)
    # define flag if only documentation is concerned
    only_docs = changed_files_in_docs_mr == len(changed_files_mr)
    if len(changed_files_mr) == 0:
        only_docs = False

    # approve MR if only documentation is concerned
    if only_docs:
        print("MR is being approved - only documentaion is concerned.")
        mr.notes.create({"body": "Only documentation is concerened - approving."})
        try:
            mr.approve()
        except gitlab.exceptions.GitlabAuthenticationError:
            print("Approving not permitted.")
        mr.save()
    else:
        print("Checking python files with pylint.")
        # Commenting on mr about new pylint status of changed python files
        changed_files_python = fnmatch.filter(changed_files_mr, "*.py")
        if len(changed_files_python) > 0:
            pylint_msg = pylint_mr_message(changed_files_python)
            post_new_pylint_mr = True
            # check if the bot already commented about it, if yes, the comment is
            # being updated, otherwise it will be posted
            for note_i in mr.notes.list():
                if "The umami framework is currently changing to pylint" in note_i.body:
                    note_i.body = pylint_msg
                    note_i.save()
                    post_new_pylint_mr = False
                    break
            if post_new_pylint_mr:
                mr.notes.create({"body": pylint_msg})
