"""Checks a gitlab MR and add labels to it."""
import os

import gitlab  # pylint: disable=import-error
import yaml


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
    with open(".gitlab/workflow/label_mapping.yaml", "r") as file:
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
    changelog_changed = "changelog.md" in changed_files_mr
    mr_labels, changed_files_in_docs_mr = get_labels(changed_files_mr, mr.labels)

    mr.labels = mr_labels
    mr.save()
    print("Found following labels:", mr_labels)
    # define flag if only documentation is concerned
    only_docs = (changed_files_in_docs_mr + changelog_changed) == len(changed_files_mr)
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
