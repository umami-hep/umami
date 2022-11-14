"""Checks a gitlab MR and add labels to it."""
import os
from pathlib import Path

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


def get_script_doc_code_replacement(file: str):
    """Get the name of script which is being replaced in docs.

    Parameters
    ----------
    file : str
        File name of the file that is searched for the
        placeholders

    Returns
    -------
    list
        Script names which are used for code replacement in file
    """
    script_names = []
    with open(file, "r") as md_file:
        for line in md_file:
            if "§§§" in line:
                # select string which is enclosed in §§§
                selected_line = line.split("§§§")[1]
                if ":" in selected_line:
                    selected_line = selected_line.split(":")[0]
                script_names.append(selected_line)
    return script_names


def get_file_pattern_directory(directory: str = "docs/", file_extension: list = "*.md"):
    """Searches in directory all files for specific string patterns.

    Parameters
    ----------
    directory : str
        Directory in which to search files.
    file_extension: str
        File extension to be used, by default `*.md`

    Returns
    -------
    set
        Set of script names which are used for code replacement in docs
    """
    filenames = [str(path) for path in Path(directory).rglob(file_extension)]
    if "docs/setup/development/good_practices_docs.md" in filenames:
        filenames.remove("docs/setup/development/good_practices_docs.md")

    script_names = []
    for filename in filenames:
        script_names += get_script_doc_code_replacement(filename)
    return set(script_names)


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

    # Check if files were changed which are used as placeholders in docs, if yes,
    # comment on MR to check that nothing is messed up
    docs_with_placeholders = get_file_pattern_directory()
    placeholder_files_changed = docs_with_placeholders.intersection(
        set(changed_files_mr)
    )
    if len(placeholder_files_changed) > 0:
        print(
            "Following files are changed which are used as placeholders",
            placeholder_files_changed,
        )
        NOTE = (
            "The following files are changed and used as placeholders in the docs:\n"
            + "\n".join(f"* `{elem}`" for elem in placeholder_files_changed)
            + "\n\n**Please check that they are still correct!**"
        )
        POST_NOTE = True
        # go through all discussions in the MR
        for discussion in mr.discussions.list():
            # go through all notes of this discussion
            for note_i in discussion.attributes["notes"]:
                if note_i["body"] == NOTE:
                    POST_NOTE = False
                    print(
                        "Comment about changed placeholders already exists "
                        "--> not posting."
                    )
                    break
        if POST_NOTE:
            # Post the note to the MR if not already done
            mr.discussions.create({"body": NOTE})
            # Get the id of the discussion that was just created
            discussion_id = mr.discussions.list()[-1].attributes["id"]
            mr_d = mr.discussions.get(discussion_id)
            # Set the discussion to unresolved, such that it has to be addressed
            mr_d.resolved = False
            print("Posting comment that placeholders should be checked.")
        mr.save()

    # define flag if only documentation is concerned
    ONLY_DOCS = (changed_files_in_docs_mr + changelog_changed) == len(changed_files_mr)
    if len(changed_files_mr) == 0:
        ONLY_DOCS = False

    # approve MR if only documentation is concerned
    if ONLY_DOCS:
        print("MR is being approved - only documentaion is concerned.")
        mr.notes.create({"body": "Only documentation is concerened - approving."})
        try:
            mr.approve()
        except gitlab.exceptions.GitlabAuthenticationError:
            print("Approving not permitted.")
        mr.save()
