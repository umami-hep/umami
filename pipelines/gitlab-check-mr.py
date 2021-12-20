import fnmatch
import os

import gitlab

labels = {
    "Training": ["umami/models/", "umami/tf_tools/", "umami/train_tools/"],
    "DIPS": [
        "umami/models/Model_Dips.py",
        "umami/models/Model_Dips_cond_att.py",
    ],
    "DL1": ["umami/models/Model_DL1.py"],
    "Umami": ["umami/models/Model_Umami.py"],
    "Documentation": ["docs/", "README.md", "mkdocs.yml"],
    "Evaluation": ["umami/evaluation_tools/", "umami/evaluate_model.py"],
    "Plotting": [
        "umami/plotting_umami.py",
        "umami/plot_input_variables.py",
        "umami/plotting_epoch_performance.py",
        "umami/input_vars_tools/",
    ],
    "Preprocessing": ["umami/preprocessing.py", "umami/preprocessing_tools/"],
    "Unit/Integration Test": ["umami/tests/"],
    "CI/CD": [".gitlab-ci.yml", "pipelines/"],
    "Setup/Dependencies": [
        "requirements.txt",
        "setup.cfg",
        "docker/",
        "MANIFEST.in",
        "run_setup.sh",
        "setup.py",
        "pyproject.toml",
        ".pre-commit-config.yaml",
    ],
}


def get_labels(changed_files: list, mr_labels: list):
    """
    Depending on the changed files in a MR, different labels are associated to
    the MR.

    Parameters
    ----------
    changed_files : list
        Files which were changed in the merge request.
    mr_labels : list
        Already associated merge request labels.

    Returns
    -------
    mr_labels: list
    changed_files_in_docs : list
    """
    changed_files_in_docs = 0
    for elem in changed_files:
        for label in labels:
            for entry in labels[label]:
                if entry in elem:
                    mr_labels.append(label)
                    if label == "Documentation":
                        changed_files_in_docs += 1

    return list(set(mr_labels)), changed_files_in_docs


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

    changed_files = [elem["new_path"] for elem in mr.changes()["changes"]]
    changed_python_files = fnmatch.filter(changed_files, "*.py")
    print(changed_python_files)
    mr_labels, changed_files_in_docs = get_labels(changed_files, mr.labels)
    # define flag if only documentation is concerned
    only_docs = changed_files_in_docs == len(changed_files)
    if len(changed_files) == 0:
        only_docs = False

    mr.labels = mr_labels
    mr.save()

    # approve MR if only documentation is concerned
    if only_docs:
        mr.notes.create({"body": "Only documentation is concerened - approving."})
        try:
            mr.approve()
        except gitlab.exceptions.GitlabAuthenticationError:
            print("Approving not permitted.")
        mr.save()
