import os

import gitlab

labels = {
    "Training": ["umami/train_", "umami/tf_tools/", "umami/train_tools/"],
    "DIPS": ["umami/train_Dips.py"],
    "DL1": ["umami/train_DL1.py"],
    "Umami": ["umami/train_umami.py"],
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


# connecting to the CERN gitlab API
gl = gitlab.Gitlab(
    "https://gitlab.cern.ch", private_token=os.environ["API_UMAMIBOT_TOKEN"]
)
# specifying the project, in this case umami
project = gl.projects.get("79534")

mr_id = os.environ["CI_MERGE_REQUEST_IID"]
mr = project.mergerequests.get(mr_id)

changed_files = [elem["new_path"] for elem in mr.changes()["changes"]]

mr_labels, changed_files_in_docs = get_labels(changed_files, mr.labels)
# define flag if only documentation is concerned
only_docs = changed_files_in_docs == len(changed_files)
# if only documentation is concerned adding a Skip-CI label
if only_docs:
    mr_labels.append("Skip-CI")
# in case other files are changed in a later commit, removing the skip-ci flag
if not only_docs:
    if "Skip-CI" in mr_labels:
        mr_labels.remove("Skip-CI")
mr.labels = mr_labels
mr.save()

# approve MR if only documentation is concerned
if only_docs:
    mr.notes.create({"body": "Only documentation is concerened - approving."})
    mr.approve()
    mr.save()
