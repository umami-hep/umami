### CERN Gitlab / GitHub instances

There are two instances of this project:

1. [https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami)
2. [https://github.com/umami-hep/umami](https://github.com/umami-hep/umami)

The primary instance is hosted by CERN at the GitLab instance. As only CERN computing accounts can have full access and make contributions, there is a second instance hosted by GitHub. The primary instance is regularly mirrored to the GitHub instance. The purpose of the GitHub instance is to allow everybody to report issues and open merge/pull requests which are then ported to the GitLab instance after they have been merged on GitHub.


### Branch Workflow

If you want to contribute to the development of Umami, you can clone the project and then should make changes inside a [feature branch](https://docs.gitlab.com/ee/gitlab-basics/feature_branch_workflow.html). When your feature is ready, open a [merge request](https://docs.gitlab.com/ee/user/project/merge_requests/) (GitLab) / [pull request](https://docs.github.com/de/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) (GitHub) to the `master` branch. 


```bash
# option 1: clone from gitlab (CERN)
git clone ssh://git@gitlab.cern.ch:7999/atlas-flavor-tagging-tools/algorithms/umami.git
# option 2: clone from github (everyone)
git clone git@github.com:umami-hep/umami.git

# open new branch
git checkout -b dev_<feature_name>
```


### Development guidelines

Good coding standards are highly encouraged. You can find some more information in the [development guide for Umami](https://umami-docs.web.cern.ch/setup/development/) or in the [coding style suggestions](https://umami-docs.web.cern.ch/setup/development/good_practices_code/).

CERN users also have access to a [coding tutorial](https://ftag.docs.cern.ch/software/tutorials/tutorial-coding/) (only for CERN users).
In short, aim to write clean, readible and type-hinted code with module and function docstrings, and plenty of inline comments.

VS Code is the recommended editor when developing for Umami. See also the [umami guide](https://umami-docs.web.cern.ch/setup/development/VS_code/) for development with VS Code.
