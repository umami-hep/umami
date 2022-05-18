"""Posts the urls in MRs for temporary docs and coverage reports."""
import os

import gitlab  # pylint: disable=import-error

# connecting to the CERN gitlab API
gl = gitlab.Gitlab(
    "https://gitlab.cern.ch", private_token=os.environ["API_UMAMIBOT_TOKEN"]
)
# specifying the project, in this case umami
project = gl.projects.get("79534")

mr_id = os.environ["CI_MERGE_REQUEST_IID"]
mr_docs = f"https://umami-ci-provider.web.cern.ch/mr-docs/{mr_id}"
mr_docs_sphinx = f"https://umami-ci-provider.web.cern.ch/mr-docs/{mr_id}/sphinx-docs"

note = (
    "The detailed coverage is available under"
    f" https://umami-ci-coverage.web.cern.ch/{mr_id}/"
    f" \n\n The docs built for the MR are available under {mr_docs}"
    f" and the API reference is available via {mr_docs_sphinx}"
)
mr = project.mergerequests.get(mr_id)

post_coverage = True
for note_i in mr.notes.list():
    if note_i.body == note:
        post_coverage = False
        break
if post_coverage:
    # Post the note to the MR if not already done
    mr.notes.create({"body": note})
