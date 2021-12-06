import gitlab
import os

# connecting to the CERN gitlab API
gl = gitlab.Gitlab(
    "https://gitlab.cern.ch", private_token=os.environ["API_UMAMIBOT_TOKEN"]
)
# specifying the project, in this case umami
project = gl.projects.get("79534")

mr_id = os.environ["CI_MERGE_REQUEST_IID"]
note = f"The detailed coverage is available under https://umami-ci-coverage.web.cern.ch/{mr_id}/"
mr = project.mergerequests.get(mr_id)
post_coverage = True
for note_i in mr.notes.list():
    if note_i.body == note:
        post_coverage = False
        break
if post_coverage:
    # Post the note to the MR if not already done
    mr.notes.create({"body": note})
