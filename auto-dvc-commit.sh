#!/bin/bash

echo '[DvC] dvc pull&repro&commit&push'
dvc pull || true
dvc_repo = $(dvc repro) || true
dvc commit
dvc push

echo '[DvC] git commit dvc.lock'
git add dvc.lock
if git commit -m '[DvC] Add new dvc.lock'
  then
    export GIT_SSL_NO_VERIFY=1
    echo '[DvC] dvc.lock changed. Pipeline restarts for CML. This is not an error!'
    git push https://CICD_TOKEN:"$CICD_TOKEN"@"$CI_SERVER_HOST"/"$CI_PROJECT_PATH".git HEAD:"$CI_COMMIT_REF_NAME"
fi

exit dvc_repo