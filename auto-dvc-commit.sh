#!/bin/bash

echo '[DVC] pull dvc data'
dvc pull || true

echo '[DVC] dvc repro'
if dvc repro
  then
    echo '[DVC] dvc repro success!'
  else
    echo '[DVC] dvc repro error!'
    exit 1
fi

dvc commit -f
dvc push
git add dvc.lock

echo '[DVC] git commit'
if git commit -m '[DVC] Add new dvc.lock'
  then
    echo '[DVC] dvc.lock changed. Pipeline restarts to prevent release. This is not an error!'
    export GIT_SSL_NO_VERIFY=1  
    git push https://CICD_TOKEN:"$CICD_TOKEN"@"$CI_SERVER_HOST"/"$CI_PROJECT_PATH".git HEAD:"$CI_COMMIT_REF_NAME"
    exit 1
  else
    echo '[DVC] dvc is up-to-date, ready for release!'
fi
