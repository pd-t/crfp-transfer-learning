#!/bin/bash

git config user.email "cml@gitlab.wogra.com"
git config user.name "Gitlab CML"
echo '[CML] pull dvc data'
dvc pull || true

echo '[CML] dvc repro'
if dvc repro
  then
    echo '[CML] dvc repro success!'
  else
    echo '[CML] dvc repro error!'
    exit 1
fi

dvc commit
dvc push
git add dvc.lock

echo '[CML] git commit'
if git commit -m '[CML] Add new dvc.lock'
  then
    export GIT_SSL_NO_VERIFY=1
    echo '[CML] dvc.lock changed. Pipeline restarts to prevent release. This is not an error!'
    git push https://CML_TOKEN:"$CML_TOKEN"@"$CI_SERVER_HOST"/"$CI_PROJECT_PATH".git HEAD:"$CI_COMMIT_REF_NAME"
    exit 1
  else
    echo '[CML] dvc is up-to-date, ready for release!'
fi


