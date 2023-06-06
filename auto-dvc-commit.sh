#!/bin/bash

echo '[DvC] execute dvc pipeline'
dvc pull || true
dvc repro
dvc_repro_exit_status=$?
if [ $dvc_repro_exit_status -eq 0 ]; then
    echo "[DvC] dvc repro executed successfully."
else
    echo "[DvC] dvc repro failed with exit status: $dvc_repro_exit_status."
fi

dvc commit
dvc push
git add dvc.lock
git_add_dvc_lock=$?

if [ $git_add_dvc_lock -eq 0 ]; then
    echo "[DvC] dvc.lock changed. Commit Changes."
    if [ $dvc_repro_exit_status -eq 0 ]; then
        git commit -m "[DvC] dvc.lock changes."
        exit 0
    else
        git commit -m "[DvC] dvc.lock changes. [skip ci]"
        exit 1
    fi 
    export GIT_SSL_NO_VERIFY=1
    git push https://CICD_TOKEN:"$CICD_TOKEN"@"$CI_SERVER_HOST"/"$CI_PROJECT_PATH".git HEAD:"$CI_COMMIT_REF_NAME"
else
    echo "[DvC] dvc.lock not changed. Nothing to commit."
    if [ $dvc_repro_exit_status -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
fi
