#!/bin/bash

echo '[DvC] execute dvc pipeline'
dvc pull || true
dvc repro
dvc_repro_exit_status=$?
dvc commit
dvc push
git add dvc.lock
git_add_dvc_lock=$?

if [ $git_add_dvc_lock -eq 0 ]; then
    echo "[DvC] dvc.lock changed. Commit Changes."
    if [ $dvc_repro_exit_status -eq 0 ]; then
        echo "[DvC] dvc repro executed successfully. Committing dvc.lock changes."
        git commit -m "[DvC] dvc repro executed successfully. Committing dvc.lock changes."
        exit 0
    else
        echo "[DvC] dvc repro failed with exit status: $dvc_repro_exit_status. Committing dvc.lock changes."
        git commit -m "[DvC] dvc repro failed with exit status: $dvc_repro_exit_status. Committing dvc.lock changes. [skip ci]"
        exit 1
    fi 
    export GIT_SSL_NO_VERIFY=1
    git push https://CICD_TOKEN:"$CICD_TOKEN"@"$CI_SERVER_HOST"/"$CI_PROJECT_PATH".git HEAD:"$CI_COMMIT_REF_NAME"
else
    if [ $dvc_repro_exit_status -eq 0 ]; then
        echo "[DvC] dvc repro executed successfully. Nothing to commit. Proceeding with CML."
        exit 0
    else
        echo "[DvC] dvc repro failed with exit status: $dvc_repro_exit_status. No dvc.lock changes."
        exit 1
    fi
fi
