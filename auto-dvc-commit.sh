#!/bin/bash

echo '[DvC] dvc reproduce'
dvc pull || true
dvc repro
dvc_repro_exit_status=$?
dvc commit
dvc push

echo '[DvC] commit dvc.lock if needed'
git add dvc.lock
dvc_lock_changed=$?
if [ $dvc_lock_changed -eq 0 ]; then
    git commit -m '[DvC] Add new dvc.lock'
    export GIT_SSL_NO_VERIFY=1
    git push https://CICD_TOKEN:"$CICD_TOKEN"@"$CI_SERVER_HOST"/"$CI_PROJECT_PATH".git HEAD:"$CI_COMMIT_REF_NAME"
fi

# If git_commit is 0 and dvc_repro_exit_status is 0, then exit 0. Else exit 1 but mention when git_commit is 1 and repro status 0 that this is not an error.
if [ $dvc_lock_changed -eq 0 ] && [ $dvc_repro_exit_status -eq 0 ]; then
    echo "[DvC] dvc repro executed successfully. Nothing to commit."
    exit 0
elif [ $dvc_lock_changed -eq 1 ] && [ $dvc_repro_exit_status -eq 0 ]; then
    echo "[DvC] dvc repro executed successfully, however dvc.lock changed. Pipeline restarts for CML. This is not an error!"
    exit 0
elif [ $dvc_lock_changed -eq 1 ] && [ $dvc_repro_exit_status -eq 1 ]; then
    echo "[DvC] dvc repro failed with exit status: $dvc_repro_exit_status. However dvc.lock changed. Pipeline restarts but will fail!"
    exit 0
else
    echo "[DvC] dvc repro failed with exit status: $dvc_repro_exit_status. No dvc.lock changes. Pipeline failed!"
    exit 1
fi
