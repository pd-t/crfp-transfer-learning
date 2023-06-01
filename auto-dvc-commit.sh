#!/bin/bash

echo '[DvC] execute dvc pipeline'
dvc pull || true
dvc repro
dvc_repro_exit_status=$?
dvc commit
dvc push
git add dvc.lock

if [ $dvc_repro_exit_status -eq 1 ]; then
    commit_message="[DvC] dvc repro failed, update dvc.lock [skip ci]"
else
    commit_message="[DvC] dvc repro executed successfully, update dvc.lock"
fi
git commit -m "$commit_message"
git_add_dvc_lock=$?

if [ $git_add_dvc_lock -eq 0 ]; then
    export GIT_SSL_NO_VERIFY=1
    git push https://CICD_TOKEN:"$CICD_TOKEN"@"$CI_SERVER_HOST"/"$CI_PROJECT_PATH".git HEAD:"$CI_COMMIT_REF_NAME"
fi

if [ $git_add_dvc_lock -eq 0 ] && [ $dvc_repro_exit_status -eq 0 ]; then
    echo "[DvC] dvc repro executed successfully. Nothing to commit. Ready for CML."
    exit 0
elif [ $git_add_dvc_lock -eq 1 ] && [ $dvc_repro_exit_status -eq 0 ]; then
    echo "[DvC] dvc repro executed successfully, however dvc.lock changed. Pipeline restarts for CML. This is not an error!"
    exit 0
elif [ $git_add_dvc_lock -eq 1 ] && [ $dvc_repro_exit_status -eq 1 ]; then
    echo "[DvC] dvc repro failed with exit status: $dvc_repro_exit_status. But dvc.lock changed."
    exit 0
else
    echo "[DvC] dvc repro failed with exit status: $dvc_repro_exit_status. No dvc.lock changes."
    exit 1
fi
