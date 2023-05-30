#!/bin/bash

CREDENTIAL_FILE=".dvc/.minio_credentials"

echo "[default]" > $CREDENTIAL_FILE
echo "aws_access_key_id=$DVC_S3_AWS_ACCESS_KEY_ID" >> $CREDENTIAL_FILE
echo "aws_secret_access_key=$DVC_S3_AWS_SECRET_ACCESS_KEY" >> $CREDENTIAL_FILE

