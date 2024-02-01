#!/bin/bash

CREDENTIAL_FILE="./.s3_credentials"

echo "[default]" > $CREDENTIAL_FILE
echo "aws_access_key_id=$S3_AWS_ACCESS_KEY_ID" >> $CREDENTIAL_FILE
echo "aws_secret_access_key=$S3_AWS_SECRET_ACCESS_KEY" >> $CREDENTIAL_FILE

