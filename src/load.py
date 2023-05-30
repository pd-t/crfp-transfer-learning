import boto3
from datasets import load_dataset

def download_data(
           endpoint: str,
           bucket: str,
           path: str,
           credential_path: str = '.dvc/.minio_credentials'):
    os.environ['AWS_SHARED_CREDENTIALS_FILE'] = credential_path

    s3 = boto3.resource("s3", endpoint_url=endpoint_url)
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(path, "data.zip")
    return "data"

endpoint_url = "https://storage.s3.mlops.wogra.com"
bucket_name = "data"
file_name = "dlr/Tapelegedaten2023.zip"
data_directory = download_data(endpoint_url, bucket_name, file_name)
dataset = load_dataset("imagefolder", data_dir=data_directory, split="train")
