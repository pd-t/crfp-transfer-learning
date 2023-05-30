import boto3
from datasets import load_dataset


def download_data(endpoint_url, bucket_name, file_name):
    s3 = boto3.resource("s3", endpoint_url=endpoint_url)
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(file_name, "data.zip")
    get_ipython().system("rm -r data")
    get_ipython().system("unzip data.zip -d data")
    return "data"


endpoint_url = "https://storage.s3.mlops.wogra.com"
bucket_name = "data"
file_name = "dlr/Tapelegedaten2023.zip"
data_directory = download_data(endpoint_url, bucket_name, file_name)
dataset = load_dataset("imagefolder", data_dir=data_directory, split="train")
