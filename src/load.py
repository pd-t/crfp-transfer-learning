import os
import datasets
import boto3
from datasets import load_dataset
from zipfile import ZipFile
from pathlib import Path

def download_data_from_s3(
           endpoint_url: str,
           bucket: str,
           path: str,
           local_path: str,
           credential_path: str = '.dvc/.minio_credentials'):
    os.environ['AWS_SHARED_CREDENTIALS_FILE'] = credential_path

    session = boto3.Session()
    client = session.client(
            's3',
            endpoint_url=endpoint_url
            )
    with open(local_path, 'wb') as f:
        client.download_fileobj(bucket, path, f)

def unzip(source_file: str,
          target_path: str):
    with ZipFile(source_file, 'r') as zipObj:
        zipObj.extractall(path=target_path)

def load(
    s3_endpoint_url: str,
    s3_bucket: str,
    s3_path: str,
    local_path: str = 'data/data.zip'
) -> datasets.Dataset:

    print('load: download data')
    download_data_from_s3(
            endpoint_url=s3_endpoint_url,
            bucket=s3_bucket,
            path=s3_path,
            local_path=local_path
            )

    print('load: unzip data')
    unzip(
        source_file=local_path,
        target_path='data/tmp.dir'
    )

    print('load: create dataset')
    dataset = load_dataset("imagefolder", data_dir='data/tmp.dir', split="train")
    dataset = dataset.shuffle(seed=42).select(range(1000))
    return dataset

if __name__ == '__main__':
    Path('data/load.dir').mkdir(parents=True, exist_ok=True)
    Path('data/tmp.dir').mkdir(parents=True, exist_ok=True)
    
    endpoint_url = "https://storage.s3.mlops.wogra.com"
    bucket_name = "data"
    file_name = "dlr/Tapelegedaten2023.zip"
    
    loaded_dataset = load(endpoint_url, bucket_name, file_name)
    
    loaded_dataset.save_to_disk('data/load.dir/dataset')

