import os
import datasets
import boto3
from datasets import load_dataset
from zipfile import ZipFile
from pathlib import Path
import dvc.api

def download_data_from_s3(
           endpoint_url: str,
           bucket: str,
           path: str,
           local_path: str):

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
    local_path: str,
    **kwargs
) -> datasets.Dataset:
    print('load: download data')
    download_data_from_s3(
            endpoint_url=kwargs["remote"]["s3_endpoint_url"],
            bucket=kwargs["remote"]["s3_bucket"],
            path=kwargs["remote"]["s3_path"],
            local_path=local_path
            )

    print('load: unzip data')
    unzip(
        source_file=local_path,
        target_path='data/tmp.dir/images'
    )

    print('load: create dataset')
    dataset = load_dataset("imagefolder", data_dir='data/tmp.dir/images', split="train")
    if kwargs["select"] != 'Complete':
        dataset = dataset.shuffle(kwargs["seed"]).select(range(kwargs["select"]))
    return dataset

if __name__ == '__main__':
    Path('data/load.dir').mkdir(parents=True, exist_ok=True)
    Path('data/tmp.dir').mkdir(parents=True, exist_ok=True)
    
    params = dvc.api.params_show(stages=['load'])
    
    loaded_dataset = load('data/tmp.dir/data.zip', **params['data'])
    
    loaded_dataset.save_to_disk('data/load.dir/dataset')
