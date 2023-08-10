import numpy as np
import dvc.api
from datasets import concatenate_datasets, DatasetDict, Dataset, load_from_disk
from pathlib import Path

def train_test_split(dataset, **kwargs):
    dev_test_split = dataset.train_test_split(
        train_size=kwargs["development-size"], 
        seed=kwargs["seed"]
    )
    train_eval_split = dev_test_split["test"].train_test_split(
        train_size=kwargs["development-train-size"], 
        seed=kwargs["seed"]
    )
    dataset = DatasetDict(
        {
            "train": train_eval_split["train"],
            "validate": train_eval_split["test"],
            "test": dev_test_split["test"],
        }
    )
    return dataset

def balance(dataset: Dataset) -> Dataset:
    dataset = dataset.filter(lambda example: np.mean(example["image"]) != 0)

    labels = dataset.features["label"].names
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    labelids = [int(id) for id in id2label.keys()]

    sorted_datasets = [
            dataset.filter(lambda example: example["label"] == labelid, batch_size=1024)
            for labelid in labelids
            ]
    
    min_examples = min(data.num_rows for data in sorted_datasets)
    
    weighted_dataset = concatenate_datasets(
            [fd.shuffle(seed=42).select(range(min_examples)) for fd in sorted_datasets]
            )

    return weighted_dataset

def prepare(dataset: Dataset, **kwargs) -> Dataset:
    splitted_dataset = train_test_split(
        dataset, 
        **params['data']
    )
    balanced_train_dataset = balance(splitted_dataset["train"])
    splitted_dataset["train"] = balanced_train_dataset
    return splitted_dataset

if __name__ == '__main__':
    Path('data/prepare.dir').mkdir(parents=True, exist_ok=True)
    
    params = dvc.api.params_show(stages=['prepare'])

    loaded_dataset = load_from_disk("data/load.dir/dataset")
    prepared_dataset = prepare(loaded_dataset, **params['data'])
    prepared_dataset.save_to_disk('data/prepare.dir/dataset')
