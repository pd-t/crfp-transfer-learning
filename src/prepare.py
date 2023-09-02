import dvc.api
from datasets import DatasetDict, Dataset, load_from_disk
from pathlib import Path
from shared.helpers import write_json
from shared.data import balance


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


def save_data_metrics(dataset: Dataset, labels_per_category: int):
    data_metrics = {
            split: dataset[split].num_rows 
            for split in dataset.keys()
            }
    data_metrics["labels_per_category"] = labels_per_category
    write_json("data.json", data_metrics)

def prepare(dataset: Dataset, **kwargs) -> Dataset:
    splitted_dataset = train_test_split(
        dataset, 
        **params['data']
    )
    balanced_train_dataset, labels_per_category = balance(splitted_dataset["train"])
    save_data_metrics(splitted_dataset, labels_per_category)
    splitted_dataset["train"] = balanced_train_dataset
    return splitted_dataset

if __name__ == '__main__':
    Path('data/prepare.dir').mkdir(parents=True, exist_ok=True)
    
    params = dvc.api.params_show(stages=['prepare'])

    loaded_dataset = load_from_disk("data/load.dir/dataset")
    prepared_dataset = prepare(loaded_dataset, **params['data'])
    prepared_dataset.save_to_disk('data/prepare.dir/dataset')
