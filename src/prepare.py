import dvc.api
from datasets import DatasetDict, Dataset, load_from_disk
from pathlib import Path
from shared.helpers import write_json
from collections import Counter
from shared.data import get_labels

def train_test_split(dataset, **kwargs):
    dev_test_split = dataset.train_test_split(
        test_size=kwargs["test-size"], 
        seed=kwargs["seed"]
    )
    train_eval_split = dev_test_split["train"].train_test_split(
        train_size=kwargs["train-size"], 
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


def get_metrics(dataset: Dataset, **kwargs):
    metrics = {}
    labels = get_labels(dataset["test"])
    for split_name, split in dataset.items():
        samples_per_label = dict(Counter(split['label']))
        # replace key of samples_per_label with key of labels using the value if labels
        labels_per_label = {labels[str(key)]: value for key, value in samples_per_label.items()}
        # alphabetically sort labels_per_label
        labels_per_label = dict(sorted(labels_per_label.items()))
        metrics[split_name] = {
            'size': len(split),
            'labels': labels_per_label,
        }
    return metrics


def prepare(dataset: Dataset, **kwargs):
    splitted_dataset = train_test_split(
        dataset, 
        **params['data']
    )
    metrics = get_metrics(splitted_dataset, **params['data'])
    return splitted_dataset, metrics

if __name__ == '__main__':
    Path('data/prepare.dir').mkdir(parents=True, exist_ok=True)
    
    params = dvc.api.params_show(stages=['prepare'])

    loaded_dataset = load_from_disk("data/load.dir/dataset")
    prepared_dataset, prepared_metrics = prepare(loaded_dataset, **params['data'])
    prepared_dataset.save_to_disk("data/prepare.dir/dataset")

    write_json("data/" + params['data']['logging_file'], prepared_metrics)