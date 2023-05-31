import numpy as np
from datasets import concatenate_datasets, Dataset, load_from_disk
from pathlib import Path

def prepare(dataset: Dataset) -> Dataset:
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
    

if __name__ == '__main__':
    Path('data/prepare.dir').mkdir(parents=True, exist_ok=True)
    loaded_dataset = load_from_disk("data/load.dir/dataset")
    prepared_dataset = prepare(loaded_dataset)
    prepared_dataset.save_to_disk('data/prepare.dir/dataset')

