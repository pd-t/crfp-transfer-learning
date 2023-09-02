import numpy as np 
from datasets import concatenate_datasets, Dataset

def balance(
        dataset: Dataset, 
        labels_per_category=None, 
        seed=42
        ) -> Dataset:
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
    
    max_labels_per_category = min(data.num_rows for data in sorted_datasets)
    
    if labels_per_category is not None:
        max_labels_per_category = min(max_labels_per_category, 
                                              labels_per_category)
        
    
    balanced_dataset = concatenate_datasets(
            [fd.shuffle(seed=seed).select(range(max_labels_per_category)) for fd in sorted_datasets]
            )

    return balanced_dataset, max_labels_per_category