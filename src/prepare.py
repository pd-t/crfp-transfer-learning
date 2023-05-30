import numpy as np
from datasets import concatenate_datasets, load_dataset, load_from_disk

dataset = load_from_disk("raw_dataset")


def compute_mean(image):
    image = np.array(image)
    return np.mean(image)


def compute_std(image):
    image = np.array(image)
    return np.std(image)


def add_mean_and_std(example):
    example["mean"] = compute_mean(example["image"])
    example["std"] = compute_std(example["image"])
    return example


dataset_mean_std = dataset.map(add_mean_and_std)
clean_dataset = dataset_mean_std.filter(lambda example: example["mean"] != 0)
labels = clean_dataset.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
labelids = [int(id) for id in id2label.keys()]
sorted_datasets = [
    clean_dataset.filter(lambda example: example["label"] == labelid, batch_size=1024)
    for labelid in labelids
]
min_examples = min(data.num_rows for data in sorted_datasets)
weighted_dataset = concatenate_datasets(
    [fd.shuffle(seed=42).select(range(min_examples)) for fd in sorted_datasets]
)
