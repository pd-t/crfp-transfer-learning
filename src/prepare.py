import numpy as np
from datasets import concatenate_datasets, Dataset, load_from_disk
from pathlib import Path


def preprocess(image):
    # convert image to tensor
    tensor = transforms.ToTensor()(image)
    tensor = tensor - np.mean(tensor.numpy()) + 128
    tensor[tensor < 0] = 0
    tensor[tensor > 255] = 255
    tensor = tensor / 255
    # repeat the tensor two ore times to get 3 channels and use the repeat function
    tensor = tensor.repeat(3, 1, 1)
    return tensor


class Preprocess:
    def __init__(self):
        self.transforms = transforms.Compose(
            [
                transforms.Lambda(preprocess),
                transforms.CenterCrop(224),
            ]
        )

    def __call__(self, example_batch):
        example_batch["pixel_values"] = [
            self.transforms(img) for img in example_batch["image"]
        ]
        del example_batch["image"]
        return example_batch

def prepare(dataset: Dataset) -> Dataset:
    dataset = dataset.filter(lambda example: np.mean(example["image"]) != 0)
    
    dataset = dataset.with_transform(Preprocess())

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

