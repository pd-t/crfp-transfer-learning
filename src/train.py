import numpy as np
import transformers
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

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
checkpoint = "google/vit-base-patch16-224-in21k"
labels = clean_dataset.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
labelids = [int(id) for id in id2label.keys()]
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)
batch_size = 14 * 16
training_args = TrainingArguments(
    output_dir="my_tapelegen_model",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)
data_collator = DefaultDataCollator()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


image_processor = AutoImageProcessor.from_pretrained(checkpoint)
sorted_datasets = [
    clean_dataset.filter(lambda example: example["label"] == labelid, batch_size=1024)
    for labelid in labelids
]
min_examples = min(data.num_rows for data in sorted_datasets)
weighted_dataset = concatenate_datasets(
    [fd.shuffle(seed=42).select(range(min_examples)) for fd in sorted_datasets]
)
split_one = weighted_dataset.train_test_split(test_size=0.2, seed=42)
split_two = split_one["test"].train_test_split(test_size=0.5, seed=42)
train_test_valid_dataset = DatasetDict(
    {
        "train": split_one["train"],
        "validate": split_two["train"],
        "test": split_two["test"],
    }
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_test_valid_dataset["train"],
    eval_dataset=train_test_valid_dataset["validate"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)
