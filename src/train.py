from pathlib import Path
from torchvision import transforms
import numpy as np
from datasets import DatasetDict, load_from_disk
import evaluate
from pathlib import Path
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)
import dvc.api
import json

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
    
    
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def predict(trainer, dataset):
    prediction = trainer.predict(dataset)
    predicted_labels = np.argmax(prediction.predictions, axis=1)
    id2label = dataset.features["label"].names
    result = {"labels": [{"actual_label": id2label[a], "predicted_label": id2label[p]} for a, p in zip(prediction.label_ids, predicted_labels)]}
    return result

def train_test_split(dataset, **kwargs):
    split_one = dataset.train_test_split(
        test_size=kwargs["eval-test-size"], 
        seed=kwargs["seed"]
    )
    split_two = split_one["test"].train_test_split(
        test_size=kwargs["test-size"], 
        seed=kwargs["seed"]
    )
    dataset = DatasetDict(
            {
                "train": split_one["train"],
                "validate": split_two["train"],
                "test": split_two["test"],
                }
            )
    return dataset

def train(dataset, **kwargs):
    dataset = dataset.with_transform(Preprocess())
    checkpoint = kwargs["checkpoint"]

    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label 

    model = AutoModelForImageClassification.from_pretrained(
            checkpoint,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id
            )
    model.to('cuda')

    training_args = TrainingArguments(
            output_dir="./data/tmp.dir/model",
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
        learning_rate=kwargs["learning_rate"],
        per_device_train_batch_size=kwargs["batch_size"],
        gradient_accumulation_steps=kwargs["gradient_accumulation_steps"],
        per_device_eval_batch_size=kwargs["batch_size"],
        num_train_epochs=kwargs["num_train_epochs"],
        warmup_ratio=kwargs["warmup_ratio"],
        logging_steps=kwargs["logging_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    data_collator = DefaultDataCollator()
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    
    trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validate"],
            tokenizer=image_processor,
            compute_metrics=compute_metrics,
            )
    trainer.train()

    test_labels = predict(trainer, dataset["test"])
    return trainer, test_labels

def write_json(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    Path('data/tmp.dir').mkdir(parents=True, exist_ok=True)
    Path('data/train.dir').mkdir(parents=True, exist_ok=True)

    params = dvc.api.params_show(stages=['train'])

    prepared_dataset = load_from_disk("data/prepare.dir/dataset")
    splitted_dataset = train_test_split(prepared_dataset, **params['model'])
    splitted_dataset.save_to_disk(
        'data/train.dir/dataset', 
        )

    trained_trainer, tested_labels = train(splitted_dataset, **params['model'])
    trained_trainer.save_model("data/train.dir/model")

    trainer_log = {"trainer": trained_trainer.state.log_history}
    write_json('data/train.dir/trainer_log.json', trainer_log)
    write_json('data/tested_labels.json', tested_labels)
 