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

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

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

def train(dataset):
    split_one = dataset.train_test_split(test_size=0.2, seed=42)
    split_two = split_one["test"].train_test_split(test_size=0.5, seed=42)
    train_test_valid_dataset = DatasetDict(
            {
                "train": split_one["train"],
                "validate": split_two["train"],
                "test": split_two["test"],
                }
            )
    train_test_valid_dataset.with_transform(Preprocess())

    checkpoint = "google/vit-base-patch16-224-in21k"
    labels = dataset.features["label"].names
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
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    
    trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_test_valid_dataset["train"],
            eval_dataset=train_test_valid_dataset["validate"],
            tokenizer=image_processor,
            compute_metrics=compute_metrics,
            )
    trainer.train()
    return train_test_valid_dataset, trainer

if __name__ == '__main__':
    Path('data/train.dir').mkdir(parents=True, exist_ok=True)
    prepared_dataset = load_from_disk("data/prepare.dir/dataset")
    trained_dataset, trianed_trainer = train(prepared_dataset)
    trained_dataset.save_to_disk('data/train.dir/dataset')
    trianed_trainer.save_model("data/train.dir/model")

