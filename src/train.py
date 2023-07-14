from pathlib import Path
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
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

def cost(config):
    z = config["x"]**2 + config["y"]**2    
    tune.report(mean_accuracy=z)

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
    metrics = accuracy.compute(predictions=predictions, references=labels)
    return metrics

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

def ray_hp_space(learning_rate_min, learning_rate_max, batch_sizes, trial):
    return {
        "learning_rate": tune.loguniform(learning_rate_min, 
                                         learning_rate_max),
        "per_device_train_batch_size": tune.choice(batch_sizes),
    }

def model_init(dataset, checkpoint):
    labels, label2id, id2label = get_labels(dataset) 
    model = AutoModelForImageClassification.from_pretrained(
            checkpoint,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id
            )
    model.to('cuda')
    return model

def get_labels(dataset):
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    return labels,label2id,id2label

def get_training_args(
        params, 
        output_dir="./data/train.dir/model",
        save_strategy="epoch"
        ):
    if save_strategy == "no":
        load_best_model_at_end = False
    else:
        load_best_model_at_end = True
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy=save_strategy,
        save_total_limit=2,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="accuracy",
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=params["batch_size"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        per_device_eval_batch_size=params["batch_size"],
        num_train_epochs=params["num_train_epochs"],
        warmup_ratio=params["warmup_ratio"],
        logging_steps=params["logging_steps"]
    )
    return training_args

def train(dataset, **kwargs):
    dataset = dataset.with_transform(Preprocess())
    data_collator = DefaultDataCollator()
    image_processor = AutoImageProcessor.from_pretrained(kwargs["checkpoint"])
    
    search_trainer = Trainer(
            model_init=lambda: model_init(dataset, kwargs["checkpoint"]),
            args=get_training_args(kwargs, save_strategy="no", output_dir="./"),
            data_collator=data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validate"],
            tokenizer=image_processor,
            compute_metrics=compute_metrics,
            )

    search_result = search_trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        hp_space=lambda trial: ray_hp_space(
            kwargs["hyperparameters"]["learning_rate_min"],
            kwargs["hyperparameters"]["learning_rate_max"],
            kwargs["hyperparameters"]["batch_sizes"],
            trial),
        scheduler=AsyncHyperBandScheduler(
            metric="objective", 
            mode="max", 
            max_t=kwargs["asha"]["max_t"], 
            grace_period=kwargs["asha"]["grace_period"], 
            reduction_factor=kwargs["asha"]["reduction_factor"]
            ),
        resources_per_trial={
            "cpu": kwargs["asha"]["trial_cpus"], 
            "gpu": kwargs["asha"]["trial_gpus"]
            },
        n_trials=kwargs["asha"]["n_trials"],
        local_dir="./data/train.dir",
        name="tune_asha",
        log_to_file=True
        )
    
    optimized_parameters = search_result.hyperparameters
    kwargs["learning_rate"] = optimized_parameters["learning_rate"] 
    kwargs["batch_size"] = optimized_parameters["per_device_train_batch_size"]
    optimized_trainer = Trainer(
            model_init=lambda: model_init(dataset, kwargs["checkpoint"]),
            args=get_training_args(kwargs),
            data_collator=data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validate"],
            tokenizer=image_processor,
            compute_metrics=compute_metrics,
            )

    optimized_trainer.train()
    test_labels = predict(optimized_trainer, dataset["test"])
    return optimized_trainer, test_labels

def write_json(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    Path('data/tmp.dir').mkdir(parents=True, exist_ok=True)
    Path('data/train.dir').mkdir(parents=True, exist_ok=True)

    params = dvc.api.params_show(stages=['train'])

    prepared_dataset = load_from_disk("data/prepare.dir/dataset")
    splitted_dataset = train_test_split(prepared_dataset, **params['model'])

    trained_trainer, tested_labels = train(splitted_dataset, **params['model'])

    trainer_log = {"trainer": trained_trainer.state.log_history}
    write_json('data/train.dir/trainer_log.json', trainer_log)
    write_json('data/train.dir/tested_labels.json', tested_labels)
 
