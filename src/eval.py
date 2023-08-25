from copy import copy
from shared.helpers import load_json, write_json

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)
def extract_values_from_log(trainer_log, key):
    return [item[key] for item in trainer_log["trainer"] if key in item.keys()]

def write_values_to_json(trainer_log, key, file_name):
    values = extract_values_from_log(trainer_log, key)
    epochs = extract_values_from_log(trainer_log, "epoch")
    data = {"train": [{"epoch": epoch, key: value}  for epoch, value in zip(epochs, values)]}
    write_json(file_name, data)

def write_metrics(trainer_log, keys, file_name):
    metrics = {key: extract_values_from_log(trainer_log, key)[-1] for key in keys}
    write_json(file_name, metrics)

def evaluate(trainer_log):
    write_values_to_json(trainer_log, "eval_accuracy", "accuracy.json")
    write_values_to_json(trainer_log, "eval_loss", "loss.json")
    write_metrics(trainer_log, ["eval_loss", "eval_accuracy"], "metrics.json")

def predict(trainer, dataset):
    prediction = trainer.predict(dataset)
    predicted_labels = np.argmax(prediction.predictions, axis=1)
    id2label = dataset.features["label"].names
    result = {"labels": [{"actual_label": id2label[a], "predicted_label": id2label[p]} for a, p in zip(prediction.label_ids, predicted_labels)]}
    return result

    kwargs["learning_rate"] = hyperparameters["learning_rate"] 
    kwargs["batch_size"] = hyperparameters["per_device_train_batch_size"]
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


if __name__=='__main__':

    #tested_labels = load_json("data/train.dir/tested_labels.json")
    #trainer_log = load_json("data/train.dir/trainer_log.json")
    #evaluate(trainer_log)
    ## save tested labels
    #write_json("labels.json", tested_labels)

    #write_json('data/train.dir/tested_labels.json', tested_labels)
    #trainer_log = {"trainer": trained_trainer.state.log_history}
    #write_json('data/train.dir/trainer_log.json', trainer_log)

    # load hyperparameters
    hyperparameters = load_json("data/train.dir/hyperparameters.json")
    # load dataset
    dataset = load_dataset("data/train.dir/dataset")
    
    Path('data/tmp.dir').mkdir(parents=True, exist_ok=True)
    Path('data/search.dir').mkdir(parents=True, exist_ok=True)

    params = dvc.api.params_show(stages=['search'])

    prepared_dataset = datasets.DatasetDict.load_from_disk("data/prepare.dir/dataset")

    searched_hyperparameters = search(prepared_dataset, **params)

    write_json(
        "data/search.dir/hyperparameters.json", 
        searched_hyperparameters
    )