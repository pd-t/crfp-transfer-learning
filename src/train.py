import copy
import random
import dvc.api
from pathlib import Path
from shared.helpers import load_json, write_json
import datasets
import numpy as np
import evaluate
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)
from shared.learning import ModelMaker

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

#def evaluate(trainer_log):
#    write_values_to_json(trainer_log, "eval_accuracy", "accuracy.json")
#    write_values_to_json(trainer_log, "eval_loss", "loss.json")
#    write_metrics(trainer_log, ["eval_loss", "eval_accuracy"], "metrics.json")
#
def get_accuracy(trainer, dataset):
    prediction = trainer.predict(dataset)
    predictions = np.argmax(trainer.predict(dataset), axis=1)
    references = prediction.label_ids
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=references)

def select_training_data(dataset, limit, seed):
    return dataset.shuffle(seed=seed).select(range(limit))

def evaluate(dataset: datasets.DatasetDict, hyperparameters: dict, **kwargs):
    kwargs["learning_rate"] = hyperparameters["learning_rate"] 
    kwargs["batch_size"] = hyperparameters["per_device_train_batch_size"]
    
    model_maker = ModelMaker(
        checkpoints=kwargs["model"]["checkpoint"],
        output_dir="./data/tmp.dir/model")

    original_training_dataset = copy.deepcopy(dataset["train"])

    results = {}
    for limit in kwargs["data"]["limits"]:
        dataset["train"] = select_training_data(original_training_dataset, limit, kwargs["data"]["seed"])
        trainer = model_maker.get_trainer(dataset, trainer_args=kwargs["trainer"])
        trainer.train()
        accuracy = get_accuracy(trainer, dataset["test"])
        results[limit] = {"accuracy": accuracy}
    write_json("data/evaluate.dir/results.json", results)    

if __name__=='__main__':
    #tested_labels = load_json("data/train.dir/tested_labels.json")
    #trainer_log = load_json("data/train.dir/trainer_log.json")
    #evaluate(trainer_log)
    ## save tested labels
    #write_json("labels.json", tested_labels)

    #write_json('data/train.dir/tested_labels.json', tested_labels)
    #trainer_log = {"trainer": trained_trainer.state.log_history}
    #write_json('data/train.dir/trainer_log.json', trainer_log)

    Path('data/tmp.dir').mkdir(parents=True, exist_ok=True)
    Path('data/evaluate.dir').mkdir(parents=True, exist_ok=True)

    params = dvc.api.params_show(stages=['train'])
    
    prepared_dataset = datasets.DatasetDict.load_from_disk("data/prepare.dir/dataset")
    
    searched_hyperparameters = load_json("data/search.dir/hyperparameters.json")

    evaluate(prepared_dataset, searched_hyperparameters, **params)
