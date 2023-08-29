import copy
import dvc.api
from pathlib import Path
from shared.helpers import load_json, write_json
from shared.learning import ModelMaker
import datasets
import numpy as np
import evaluate


def save_labels(predicted_dataset, id2label, filename):
    predicted_labels = np.argmax(predicted_dataset.predictions, axis=1)
    labels = {"labels": [{"actual_label": id2label[a], "predicted_label": id2label[p]} for a, p in zip(predicted_dataset.label_ids, predicted_labels)]}
    write_json(filename, labels)


def save_trainer_log(trainer, filename):
    log = {"trainer": trainer.state.log_history}
    write_json(filename, log)


def get_accuracy(predicted_dataset):
    predictions = np.argmax(predicted_dataset.predictions, axis=1)
    references = predicted_dataset.label_ids
    #get unique labels in references
    unique_labels = np.unique(references)
    #get accuracy for each label
    accuracy = evaluate.load("accuracy")
    accuracies = accuracy.compute(predictions=predictions, references=references)
    for label in unique_labels:
        label_indices = np.where(references == label)
        label_predictions = predictions[label_indices]
        accuracy = np.sum(label_predictions == label) / len(label_predictions)
        accuracies.update({str(label): accuracy})
    return accuracies


def select_training_data(dataset, limit, seed):
    return dataset.shuffle(seed=seed).select(range(limit))


def train(dataset: datasets.DatasetDict, hyperparameters: dict, **kwargs):
    kwargs["learning_rate"] = hyperparameters["learning_rate"] 
    kwargs["batch_size"] = hyperparameters["per_device_train_batch_size"]
    
    id2label = dataset["test"].features["label"].names
    labels = {str(i): label for i, label in enumerate(id2label)}
    
    model_maker = ModelMaker(
        checkpoints=kwargs["model"]["checkpoint"],
        output_dir="./data/tmp.dir/model")

    original_training_dataset = copy.deepcopy(dataset["train"])

    metrics = {}
    metrics.update({"model": kwargs["model"]["checkpoint"]})
    metrics.update({"learning_rate": hyperparameters["learning_rate"]})
    metrics.update({"batch_size": hyperparameters["per_device_train_batch_size"]})
    metrics.update({"labels": labels})
    metrics.update({"limits": []})
    for limit in kwargs["data"]["limits"]:
        limit_metrics = {}
        limit_metrics.update({"limit": limit})
        dataset["train"] = select_training_data(original_training_dataset, limit, kwargs["data"]["seed"])
        trainer = model_maker.get_trainer(dataset, trainer_args=kwargs["trainer"], save_strategy='yes')
        trainer.train()
        save_trainer_log(trainer, 'data/train.dir/trainer_log_' + str(limit) + '.json')
        predicted_test_dataset = model_maker.predict(trainer, dataset["test"].select(range(100)))
        save_labels(predicted_test_dataset, id2label, 'data/train.dir/labels_' + str(limit) + '.json')
        accuracy = get_accuracy(predicted_test_dataset)
        limit_metrics.update(accuracy)
        metrics["limits"].append(limit_metrics)
    write_json("metrics.json", metrics)


if __name__=='__main__':
    Path('data/tmp.dir').mkdir(parents=True, exist_ok=True)
    Path('data/train.dir').mkdir(parents=True, exist_ok=True)

    stage_params = dvc.api.params_show(stages=['train'])
    
    prepared_dataset = datasets.DatasetDict.load_from_disk("data/prepare.dir/dataset")
    
    searched_hyperparameters = load_json("data/search.dir/hyperparameters.json")

    train(prepared_dataset, searched_hyperparameters, **stage_params)
