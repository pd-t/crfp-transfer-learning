import copy
import shutil
import dvc.api
from pathlib import Path
from shared.helpers import load_json, write_json
from shared.data import balance_dataset
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


def get_labels(dataset):
    labels = {str(i): label for i, label in enumerate(get_id2labels(dataset))}
    return labels


def get_id2labels(dataset):
    id2label = dataset.features["label"].names
    return id2label


def get_accuracy(predicted_dataset):
    predictions = np.argmax(predicted_dataset.predictions, axis=1)
    references = predicted_dataset.label_ids
    unique_labels = np.unique(references)
    accuracy = evaluate.load("accuracy")
    accuracies = accuracy.compute(predictions=predictions, references=references)
    for label in unique_labels:
        label_indices = np.where(references == label)
        label_predictions = predictions[label_indices]
        accuracy = np.sum(label_predictions == label) / len(label_predictions)
        accuracies.update({str(label): accuracy})
    return accuracies


def select_training_data(
        dataset, 
        labels_per_category, 
        seed
        ):
    return dataset.shuffle(seed=seed).select(range(labels_per_category))


def train_trainer(dataset, model_maker, path, kwargs):
    temp_dir = 'data/tmp.dir'
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    trainer = model_maker.get_trainer(
            dataset, 
            trainer_args=kwargs["trainer"], 
            save_best_model=True, 
            output_dir=temp_dir,
            )
    trainer.train()
    # join path and filename
    log_filename = path + '/trainer_log.json'
    save_trainer_log(trainer, log_filename)
    shutil.rmtree(temp_dir, ignore_errors=False, onerror=None)
    return trainer


def test_trainer(dataset, model_maker, trainer, model_dir):
    predicted_test_dataset = model_maker.predict(trainer, dataset['test'])
    save_labels(predicted_test_dataset, get_id2labels(dataset['test']), model_dir + '/predicted_test_dataset.json')
    return get_accuracy(predicted_test_dataset)


def train(dataset: datasets.DatasetDict, hyperparameters: dict, train_dir: str, **kwargs):
    kwargs["learning_rate"] = hyperparameters["learning_rate"] 
    kwargs["per_device_train_batch_size"] = hyperparameters["per_device_train_batch_size"]
    
    model_maker = ModelMaker(checkpoints=kwargs["model"]["checkpoint"])

    original_training_dataset = copy.deepcopy(dataset["train"])

    metrics = {}
    metrics.update({"checkpoint": kwargs["model"]["checkpoint"]})
    metrics.update({"learning_rate": kwargs["learning_rate"]})
    metrics.update({"batch_size": kwargs["per_device_train_batch_size"]})
    metrics.update({"labels": get_labels(dataset["test"])})
    metrics.update({"models": []})

    for labels_per_category in kwargs["model"]["labels_per_category"]:
        model_metrics = {}

        model_dir = train_dir + '/' + str(labels_per_category)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        dataset["train"], selected_labels_per_category = balance_dataset(
            dataset=original_training_dataset,
            labels_per_category=labels_per_category, 
            seed=kwargs["data"]["seed"]
            )
        model_metrics.update({"labels_per_category": selected_labels_per_category})

        trainer = train_trainer(dataset, model_maker, model_dir, kwargs)
        
        accuracies = test_trainer(dataset, model_maker, trainer, model_dir)
        model_metrics.update(accuracies)

        metrics["models"].append(model_metrics)
    return metrics


if __name__=='__main__':
    train_dir = 'data/train.dir'
    Path(train_dir).mkdir(parents=True, exist_ok=True)

    stage_params = dvc.api.params_show(stages=['train'])
    
    prepared_dataset = datasets.DatasetDict.load_from_disk("data/prepare.dir/dataset")
    
    searched_hyperparameters = load_json("data/search.dir/hyperparameters.json")

    trained_metrics = train(prepared_dataset, searched_hyperparameters, train_dir, **stage_params)

    write_json("data/" + stage_params['model']['logging_file'], trained_metrics)