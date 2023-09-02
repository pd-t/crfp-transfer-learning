import copy
import shutil
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
    unique_labels = np.unique(references)
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
    
    model_maker = ModelMaker(checkpoints=kwargs["model"]["checkpoint"])

    original_training_dataset = copy.deepcopy(dataset["train"])

    metrics = {}
    metrics.update({"model": kwargs["model"]["checkpoint"]})
    metrics.update({"learning_rate": hyperparameters["learning_rate"]})
    metrics.update({"batch_size": hyperparameters["per_device_train_batch_size"]})
    metrics.update({"labels": labels})
    metrics.update({"labels_per_category": []})

    for lpc in kwargs["model"]["labels_per_category"]:
        temp_dir = 'data/tmp.dir'
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        dataset["train"] = select_training_data(original_training_dataset, lpc, kwargs["data"]["seed"])

        trainer = model_maker.get_trainer(
            dataset, 
            trainer_args=kwargs["trainer"], 
            save_best_model=True, 
            output_dir=temp_dir,
            )
        trainer.train()
        save_trainer_log(trainer, 'data/train.dir/trainer_log_' + str(lpc) + '.json')
        
        model_metrics = {}
        model_metrics.update({"labels_per_category": lpc})
        predicted_test_dataset = model_maker.predict(trainer, dataset["test"].select(range(100)))
        save_labels(predicted_test_dataset, id2label, 'data/train.dir/labels_' + str(lpc) + '.json')
        accuracy = get_accuracy(predicted_test_dataset)
        model_metrics.update(accuracy)
        metrics["labels_per_category"].append(model_metrics)
        
        shutil.rmtree(temp_dir, ignore_errors=False, onerror=None)
    return metrics

if __name__=='__main__':
    Path('data/train.dir').mkdir(parents=True, exist_ok=True)

    stage_params = dvc.api.params_show(stages=['train'])
    
    prepared_dataset = datasets.DatasetDict.load_from_disk("data/prepare.dir/dataset")
    
    searched_hyperparameters = load_json("data/search.dir/hyperparameters.json")

    trained_metrics = train(prepared_dataset, searched_hyperparameters, **stage_params)
    write_json("models.json", trained_metrics)