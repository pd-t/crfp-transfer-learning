import copy
import shutil
import dvc.api
from pathlib import Path
from shared.helpers import load_json, write_json
from shared.data import balance_dataset, get_id2label
from shared.learning import ModelMaker
import datasets
import numpy as np
import os


def make_prediction_readable(predicted_dataset, id2label):
    predicted_labels = np.argmax(predicted_dataset.predictions, axis=1)
    labels = {"labels": [{"actual_label": id2label[a], "predicted_label": id2label[p]} for a, p in zip(predicted_dataset.label_ids, predicted_labels)]}
    return labels


def save_trainer_log(trainer, filename):
    log = {"trainer": trainer.state.log_history}
    write_json(filename, log)


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


def make_prediction(dataset, model_maker, trainer):
    return model_maker.predict(trainer, dataset)


def create_model_dir(path, labels_per_category):
    model_dir = path + '/' + str(labels_per_category)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    return model_dir


def save_predictions(dataset, model_dir, predicted_test_dataset):
    id2label = get_id2label(dataset['test'])
    labels = make_prediction_readable(predicted_test_dataset, id2label)
    write_json(model_dir + '/predicted_test_dataset.json', labels)


def train(dataset: datasets.DatasetDict, hyperparameters: dict, train_dir: str, **kwargs):
    kwargs["learning_rate"] = hyperparameters["learning_rate"] 
    kwargs["per_device_train_batch_size"] = hyperparameters["per_device_train_batch_size"]
    
    model_maker = ModelMaker(checkpoints=kwargs["model"]["checkpoint"])

    original_training_dataset = copy.deepcopy(dataset["train"])

    metrics = {}
    metrics.update({"checkpoint": kwargs["model"]["checkpoint"]})
    metrics.update({"learning_rate": kwargs["learning_rate"]})
    metrics.update({"batch_size": kwargs["per_device_train_batch_size"]})

    model_label_per_categories = []
    for labels_per_category in kwargs["model"]["labels_per_category"]:

        model_dir = create_model_dir(train_dir, labels_per_category)

        dataset["train"], selected_labels_per_category = balance_dataset(
            dataset=original_training_dataset,
            labels_per_category=labels_per_category, 
            seed=kwargs["data"]["seed"]
            )
        model_label_per_categories.append(selected_labels_per_category)

        trainer = train_trainer(dataset, model_maker, model_dir, kwargs)
        
        predicted_test_dataset = make_prediction(dataset['test'], model_maker, trainer)
        save_predictions(dataset, model_dir, predicted_test_dataset)

    metrics.update({"models": model_label_per_categories})
    return metrics


if __name__=='__main__':
    print("Visible CUDA Devices: " + os.environ["CUDA_VISIBLE_DEVICES"])

    train_dir = 'data/train.dir'
    Path(train_dir).mkdir(parents=True, exist_ok=True)

    stage_params = dvc.api.params_show(stages=['train'])
    
    prepared_dataset = datasets.DatasetDict.load_from_disk("data/prepare.dir/dataset")
    
    searched_hyperparameters = load_json("data/search.dir/hyperparameters.json")

    trained_metrics = train(prepared_dataset, searched_hyperparameters, train_dir, **stage_params)

    write_json("data/" + stage_params['model']['logging_file'], trained_metrics)