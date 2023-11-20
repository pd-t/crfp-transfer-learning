import os
from sklearn.metrics import confusion_matrix
import dvc.api
from pathlib import Path
from shared.helpers import load_json, write_json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def extract_values_from_log(trainer_log, key):
    return [item[key] for item in trainer_log["trainer"] if key in item.keys()]


def get_values_from_trainer_log(trainer_log, key):
    values = extract_values_from_log(trainer_log, key)
    epochs = extract_values_from_log(trainer_log, "epoch")
    data = {"train": [{"epoch": epoch, key: value}  for epoch, value in zip(epochs, values)]}
    return data


def make_trainer_plot(property, input_dir, output_dir): 
    trainer_log = load_json(input_dir + '/trainer_log.json')
    data = get_values_from_trainer_log(trainer_log, property)

    df = pd.DataFrame(data['train'])
    plt.clf()
    sns.lineplot(data=df, x="epoch", y=property)
    plt.savefig(output_dir + '/' + property + '.png')
    return df


def make_cm(input_dir, output_dir, normalize=None):
    data = load_json(input_dir + '/predicted_test_dataset.json')
    df = pd.DataFrame(data['labels'])
    cm = confusion_matrix(df["actual_label"], df["predicted_label"], normalize=normalize)
    plt.clf()
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.savefig(output_dir + '/cm_normalize_' + str(normalize) + '.png')


def make_plots_per_path(output_dir, input_path):
    input_subdir = str(input_path)
    tag = input_subdir.split('/')[-1]
    output_subpath = os.path.join(output_dir, tag)
    Path(output_subpath).mkdir(parents=True, exist_ok=True)
    output_subdir = str(output_subpath)
        
    make_cm(input_subdir, output_subdir, normalize="true")
    make_cm(input_subdir, output_subdir)

    losses = make_trainer_plot("eval_loss", input_subdir, output_subdir)
    accuracies = make_trainer_plot("eval_accuracy", input_subdir, output_subdir)
    return tag, losses, accuracies


def plot_overview(losses_and_accuracies, key, property):
    plt.clf()
    for tag, values in losses_and_accuracies.items():
        sns.lineplot(data=values[key], x="epoch", y=property, label=tag)
    plt.legend()
    plt.savefig(key + '.png')


def eval(input_dir, output_dir, **kwargs):
    # itearte over all subfolders in the input directory and for ech subfolder make an output subfolder
    losses_and_accuracies = {}
    for input_path in Path(input_dir).iterdir():
        tag, losses, accuracies = make_plots_per_path(output_dir, input_path)
        losses_and_accuracies[tag] = {"losses": losses, "accuracies": accuracies}  
    # plot all losses in a single plot
    plot_overview(losses_and_accuracies, "losses", "eval_loss")
    # plot all accuracies in a single plot
    plot_overview(losses_and_accuracies, "accuracies", "eval_accuracy")
    return get_metrics(dvc.api.params_show(stages=['train']))


def get_metrics(**kwargs):
    #get models, learning rate and batch size from model.json
    model = load_json("data/" + kwargs['logging_file'])
    metrics = {}
    metrics.update({"model": model["checkpoint"]})
    metrics.update({"learning_rate": model["learning_rate"]})
    metrics.update({"batch_size": model["per_device_train_batch_size"]})
    # iterate through models in models json and add labels per category as key to metrics with value of accuracy
    for model in model["models"]:
        metrics.update({model["labels_per_category"]: model["accuracy"]})
    return metrics


if __name__=='__main__':
    input_dir = 'data/train.dir'
    output_dir = 'data/eval.dir'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stage_params = dvc.api.params_show(stages=['eval'])
    eval_metrics = eval(input_dir, output_dir, **stage_params)
    write_json("data/" + stage_params['logging_file'], eval_metrics)
