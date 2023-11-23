import os
from sklearn.metrics import confusion_matrix
import dvc.api
from pathlib import Path
from shared.helpers import load_json, write_json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score


def extract_values_from_log(trainer_log, key):
    return [item[key] for item in trainer_log["trainer"] if key in item.keys()]


def get_values_from_trainer_log(trainer_log, key):
    values = extract_values_from_log(trainer_log, key)
    epochs = extract_values_from_log(trainer_log, "epoch")
    data = {"train": [{"epoch": epoch, key: value}  for epoch, value in zip(epochs, values)]}
    return data


def plot_trainer_property(property, trainer_dir, eval_dir): 
    data = get_trainer_property(property, trainer_dir)
    df = pd.DataFrame(data)
    plt.clf()
    sns.lineplot(data=df, x="epoch", y=property)
    plt.savefig(eval_dir + '/' + property + '.png')


def get_trainer_property(property, trainer_dir):
    trainer_log = load_json(trainer_dir + '/trainer_log.json')
    data = get_values_from_trainer_log(trainer_log, property)
    return data['train']


def plot_cm(trainer_dir, eval_dir, normalize=None):
    data = load_json(trainer_dir + '/predicted_test_dataset.json')
    df = pd.DataFrame(data['labels'])
    cm = confusion_matrix(df["actual_label"], df["predicted_label"], normalize=normalize)
    plt.clf()
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.savefig(eval_dir + '/cm_normalize_' + str(normalize) + '.png')


def plot_overview(data, property):
    plt.clf()
    for tag, values in data.items():
        df = pd.DataFrame(values[property])
        sns.lineplot(data=df, x="epoch", y=property, label=tag)
    plt.legend()
    plt.savefig("plots/" + property + '.png')


def prepare_output(output_dir, input_path):
    input_subdir = str(input_path)
    tag = input_subdir.split('/')[-1]
    output_subpath = os.path.join(output_dir, tag)
    Path(output_subpath).mkdir(parents=True, exist_ok=True)
    output_subdir = str(output_subpath)
    return input_subdir, output_subdir


def eval(input_dir, output_dir, **kwargs):
    collected_data = {}
    for input_path in Path(input_dir).iterdir():
        trainer_dir, eval_dir = prepare_output(output_dir, input_path)

        plot_cm(trainer_dir, eval_dir, normalize="true")
        plot_cm(trainer_dir, eval_dir)

        trainer_metrics = {}
        for property in ["eval_loss", "eval_accuracy"]:
            plot_trainer_property(property, trainer_dir, eval_dir)
            trainer_metrics[property] = get_trainer_property(property, trainer_dir)
        
        tag = input_path.name
        collected_data[tag] = trainer_metrics
    # plot all losses in a single plot
    plot_overview(collected_data, "eval_loss")
    # plot all accuracies in a single plot
    plot_overview(collected_data, "eval_accuracy")
    metrics = {}
    metrics["f1-scores"] = get_f1scores(input_dir)
    return metrics


def get_f1scores(trainer_dir, average='weighted'):
    scores = {}
    for input_path in Path(trainer_dir).iterdir():
        tag = input_path.name
        predictions = load_json(input_path / 'predicted_test_dataset.json')
        df = pd.DataFrame(predictions["labels"])
        scores[tag] = f1_score(df["actual_label"], df["predicted_label"], average=average)
    return scores


if __name__=='__main__':
    input_dir = 'data/train.dir'
    output_dir = 'data/eval.dir'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path('plots').mkdir(parents=True, exist_ok=True)

    stage_params = dvc.api.params_show(stages=['eval'])
    eval_metrics = eval(input_dir, output_dir, **stage_params)
    write_json("data/" + stage_params['report']['logging_file'], eval_metrics)
