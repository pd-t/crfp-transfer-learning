import json
from pathlib import Path
import dvc.api
from datasets import load_from_disk

def evaluate(trainer_log, dataset, **kwargs):
    eval_loss = [item["eval_loss"] for item in trainer_log["trainer"] if "eval_loss" in item.keys()]
    eval_accuracy = [item["eval_accuracy"] for item in trainer_log["trainer"] if "eval_accuracy" in item.keys()]

    plots = [{"loss": loss, "accuracy": accuracy}  for loss, accuracy in zip(eval_loss, eval_accuracy)]
    with open('data/plots.json', 'w') as f:
        json.dump({"train": plots}, f, indent=4)

    metrics = {"loss": eval_loss[-1], "accuracy": eval_accuracy[-1]}
    with open('data/metrics.json', 'w') as f:
        json.dump({"train": metrics}, f, indent=4)
    
    # create an empty file in evaluate.dir
    Path('data/evaluate.dir/evaluate').touch()


if __name__=='__main__':
    Path('data/evaluate.dir').mkdir(parents=True, exist_ok=True)

    params = dvc.api.params_show(stages=['evaluate'])

    trained_dataset = load_from_disk("data/train.dir/dataset")
    # read trainer_log.json file as a dictionary
    with open('data/train.dir/trainer_log.json', 'r') as f:
        trainer_log = json.load(f)
    print(trainer_log)
    print(type(trainer_log))
    evaluate(trainer_log, trained_dataset, **params)
