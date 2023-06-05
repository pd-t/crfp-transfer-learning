import json
from pathlib import Path
import dvc.api
from datasets import load_from_disk

def evaluate(trainer, dataset, **kwargs):
    # make list of 10 values from 0 to 9 and multiply it by scaling
    x = [{"accuracy": i, "loss": (10-i)* params['model']['scaling']}  for i in range(10)]
    metrics = {'train': x}

    #save x to json file
    with open('data/metrics.json', 'w') as f:
        json.dump(metrics, f)


    #save x to json file
    with open('data/tests.json', 'w') as f:
        json.dump(metrics, f)


if __name__=='__main__':
    Path('data/evaluate.dir').mkdir(parents=True, exist_ok=True)

    params = dvc.api.params_show(stages=['evaluate'])

    trained_dataset = load_from_disk("data/train.dir/dataset")
    evaluate(None, trained_dataset, **params)
