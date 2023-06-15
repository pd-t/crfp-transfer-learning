import json

def write_json(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

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

if __name__=='__main__':
    trainer_log = load_json("data/train.dir/trainer_log.json")
    evaluate(trainer_log)

