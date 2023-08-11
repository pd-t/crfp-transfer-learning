from pathlib import Path
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
import datasets
from pathlib import Path
import dvc.api
from shared.helpers import write_json
from shared.learning import ModelMaker
from shared.data import balance

def ray_hp_space(
        learning_rate_min: float,
        learning_rate_max: float, 
        batch_sizes:int,
        trial
    ):
    return {
        "learning_rate": tune.loguniform(learning_rate_min, 
                                         learning_rate_max),
        "per_device_train_batch_size": tune.choice(batch_sizes),
    }


def search(dataset, **kwargs):

    labels_per_category = kwargs["hyperparameters"]["data"]["labels_per_category"]

    search_training_dataset = balance(
        dataset["train"],
        labels_per_category=labels_per_category
    )
    dataset["train"] = search_training_dataset

    model_maker = ModelMaker(kwargs["model"]["checkpoint"])
    trainer = model_maker.get_trainer(dataset, trainer_args=kwargs["trainer"])
    hyperparameter_search = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        hp_space=lambda trial: ray_hp_space(
            kwargs["hyperparameters"]["learning_rates"][0],
            kwargs["hyperparameters"]["learning_rates"][1],
            kwargs["hyperparameters"]["batch_sizes"],
            trial),
        scheduler=AsyncHyperBandScheduler(
            metric="objective", 
            mode="max", 
            max_t=kwargs["asha"]["max_t"], 
            grace_period=kwargs["asha"]["grace_period"], 
            reduction_factor=kwargs["asha"]["reduction_factor"]
            ),
        resources_per_trial={
            "cpu": kwargs["asha"]["trial_cpus"], 
            "gpu": kwargs["asha"]["trial_gpus"]
            },
        n_trials=kwargs["asha"]["n_trials"],
        local_dir="./data/train.dir",
        name="tune_asha",
        log_to_file=True
    )
    return hyperparameter_search.hyperparameters


if __name__ == '__main__':
    Path('data/tmp.dir').mkdir(parents=True, exist_ok=True)
    Path('data/train.dir').mkdir(parents=True, exist_ok=True)

    params = dvc.api.params_show(stages=['search-hyperparameters'])

    prepared_dataset = datasets.DatasetDict.load_from_disk("data/prepare.dir/dataset")

    searched_hyperparameters = search(prepared_dataset, **params)

    write_json(
        "data/train.dir/hyperparameters.json", 
        searched_hyperparameters
    )