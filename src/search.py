from pathlib import Path
import shutil
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
import datasets
from pathlib import Path
import dvc.api
from shared.helpers import write_json
from shared.learning import ModelMaker
from shared.data import balance_dataset

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
    labels_per_category = kwargs["hyperparameters"]["labels_per_category"]

    search_training_dataset, labels_per_category = balance_dataset(
        dataset["train"],
        labels_per_category=labels_per_category,
        seed=kwargs["data"]["seed"]
    )
    dataset["train"] = search_training_dataset


    temp_dir = 'data/tmp.dir'
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    model_maker = ModelMaker(checkpoints=kwargs["model"]["checkpoint"])
    trainer = model_maker.get_trainer(
        dataset, 
        output_dir=temp_dir,
        save_best_model=False,
        trainer_args=kwargs["trainer"]
    )

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
        local_dir="./data/search.dir",
        name="tune_asha",
        log_to_file=True
    )
    shutil.rmtree(temp_dir, ignore_errors=False, onerror=None)
    return hyperparameter_search.hyperparameters


if __name__ == '__main__':
    Path('data/search.dir').mkdir(parents=True, exist_ok=True)
    params = dvc.api.params_show(stages=['search'])
    
    prepared_dataset = datasets.DatasetDict.load_from_disk("data/prepare.dir/dataset")
    searched_hyperparameters = search(
        prepared_dataset, 
        **params
    )

    write_json(
        "data/search.dir/hyperparameters.json", 
        searched_hyperparameters
    )
