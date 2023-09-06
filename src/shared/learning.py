import numpy as np
import evaluate
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments
)
from datasets import DatasetDict

def preprocess(image):
    # convert image to tensor
    tensor = transforms.ToTensor()(image)
    tensor = tensor - np.mean(tensor.numpy()) + 128
    tensor[tensor < 0] = 0
    tensor[tensor > 255] = 255
    tensor = tensor / 255
    # repeat the tensor two ore times to get 3 channels and use the repeat function
    tensor = tensor.repeat(3, 1, 1)
    return tensor

class Preprocess:
    def __init__(self):
        self.transforms = transforms.Compose(
            [
                transforms.Lambda(preprocess),
                transforms.CenterCrop(224),
            ]
        )

    def __call__(self, example_batch):
        example_batch["pixel_values"] = [
            self.transforms(img) for img in example_batch["image"]
        ]
        del example_batch["image"]
        return example_batch
    
class ModelMaker:
    def __init__(
            self, 
            checkpoints: str,
            ) -> None:
        self.checkpoint=checkpoints

    @staticmethod
    def __get_labels(dataset):
        labels = dataset["train"].features["label"].names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label
        return labels,label2id,id2label

    def __get_model(self, dataset):
        labels, label2id, id2label = self.__get_labels(dataset) 
        model = AutoModelForImageClassification.from_pretrained(
                self.checkpoint,
                num_labels=len(labels),
                id2label=id2label,
                label2id=label2id
                )
        model.to('cuda')
        return model
    
    @staticmethod
    def __compute_metrics(eval_pred):
        accuracy = evaluate.load("accuracy")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        metrics = accuracy.compute(predictions=predictions, references=labels)
        return metrics
    
    def predict(self, trainer, dataset):
        dataset = dataset.with_transform(Preprocess())
        return trainer.predict(dataset)

    def __get_trainer_args(self,
            trainer_args: dict,
            save_best_model: bool,
            output_dir: str
            ):
        if save_best_model == False:
            save_strategy = "no"
            load_best_model_at_end = False
        else:
            save_strategy = "epoch"
            load_best_model_at_end = True
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy=save_strategy,
            save_total_limit=2,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model="accuracy",
            learning_rate=trainer_args["learning_rate"],
            per_device_train_batch_size=trainer_args["per_device_train_batch_size"],
            gradient_accumulation_steps=trainer_args["gradient_accumulation_steps"],
            per_device_eval_batch_size=trainer_args["per_device_eval_batch_size"],
            num_train_epochs=trainer_args["num_train_epochs"],
            warmup_ratio=trainer_args["warmup_ratio"],
            logging_steps=trainer_args["logging_steps"],
        )
        return training_args

    def get_trainer(
            self, 
            dataset: DatasetDict,
            output_dir: str,
            save_best_model: bool,
            trainer_args: dict
        ):
        dataset = dataset.with_transform(Preprocess())
        data_collator = DefaultDataCollator()
        image_processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        
        trainer = Trainer(
                model_init=lambda: self.__get_model(dataset),
                args=self.__get_trainer_args(
                    trainer_args, 
                    save_best_model=save_best_model, 
                    output_dir=output_dir
                    ),
                data_collator=data_collator,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validate"],
                tokenizer=image_processor,
                compute_metrics=self.__compute_metrics
                )
        return trainer
