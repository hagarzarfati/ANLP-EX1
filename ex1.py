import argparse
import wandb
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from evaluate import load

metric = load("accuracy")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on MRPC for paraphrase detection")
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    return parser.parse_args()

def get_num_set_samples(dataset, arg_max_samples):
    len_dataset = len(dataset)
    if arg_max_samples == -1:
        return len_dataset
    if arg_max_samples > len_dataset:
        return len_dataset
    if 0 < arg_max_samples <= len_dataset:
        return arg_max_samples

    return len_dataset


def preprocess(example, tokenizer):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


def compute_metrics(p):
    preds = p.predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = preds.argmax(axis=1)

    result = metric.compute(predictions=preds, references=p.label_ids)
    return result


def main():
    args = parse_args()
    if args.do_predict and args.model_path:
        model_name = args.model_path
        model_name = model_name.replace("./models/", "")
        model_name += "_predict"
    else:
        model_name = f"num_epochs={args.num_train_epochs}_lr={args.lr}_batch_size={args.batch_size}"

    # wandb
    wandb.login(key="c06cc92b78e0d706c1135936736b3d5e19bbb54f", relogin=True)
    wandb.init(
        project="ANLP_EX01",
        name=model_name,
        config=vars(args)
    )

    # Load dataset
    raw_datasets = load_dataset("nyu-mll/glue", "mrpc")

    # Load model and tokenizer
    if args.do_train:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    elif args.do_predict and args.model_path:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    else:
        return

    # Tokenize dataset
    tokenized_datasets = raw_datasets.map(lambda e: preprocess(e, tokenizer), batched=True)
    tokenized_datasets["train"] = tokenized_datasets["train"].select(range(get_num_set_samples(tokenized_datasets["train"], args.max_train_samples)))
    tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(get_num_set_samples(tokenized_datasets["validation"], args.max_eval_samples)))
    tokenized_datasets["test"] = tokenized_datasets["test"].select(range(get_num_set_samples(tokenized_datasets["test"], args.max_predict_samples)))

    # Train

    # Training config
    if args.do_train:
        training_args = TrainingArguments(
            output_dir="./results",
            logging_dir="./logs",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=16,
            learning_rate=args.lr,
            num_train_epochs=args.num_train_epochs,
            logging_steps=1,
            logging_strategy="steps",
            report_to="wandb",
            save_strategy="no"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            compute_metrics=compute_metrics,
        )

        trainer.train()
        model.save_pretrained(f"./models/{model_name}")
        val_metrics = trainer.evaluate()
        val_acc = val_metrics["eval_accuracy"]
        print("Validation accuracy:", val_acc)

        # Save results to res.txt
        with open("res.txt", "a") as f:
            f.write(f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {val_acc:.4f}\n")

    if args.do_predict:
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=16,
            do_train=False,
            do_eval=False,
            logging_dir="./logs",
            report_to="wandb",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        test_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
        print(f"Test accuracy: {test_metrics['eval_accuracy']:.4f} (Model: {model_name})")

        preds = trainer.predict(tokenized_datasets["test"])
        pred_labels = np.argmax(preds.predictions, axis=1)
        raw_test = raw_datasets["test"]

        with open(f"predictions.txt", "w") as f:
            for i, label in enumerate(pred_labels):
                s1 = raw_test[i]["sentence1"]
                s2 = raw_test[i]["sentence2"]
                f.write(f"{s1}###{s2}###{label}\n")

    wandb.finish()

def compare_models(best_model_path, worst_model_path):
    dataset = load_dataset("nyu-mll/glue", "mrpc")["validation"]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    worst_model = AutoModelForSequenceClassification.from_pretrained(worst_model_path)

    def preprocess(example, tokenizer):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_dataset = dataset.map(lambda e: preprocess(e, tokenizer), batched=True)

    best_model_trainer = Trainer(model=best_model, tokenizer=tokenizer)
    worst_model_trainer = Trainer(model=worst_model, tokenizer=tokenizer)

    preds_best_model = best_model_trainer.predict(tokenized_dataset).predictions.argmax(axis=1)
    preds_worst_model = worst_model_trainer.predict(tokenized_dataset).predictions.argmax(axis=1)
    labels = np.array(dataset["label"])

    total = len(labels)
    num_ones = np.sum(labels == 1)
    num_zeros = np.sum(labels == 0)

    print(f"Total validation samples: {total}")
    print(f"Label = 1 (Paraphrase): {num_ones}")
    print(f"Label = 0 (Not Paraphrase): {num_zeros}\n")

    count = 0
    best_model_predictions_label_count = [0, 0]
    worst_model_predictions_label_count = [0, 0]
    best_model_predictions_count = [0, 0] # [0] - succeeded ,[1] - failed
    wost_model_predictions_count = [0, 0]  # [0] - succeeded ,[1] - failed

    print(f"Best Model correct / Worst Model wrong:")
    for i, (pa, pb, label) in enumerate(zip(preds_best_model, preds_worst_model, labels)):
        best_model_predictions_label_count[pa] += 1
        worst_model_predictions_label_count[pb] += 1

        if pa==label: best_model_predictions_count[0] += 1
        else: best_model_predictions_count[1] += 1

        if pb==label: wost_model_predictions_count[0] += 1
        else: wost_model_predictions_count[1] += 1

        if pa == label and pb != label:
            count += 1
            s1 = dataset[i]["sentence1"]
            s2 = dataset[i]["sentence2"]
            print(f"{i}.  Label: {label} | Best Model: {pa} | Worst Model: {pb}")
            print(f"  Sentence1: {s1}")
            print(f"  Sentence2: {s2}\n")

    print(f"\nTotal examples where Best model was right and Worst model was wrong: {count}")
    print(f"\nBest model prediction label count: 0 - {best_model_predictions_label_count[0]}, 1 - {best_model_predictions_label_count[1]}")
    print(f"Worst model prediction label count: 0 - {worst_model_predictions_label_count[0]}, 1 - {worst_model_predictions_label_count[1]}")
    print(f"\nBest model prediction count: succeeded - {best_model_predictions_count[0]}, failed - {best_model_predictions_count[1]}")
    print(f"Worst model prediction count: succeeded - {wost_model_predictions_count[0]}, failed - {wost_model_predictions_count[1]}")

if __name__ == "__main__":
    main()
    # compare_models("./models/num_epochs=2_lr=0.0001_batch_size=64", "./models/num_epochs=4_lr=0.1_batch_size=32")
