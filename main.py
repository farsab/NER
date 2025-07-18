# Requirements:
# transformers datasets seqeval pandas


import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import classification_report


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def get_dataset():
    dataset = load_dataset("conll2003")
    return dataset


def train_ner_model(model_name, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label_list = dataset["train"].features["ner_tags"].feature.names
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))
    args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        weight_decay=0.01,
        report_to="none",
    )
    collator = DataCollatorForTokenClassification(tokenizer)

    metric = load_metric("seqeval")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        true_labels, true_preds = [], []
        for pred, lab in zip(predictions, labels):
            true_label, true_pred = [], []
            for p, l in zip(pred, lab):
                if l != -100:
                    true_label.append(label_list[l])
                    true_pred.append(label_list[p])
            true_labels.append(true_label)
            true_preds.append(true_pred)
        results = metric.compute(predictions=true_preds, references=true_labels)
        return {"f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"].select(range(2000)),
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer.evaluate()


def main():
    dataset = get_dataset()
    results = []

    for model in ["distilbert-base-uncased", "roberta-base"]:
        print(f"\nTraining with {model} tokenizer...")
        metrics = train_ner_model(model, dataset)
        results.append({"Tokenizer": model, **metrics})

    df = pd.DataFrame(results)
    print("\n=== Tokenization Comparison on NER ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
