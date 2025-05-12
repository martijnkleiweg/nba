# This script fine-tunes a BERT model for Named Entity Recognition (NER) on a custom dataset.
# Student: Martinus Kleiweg
import json
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from seqeval.metrics import classification_report as seqeval_classification_report, f1_score
import os

# Load dataset from cleaned JSONL file
def load_data(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            record["ner_tags"] = record["label"]
            del record["label"]
            records.append(record)
    return Dataset.from_list(records)

# Label setup
label_list = ["O", "B-AWARD", "I-AWARD", "B-COLLEGE", "I-COLLEGE", "B-DOB", "I-DOB", "B-NATIONALITY", "I-NATIONALITY", "B-POSITION", "I-POSITION"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# Tokenize and align spans
def tokenize_and_align_labels(example):
    text = example["text"]
    spans = example["ner_tags"]
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_offsets_mapping=True)
    labels = ["O"] * len(tokenized["input_ids"])
    offset_mapping = tokenized["offset_mapping"]
    for i, (start, end) in enumerate(offset_mapping):
        for span in spans:
            s_char, e_char, lab = span["start"], span["end"], span["label"]
            if start < e_char and end > s_char:
                prefix = "B" if start == s_char else "I"
                labels[i] = f"{prefix}-{lab}"
    tokenized["labels"] = [label2id.get(l, label2id["O"]) for l in labels]
    return tokenized

# Metric calculation
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)
    true_labels = []
    true_predictions = []
    for pred, label in zip(predictions, labels):
        cur_preds, cur_labels = [], []
        for p_i, l_i in zip(pred, label):
            if l_i != -100:
                cur_preds.append(id2label[p_i])
                cur_labels.append(id2label[l_i])
        true_labels.append(cur_labels)
        true_predictions.append(cur_preds)
    report = seqeval_classification_report(true_labels, true_predictions, zero_division=0, output_dict=True)
    return {
        "f1": f1_score(true_labels, true_predictions),
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"]
    }

# Load tokenizer/model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(label_list), id2label=id2label, label2id=label2id)

# Process dataset
dataset = load_data("data_clean.jsonl")
tokenized_ds = dataset.map(tokenize_and_align_labels, batched=False)
split = tokenized_ds.train_test_split(test_size=0.1)
train_ds, eval_ds = split["train"], split["test"]

# Training arguments
args = TrainingArguments(
    output_dir="./ner_model",
    eval_strategy="epoch",
    save_strategy="epoch",                 # must match evaluation_strategy
    save_total_limit=1,                    # keep only 1 checkpoint
    logging_dir="./logs",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_strategy="epoch",
    load_best_model_at_end=True,          # needed for early stopping
    metric_for_best_model="f1",
    greater_is_better=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train and save final model
trainer.train()
trainer.save_model("./ner_model/final")
tokenizer.save_pretrained("./ner_model/final")
