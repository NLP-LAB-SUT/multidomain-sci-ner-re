import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
import torch
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)
from peft import LoraConfig, get_peft_model
import evaluate
from transformers import EarlyStoppingCallback

output_dir = os.environ.get("OUTPUT_DIR", "./results/llm_lora")
report_file_path = f"{output_dir}/evaluation_report.txt"
import os
os.makedirs(output_dir, exist_ok=True)

train_file = os.environ.get("TRAIN_FILE", "./data/ner/all_train.json")
test_file = os.environ.get("TEST_FILE", "./data/ner/all_test.json")
def debug_data_alignment(dataset, id2tag, tokenizer, num_samples=3):
    print("\n" + "="*20 + " DATA ALIGNMENT DEBUG (FIXED) " + "="*20)
    for i in range(num_samples):
        item = dataset[i]
        input_ids = item['input_ids']
        labels = item['labels']
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        print(f"\nSample {i+1}:")
        for tok, lbl in zip(tokens, labels):
            lbl_id = lbl.item() if torch.is_tensor(lbl) else lbl
            # بررسی توکن pad بر اساس pad_token_id توکنایزر
            if tok not in [tokenizer.pad_token, "[PAD]", "<pad>"]:
                if lbl_id == -100:
                    tag_name = "[IGNORE]"
                else:
                    tag_name = id2tag[lbl_id]
                
                print(f"{tok:<15} | {lbl_id:<5} | {tag_name}")
    print("\n" + "="*60 + "\n")
def convert_to_iob_format(sentence, entities_str):
    ENTITY_TAGS = ["Data", "Material", "Subject", "Parameter", "Criteria", "Theory",
                "Process", "Physical_Tools", "Non_Physical_Tools", "Method", "Policy"]
    tokens = re.findall(r'\w+|[^\w\s]', sentence)
    iob_tags = ['O'] * len(tokens)

    if not entities_str or entities_str == "None" or not isinstance(entities_str, str):
        return tokens, iob_tags

    entity_list = []
    raw_parts = entities_str.split(';')
    for part in raw_parts:
        part = part.strip()
        if ':' in part:
            last_colon_index = part.rfind(':')
            txt = part[:last_colon_index].strip().strip('"').strip("'")
            lbl = part[last_colon_index + 1:].strip()
            
            if txt and lbl in ENTITY_TAGS:
                entity_list.append({'text': txt, 'type': lbl})

    tokens_lower = [t.lower() for t in tokens]
    search_start = 0  

    for ent in entity_list:
        e_tokens = re.findall(r'\w+|[^\w\s]', ent['text'])
        if not e_tokens:
            continue

        e_tokens_lower = [t.lower() for t in e_tokens]
        e_len = len(e_tokens)

        for i in range(search_start, len(tokens) - e_len + 1):
            if tokens_lower[i:i + e_len] == e_tokens_lower:
                if all(iob_tags[k] == 'O' for k in range(i, i + e_len)):
                    iob_tags[i] = f"B-{ent['type']}"
                    for j in range(1, e_len):
                        iob_tags[i + j] = f"I-{ent['type']}"
                    
                    break
                    
    return tokens, iob_tags

def load_custom_dataset(file_path):
    print(f"Loading dataset from: {file_path}")
    sentences = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        for item in tqdm(data, desc="Parsing data"):
            sent = item.get('sentence', '')
            ents = item.get('entities', '')
            
            tokens, tags = convert_to_iob_format(sent, ents)
            
            if len(tokens) > 0:
                sentences.append(tokens)
                labels.append(tags)
                
    return sentences, labels


print("Loading Train file...")

full_train_sentences, full_train_labels = load_custom_dataset(train_file)

print("Loading Test file...")
test_sentences, test_labels = load_custom_dataset(test_file)


train_dataset = Dataset.from_dict({"tokens": full_train_sentences, "ner_tags": full_train_labels})
val_dataset = Dataset.from_dict({"tokens": test_sentences, "ner_tags": test_labels})
test_dataset = Dataset.from_dict({"tokens": test_sentences, "ner_tags": test_labels})

raw_dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

print(raw_dataset)
print(raw_dataset['train'][0:5])
print("========"*20)
all_labels_combined = full_train_labels + test_labels
flat_labels = []
for label_list in all_labels_combined:
    flat_labels.extend(label_list)
unique_labels = sorted(list(set(flat_labels)))    

additional_labels = set()
for label in unique_labels:
    if label.startswith("B-"):
        i_label = "I-" + label[2:]
        if i_label not in unique_labels:
            additional_labels.add(i_label)

unique_labels = sorted(list(set(unique_labels) | additional_labels))

label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for i, label in enumerate(unique_labels)}

print("unique_label:", unique_labels)
print("number of unique_label:", len(unique_labels))

print("label 2 id:", label2id)
print("*"*15)

model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")  # or mistralai/Mistral-7B-Instruct-v0.3
print("start load tokenizer")

tokenizer = AutoTokenizer.from_pretrained(model_name,add_prefix_space=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#test = tokenizer(raw_dataset["train"][0]["tokens"],truncation=True, is_split_into_words=True, max_length=512)
#print(test)
#print(test.word_ids)
#print(test.word_ids(batch_index=0))

def tokenize_and_align_labels(examples):
     tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=512)
     labels = []
     for i, label in enumerate(examples["ner_tags"]):
         word_ids = tokenized_inputs.word_ids(batch_index=i)
         previous_word_idx = None
         label_ids = []
         for word_idx in word_ids:
             if word_idx is None: label_ids.append(-100)
             elif word_idx != previous_word_idx: label_ids.append(label2id[label[word_idx]])
             else: label_ids.append(-100)
             previous_word_idx = word_idx
         labels.append(label_ids)
     tokenized_inputs["labels"] = labels
     return tokenized_inputs

def tokenize_and_align_labels2(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=512)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                original_tag = label[word_idx]
                if original_tag.startswith("B-"):
                    new_tag = "I-" + original_tag[2:]
                else:
                    new_tag = original_tag
                
                label_ids.append(label2id[new_tag])
            
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_dataset = raw_dataset.map(tokenize_and_align_labels, batched=True)
debug_data_alignment(tokenized_dataset["train"], id2label, tokenizer, num_samples=5)

model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels = len(unique_labels),
        id2label=id2label,
        label2id=label2id,
        device_map="auto",
        problem_type="single_label_classification",
        torch_dtype=torch.bfloat16
        )

lora_config = LoraConfig(
    r=4,                
    lora_alpha=16,          
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
    lora_dropout=0.1,
    bias="none",
    task_type="TOKEN_CLS"  )


peft_model = get_peft_model(model, lora_config)

peft_model.print_trainable_parameters()


data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

import numpy as np
from seqeval.metrics import classification_report
from sklearn.metrics import classification_report as sklearn_report
seqeval_metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
     
    flat_true_predictions = [label for sentence in true_predictions for label in sentence]
    flat_true_labels = [label for sentence in true_labels for label in sentence]
    print("\n" + "="*30)
    print("CLASSIFICATION REPORT(sqeeval):")
    print(classification_report(true_labels, true_predictions))
    print("="*30 + "\n")
    print("\n" + "="*30)
    print("CLASSIFICATION REPORT(sklearn):")
    print(sklearn_report(flat_true_labels, flat_true_predictions))
    print("="*30 + "\n")    


    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8, 
    num_train_epochs=100,             
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,   
    metric_for_best_model="f1",       
    greater_is_better=True,         
    bf16=True, 
    push_to_hub=False,
    logging_steps=10,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    #tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
)
import matplotlib.pyplot as plt
import pandas as pd

def plot_training_metrics(trainer, output_dir):
    logs = trainer.state.log_history
    
    train_logs = [log for log in logs if "loss" in log and "eval_loss" not in log]
    eval_logs = [log for log in logs if "eval_loss" in log]

    if train_logs:
        train_steps = [log["step"] for log in train_logs]
        train_loss = [log["loss"] for log in train_logs]

        plt.figure(figsize=(8,6))
        plt.plot(train_steps, train_loss)
        plt.xlabel("Step")
        plt.ylabel("Train Loss")
        plt.title("Training Loss")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/train_loss.png")
        plt.close()

    if eval_logs:
        eval_epochs = [log["epoch"] for log in eval_logs]
        eval_loss = [log["eval_loss"] for log in eval_logs]
        eval_f1 = [log.get("eval_f1", None) for log in eval_logs]

        plt.figure(figsize=(8,6))
        plt.plot(eval_epochs, eval_loss)
        plt.xlabel("Epoch")
        plt.ylabel("Eval Loss")
        plt.title("Validation Loss")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/eval_loss.png")
        plt.close()

        if any(eval_f1):
            plt.figure(figsize=(8,6))
            plt.plot(eval_epochs, eval_f1)
            plt.xlabel("Epoch")
            plt.ylabel("F1")
            plt.title("Validation F1")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/eval_f1.png")
            plt.close()
            
print("Starting training...")

trainer.train()
trainer.save_model(f"{output_dir}/best_adapter")
tokenizer.save_pretrained(f"{output_dir}/best_adapter")
import pickle
with open(f"{output_dir}/best_adapter/label2id.pkl", "wb") as f:
    pickle.dump(label2id, f)
with open(f"{output_dir}/best_adapter/id2label.pkl", "wb") as f:
    pickle.dump(id2label, f)

print(f"Best model saved to: {output_dir}/best_adapter")



print("\nRunning evaluation on TEST set...")
predictions, labels, _ = trainer.predict(tokenized_dataset["test"])
predictions = np.argmax(predictions, axis=2)

true_predictions = [
    [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

flat_preds = [item for sublist in true_predictions for item in sublist]
flat_labels = [item for sublist in true_labels for item in sublist]


flat_preds_no_o = []
flat_labels_no_o = []
for p, l in zip(flat_preds, flat_labels):
    if l != "O":
        flat_preds_no_o.append(p)
        flat_labels_no_o.append(l)

label_no_o = [label for label in unique_labels if label !='O']
def merge_bio(label):
    if label == "O": return "O"
    return label.split("-")[1] # حذف B- و I-

flat_preds_merged = [merge_bio(l) for l in flat_preds]
flat_labels_merged = [merge_bio(l) for l in flat_labels]

merged_labels = list(set(flat_labels_merged + flat_preds_merged))
merged_no_o = [label for label in merged_labels if label !="O"]
def plot_confusion_matrix(y_true, y_pred, title, filename, labels=None):
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png")
    plt.close()


with open(report_file_path, "w", encoding="utf-8") as f:
    f.write("Evaluation Report on SciERC Test Set\n")
    f.write("====================================\n\n")

    # Report 1: Seqeval (Standard NER metrics)
    f.write("1. Seqeval Classification Report (Strict Match):\n")
    f.write("-" * 50 + "\n")
    f.write(classification_report(true_labels, true_predictions))
    f.write("\n\n")

    # Report 2: Sklearn (With O)
    f.write("2. Sklearn Classification Report (Token-level, With 'O'):\n")
    f.write("-" * 50 + "\n")
    f.write(sklearn_report(flat_labels, flat_preds, digits=4))
    f.write("\n\n")

    # Report 3: Sklearn (Without O)
    f.write("3. Sklearn Classification Report (Token-level, Without 'O'):\n")
    f.write("-" * 50 + "\n")
    if flat_labels_no_o:
        f.write(sklearn_report(flat_labels, flat_preds,labels=label_no_o, digits=4))
    else:
        f.write("No non-O labels found in test set.\n")
    f.write("\n\n")

    # Report 4: Merged (Type-level, e.g., Method instead of B-Method)
    f.write("4. Merged Classification Report (Token-level, Ignored BIO prefix):\n")
    f.write("-" * 50 + "\n")
    f.write(sklearn_report(flat_labels_merged, flat_preds_merged, digits=4))
    f.write("\n\n")
    f.write("4. Merged Classification Report (Token-level, Ignored BIO prefix, NO O):\n")
    f.write("-" * 50 + "\n")
    f.write(sklearn_report(flat_labels_merged, flat_preds_merged, labels=merged_no_o,digits=4))
    f.write("\n\n")
print(f"Text reports saved to: {report_file_path}")


plot_confusion_matrix(flat_labels, flat_preds, 
                      "Confusion Matrix (With O)", "cm_with_o", labels=unique_labels)

# CM 2: Without O (فقط موجودیت‌ها را به عنوان لیبل می‌فرستیم تا O در نمودار نباشد)
unique_labels_no_o = [l for l in unique_labels if l != "O"]
plot_confusion_matrix(flat_labels, flat_preds, 
                     "Confusion Matrix (Without O)", "cm_no_o", labels=unique_labels_no_o)

# CM 3: Merged
unique_labels_merged = sorted(list(set([merge_bio(l) for l in unique_labels])))
plot_confusion_matrix(flat_labels_merged, flat_preds_merged, 
                     "Confusion Matrix (Merged Types)", "cm_merged", labels=unique_labels_merged)

print(f"Confusion matrices saved as images in: {output_dir}")

print("\nPlotting training curves...")
plot_training_metrics(trainer, output_dir)
