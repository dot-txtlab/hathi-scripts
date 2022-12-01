import os
import re
import sys
import nltk
import argparse
import subprocess
import pandas as pd
from numpy.random import choice
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_args():
    parser = argparse.ArgumentParser(description="Given a directory of randomly sampled pages, fine-tune a HuggingFace classifier.")
    parser.add_argument("-t", "--tokenizer",
                        help="classifier tokenizer")
    parser.add_argument("-m", "--model",
                        help="classifier model")
    parser.add_argument("-d", "--directory",
                        help="output directory")
    parser.add_argument("-c", "--csv",
                        help="CSV to hold fine-tune data")
    parser.add_argument("-f", "--finetune",
                        help="directory to fine-tune on")
    parser.add_argument("-o", "--output",
                        help="model out")
    parser.add_argument("-a", "--action",
                        help="action to take, i.e. files, dataset, ft")

args = get_args()

def find_dir(path):
    for root in os.walk(path):
        if len(root[1]) == 0 and len(root[2]) > 5:
            return os.path.abspath(root[0])
        else:
            continue


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


def get_file(htid, target_dir):
    htid_path = re.sub(r'\W+', "_", htid)
    dir = args.directory + target_dir + "/" + htid_path
    subprocess.run("htrc download -o \"" +
                    dir + "\" " + htid, shell=True)
    path = find_dir(dir + "/" + htid_path)
    if path:
        files = os.listdir(path)
        keep_files = choice(files, 10)
        for filename in files:
            if filename not in keep_files:
                os.remove(path + "/" + filename)
        print("Obtained and trimmed " + htid)
    else:
        print("Failure! Skipping.")


# 1. collect files
if sys.argv[1] in ["files", "dataset", "ft"]:
    print("Collecting files")
    sources = ["FIC", "NON"]
    for source in sources:
        with open(args.directory + source + ".txt", "r") as f:
            for htid in f.readlines():
                get_file(htid, source)
    print("Finished.")

# 2. make dataset
if sys.argv[1] in ["dataset", "ft"]:
    print("Preparing dataset")
    dataset = pd.DataFrame()
    dataset["text"], dataset["label"] = [], []
    print("--> Finding files...")
    dirs = ["FIC", "NON"]
    labels = [0, 1]
    print("--> Looping over directories...")
    for y, directory in enumerate(dirs):
        print("--> Processing directory " + directory)
        for volume in os.listdir(args.directory + directory):
            samples = []
            volume_path = find_dir(volume)
            for filename in os.listdir(volume_path):
                with open(volume_path + "/" + filename, "r") as f:
                    page = " ".join(f.readlines())
                    samples.append(page)
            z = pd.DataFrame()
            z["text"] = samples
            z["label"] = [labels[y] for x in range(len(samples))]
            dataset = pd.concat([dataset, z])
    print("--> Exporting CSV")
    dataset.to_csv(args.csv, index=None)
    print("Finished.")

# 3. fine-tune model
if sys.argv[1] in ["ft"]:
    print("Fine-tuning model")
    print("--> Loading dataset")
    dataset = load_dataset('csv',
                           data_files=args.csv)
    dataset = dataset["train"]
    dataset = dataset.train_test_split(test_size=0.1)
    dataset = dataset.shuffle()

    print("--> Loading model")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    m = AutoModelForSequenceClassification.from_pretrained(args.model,
                                                           num_labels=2)

    print("--> Setting training args")
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        logging_steps=2000,
        gradient_checkpointing=None,
        fp16=True,
        num_train_epochs=5,
        save_steps=10000,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=m,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("--> Training")
    trainer.train()
    print("Finished.")

    # 4. export model
    print("Exporting model")
    trainer.save_model(f"/media/secure_volume/{model.out}")
    tokenizer.save_pretrained(f"/media/secure_volume/{model.out}-tokenizer")
print("Finished!")
