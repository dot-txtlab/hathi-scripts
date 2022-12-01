import os
import re
import sys
import time
import torch
import argparse
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_args():
    parser = argparse.ArgumentParser(description="Given a HuggingFace classifier and a list of HTIDs, classify volumes with said classifier.")
    parser.add_argument("-t", "--tokenizer",
                        help="classifier tokenizer")
    parser.add_argument("-m", "--model",
                        help="classifier model")
    parser.add_argument("-d", "--directory",
                        help="output directory")
    parser.add_argument("-c", "--csv",
                        help="CSV containing volumes to be classified, denoted by HTID")

args = get_args()

tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=2)
dir = args.directory


def classify(input):
    inputs = tokenizer(input, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]


def find_dir(path):
    for root in os.walk(path):
        if len(root[1]) == 0 and len(root[2]) > 5:
            return os.path.abspath(root[0])
        else:
            continue


def process(htid):
    htid_path = re.sub(r'\W+', "_", sys.argv[1])
    try:
        try:
            path = os.path.abspath(dir + "/" + htid_path + "/" + htid)
            files = os.listdir(path)
        except FileNotFoundError:
            path = find_dir(dir + "/" + htid_path)
            files = os.listdir(path)
    except:
        print("HTID not found! Skipping volume.")
        return "N/A"
    labels = []
    for filename in files[:10]:
        with open(path + filename) as f:
            page = " ".join(f.readlines())
            labels.append(0 if classify(page) == "LABEL_0" else 1)
        return Counter(labels).most_common()[0][0]


def calc_time(t0, t1, y, x):
    iters_left = x - y
    iter_length = int(t1 - t0)
    return int(((iters_left * iter_length) / 60) / 60)


with pd.read_csv(args.csv) as df:
    x, fic = len(df), df.fic.to_list()
    for y, row in enumerate(df.iterrows()):
        if row[1]["fic"] not in [1, 0, "N/A"]:
            t0 = time.time()
            fic[y] = process(row[1]["htid"])
            df["fic"] = fic
            df.to_csv(".backup.csv")
            t1 = time.time()
            print(f"[{y}/{x}] -- {calc_time(t0, t1, y, x)}h left")
    df.to_csv("volumes.csv")
