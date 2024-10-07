import random
import pandas as pd
import os

from tqdm import tqdm

# Requires the XOR-AttribQA dataset from: https://storage.googleapis.com/gresearch/xor_attriqa/xor_attriqa.zip (https://github.com/google-research/google-research/tree/master/xor_attriqa)

# TODO might be a way to get correct answers even if not ais

random.seed(42)

SPLIT = "test" # Choose between "dev" and "test"

if SPLIT == "test":
    DATASET_PATH = "multilingual_data/xor_attriqa/in-language/ja.jsonl"
elif SPLIT == "dev":
    DATASET_PATH = "multilingual_data/xor_attriqa/in-language/val.jsonl"

xor_attriqa_split = pd.read_json(DATASET_PATH, lines=True)

if SPLIT == "dev":
    xor_attriqa_split = xor_attriqa_split[xor_attriqa_split["query_language"] == "ja"]

# Named to be consistent with the mlqa code
dataset_merged = pd.DataFrame()
dataset_merged["Document_en"] = xor_attriqa_split["passage_en"]
dataset_merged["Answer_en"] = xor_attriqa_split["prediction_translated_en"]
dataset_merged["Query_en"] = xor_attriqa_split["query_translated_en"]
dataset_merged["Document_ja"] = xor_attriqa_split["passage_in_language"]
dataset_merged["Answer_ja"] = xor_attriqa_split["prediction"]
dataset_merged["Query_ja"] = xor_attriqa_split["query"]
dataset_merged["Answer_Faithfulness_Label"] = xor_attriqa_split["ais"].apply(lambda x: 1 if x == True else 0)
dataset_merged["id"] = dataset_merged["Document_en"].astype(str) + dataset_merged["Query_en"].astype(str)
dataset_merged["id"] = dataset_merged["id"].apply(hash)


# Function to create dataset files
def create_dataset_file(dataset, doc_lang, qa_lang, filename):
    dataset_copy = pd.DataFrame()
    dataset_copy["Document"] = dataset[f"Document_{doc_lang}"]
    dataset_copy["Answer"] = dataset[f"Answer_{qa_lang}"]
    dataset_copy["Query"] = dataset[f"Query_{qa_lang}"]
    dataset_copy["id"] = dataset["id"]
    dataset_copy["doc_lang"] = doc_lang
    dataset_copy["qa_lang"] = qa_lang
    dataset_copy["Answer_Faithfulness_Label"] = dataset["Answer_Faithfulness_Label"]

    if SPLIT == "dev":
        dataset_copy = dataset_copy.sample(n=len(dataset_copy), random_state=42)
        
    dataset_copy.to_csv(filename, sep="\t", index=False)

# Create monolingual and cross-lingual datasets
create_dataset_file(dataset_merged, "en", "en", f"multilingual_data/attri_qa_{SPLIT}_en_en.tsv")
create_dataset_file(dataset_merged, "ja", "ja", f"multilingual_data/attri_qa_{SPLIT}_ja_ja.tsv")
create_dataset_file(dataset_merged, "ja", "en", f"multilingual_data/attri_qa_{SPLIT}_ja_en.tsv")
create_dataset_file(dataset_merged, "en", "ja", f"multilingual_data/attri_qa_{SPLIT}_en_ja.tsv")

# Combine all datasets
dataset = pd.concat([
    pd.read_csv(f"multilingual_data/attri_qa_{SPLIT}_en_en.tsv", sep="\t"),
    pd.read_csv(f"multilingual_data/attri_qa_{SPLIT}_ja_ja.tsv", sep="\t"),
    pd.read_csv(f"multilingual_data/attri_qa_{SPLIT}_ja_en.tsv", sep="\t"),
    pd.read_csv(f"multilingual_data/attri_qa_{SPLIT}_en_ja.tsv", sep="\t")
], axis=0, ignore_index=True)


dataset = dataset.sample(n=len(dataset), random_state=42)
dataset.to_csv(f"multilingual_data/attri_qa_{SPLIT}.tsv", sep="\t", index=False)
