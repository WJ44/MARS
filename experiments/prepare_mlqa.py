import json
import os
import random
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from itertools import product

#Note: Then negative Answer_Faithfulness samples are not a good representation of hallucinated answers. They are just random answers from the dataset.

random.seed(42)

SPLIT = "test" # Choose between "dev" and "test"

SIZE = 500

LANGS = ["en", "ar"]

# Constants for file paths                                                    
EN_INDEX_PATH = f"multilingual_data/mlqa_index_en_{SPLIT}.json"
DE_INDEX_PATH = f"multilingual_data/mlqa_index_de_{SPLIT}.json"
AR_INDEX_PATH = f"multilingual_data/mlqa_index_ar_{SPLIT}.json"

# Load external information about MLQA dataset
with open(EN_INDEX_PATH, "r") as f:
    indexes_en = json.load(f)
with open(DE_INDEX_PATH, "r") as f:
    indexes_de = json.load(f)
with open(AR_INDEX_PATH, "r") as f: 
    indexes_ar = json.load(f)

indexes = {"en": indexes_en, "de": indexes_de, "ar": indexes_ar}

def load_and_process_dataset(language_code):
    mlqa_total = load_dataset("facebook/mlqa", name=f"mlqa.{language_code}.{language_code}")
    mlqa_split = mlqa_total["validation" if SPLIT == "dev" else SPLIT].to_pandas()
    
    wikipedia_answers = [mlqa_split.iloc[row]["answers"]["text"][0] for row in tqdm(range(len(mlqa_split)))]
    
    dataset = pd.DataFrame()
    dataset[f"Document_{language_code}"] = mlqa_split["context"]
    dataset[f"Answer_{language_code}"] = wikipedia_answers
    dataset[f"Query_{language_code}"] = mlqa_split["question"]
    dataset["id"] = mlqa_split["id"]
    dataset[f"article_{language_code}"] = [indexes[language_code][id] for id in mlqa_split["id"]]
    
    return dataset

# Load and process datasets
dataset_en_en = load_and_process_dataset("en")
dataset_de_de = load_and_process_dataset("de")
dataset_ar_ar = load_and_process_dataset("ar")

# Combine English and German datasets
if "de" in LANGS:
    dataset_merged = pd.merge(dataset_en_en, dataset_de_de, on="id")
elif "ar" in LANGS:
    dataset_merged = pd.merge(dataset_en_en, dataset_ar_ar, on="id")

# Sample few-shot examples
if SPLIT == "test":
    few_shot = dataset_merged.sample(n=4, random_state=40)
    dataset_merged = dataset_merged.drop(few_shot.index)

# Function to create few-shot example files
def create_few_shot_files(few_shot):
    few_shot["Context_Relevance_Label"] = "[[Yes]]"
    few_shot["Answer_Faithfulness_Label"] = "[[Yes]]"
    few_shot["Answer_Relevance_Label"] = "[[Yes]]"
    few_shot["Language_Consistency_Label"] = "[[Yes]]"
    few_shot["Contradictory_Answer"] = "TODO"  # Added by hand

    few_shot_files = {
        f"mlqa_({LANGS[1]})_{SPLIT}_few_shot_en_en.tsv": few_shot[["Document_en", "Answer_en", "Query_en", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]],
        f"mlqa_({LANGS[1]})_{SPLIT}_few_shot_de_de.tsv": few_shot[["Document_de", "Answer_de", "Query_de", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]],
        f"mlqa_({LANGS[1]})_{SPLIT}_few_shot_en_de.tsv": few_shot[["Document_en", "Answer_de", "Query_de", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]],
        f"mlqa_({LANGS[1]})_{SPLIT}_few_shot_de_en.tsv": few_shot[["Document_de", "Answer_en", "Query_en", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]],
    }

    few_shot_en_en_wrong = few_shot_files[f"mlqa_{SPLIT}_few_shot_en_en.tsv"].copy()
    few_shot_de_de_wrong = few_shot_files[f"mlqa_{SPLIT}_few_shot_de_de.tsv"].copy()
    few_shot_en_de_wrong = few_shot_files[f"mlqa_{SPLIT}_few_shot_en_de.tsv"].copy()
    few_shot_de_en_wrong = few_shot_files[f"mlqa_{SPLIT}_few_shot_de_en.tsv"].copy()
    few_shot_en_en_wrong["Answer_en"] = few_shot_files[f"mlqa_{SPLIT}_few_shot_de_de.tsv"]["Answer_de"]
    few_shot_de_de_wrong["Answer_de"] = few_shot_files[f"mlqa_{SPLIT}_few_shot_en_en.tsv"]["Answer_en"]
    few_shot_en_de_wrong["Answer_de"] = few_shot_files[f"mlqa_{SPLIT}_few_shot_de_en.tsv"]["Answer_en"]
    few_shot_de_en_wrong["Answer_en"] = few_shot_files[f"mlqa_{SPLIT}_few_shot_en_de.tsv"]["Answer_de"]
    few_shot_en_en_wrong["Language_Consistency_Label"] = "[[No]]"
    few_shot_de_de_wrong["Language_Consistency_Label"] = "[[No]]"
    few_shot_en_de_wrong["Language_Consistency_Label"] = "[[No]]"
    few_shot_de_en_wrong["Language_Consistency_Label"] = "[[No]]"

    few_shot_files[f"mlqa_({LANGS[1]})_{SPLIT}_few_shot_en_en.tsv"] = pd.concat([few_shot_files[f"mlqa_{SPLIT}_few_shot_en_en.tsv"], few_shot_en_en_wrong], axis=0, ignore_index=True)
    few_shot_files[f"mlqa_({LANGS[1]})_{SPLIT}_few_shot_de_de.tsv"] = pd.concat([few_shot_files[f"mlqa_{SPLIT}_few_shot_de_de.tsv"], few_shot_de_de_wrong], axis=0, ignore_index=True)
    few_shot_files[f"mlqa_({LANGS[1]})_{SPLIT}_few_shot_en_de.tsv"] = pd.concat([few_shot_files[f"mlqa_{SPLIT}_few_shot_en_de.tsv"], few_shot_en_de_wrong], axis=0, ignore_index=True)
    few_shot_files[f"mlqa_({LANGS[1]})_{SPLIT}_few_shot_de_en.tsv"] = pd.concat([few_shot_files[f"mlqa_{SPLIT}_few_shot_de_en.tsv"], few_shot_de_en_wrong], axis=0, ignore_index=True)

    for filename, df in few_shot_files.items():
        df.columns = ["Document", "Answer", "Query", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]
        df.to_csv(f"multilingual_data/{filename}", sep="\t", index=False)

if SPLIT == "test" and LANGS == ["en", "de"] and not os.path.exists(f"multilingual_data/mlqa_({LANGS[1]})_{SPLIT}_few_shot_en_en.tsv"):
    create_few_shot_files(few_shot)

# Function to create dataset files
def create_dataset_file(dataset, doc_lang, qa_lang, filename):
    dataset_copy = pd.DataFrame()
    dataset_copy["Document"] = dataset[f"Document_{doc_lang}"]
    dataset_copy["Answer"] = dataset[f"Answer_{qa_lang}"]
    dataset_copy["Query"] = dataset[f"Query_{qa_lang}"]
    dataset_copy["id"] = dataset["id"]
    dataset_copy["doc_lang"] = doc_lang
    dataset_copy["qa_lang"] = qa_lang
    dataset_copy.to_csv(filename, sep="\t", index=False)

# Create monolingual and cross-lingual datasets
for lang1, lang2 in product(LANGS, repeat=2):
    create_dataset_file(dataset_merged, lang1, lang2, f"multilingual_data/mlqa_({LANGS[1]})_{SPLIT}_{lang1}_{lang2}.tsv")

# Combine all datasets
dataset = pd.concat([
    pd.read_csv(f"multilingual_data/mlqa_({LANGS[1]})_{SPLIT}_{lang1}_{lang2}.tsv", sep="\t") for lang1, lang2 in product(LANGS, repeat=2)
], axis=0, ignore_index=True)

# Precompute possible incorrect passages and answers
incorrect_passages_dict = {}
incorrect_answers_dict = {}
for lang in LANGS:
    incorrect_passages_dict[lang] = dataset_merged[[f"Document_{lang}", f"article_{lang}"]].drop_duplicates(subset=[f"Document_{lang}"])
    incorrect_answers_dict[lang] = dataset_merged[f"Answer_{lang}"].unique()

incorrect_passages = []
context_relevance_labels = []
incorrect_answers = []
answer_faithfulness_labels = []
answer_relevance_labels = []
incorrect_language = []
language_consistency_labels = []
unfaithful_answers = []
unfaithful_passages = []

# Generate negative examples
for row in tqdm(range(len(dataset))):
    id = dataset.iloc[row]["id"]
    qa_lang = dataset.iloc[row]["qa_lang"]
    doc_lang = dataset.iloc[row]["doc_lang"]
    article = indexes[doc_lang][id]
    answer = dataset_merged[dataset_merged["id"] == id].iloc[0][f"Answer_{doc_lang}"]
    document = dataset_merged[dataset_merged["id"] == id].iloc[0][f"Document_{doc_lang}"]

    # Get all passages from other articles
    wiki = [item for item in incorrect_passages_dict[doc_lang][incorrect_passages_dict[doc_lang][f"article_{doc_lang}"] != article][f"Document_{doc_lang}"] if len(item.strip().split(" ")) >= 50] #TODO does not work for Japanese

    # Get all passages from the same article that do not contain the answer
    filtered_list = [item for item in dataset_merged[(dataset_merged[f"article_{doc_lang}"] == article) & (dataset_merged["id"] != id)][f"Document_{doc_lang}"].unique() if answer not in item and len(item.strip().split(" ")) >= 50] #TODO does not work for Japanese

    # For half the negative Context Relevance examples, use a passage from the same article that does not contain the answer, for the other half use a random passage
    if row % 2 == 0 and filtered_list:
        incorrect_passage = random.choice(filtered_list)
        incorrect_passages.append(incorrect_passage)
        context_relevance_labels.append(0)
    else:
        incorrect_passage = random.choice(wiki)
        incorrect_passages.append(incorrect_passage)
        context_relevance_labels.append(0)
    if row % 4 == 0 or row % 4 == 1:
        unfaithful_passages.append(incorrect_passage)
        answer_faithfulness_labels.append(0)
    else:
        unfaithful_passages.append(document)

    # Sample incorrect answer
    if qa_lang == doc_lang:
        incorrect_answer = random.choice([ans for ans in incorrect_answers_dict[qa_lang] if ans != answer])
        incorrect_answers.append(incorrect_answer)
        answer_relevance_labels.append(0)

    if row % 4 == 2 or row % 4 == 3:
        unfaithful_answers.append(incorrect_answer)
        answer_faithfulness_labels.append(0)
    else:
        unfaithful_answers.append(answer)

    # Sample incorrect language answer
    incorrect_language.append(dataset_merged.loc[dataset_merged["id"] == id].iloc[0][f"Answer_{LANGS[0] if qa_lang == LANGS[1] else LANGS[1]}"])
    language_consistency_labels.append(0)

dataset_copy_1 = dataset.copy()
dataset_copy_2 = dataset.copy()
dataset_copy_3 = dataset.copy()
dataset_copy_4 = dataset.copy()

dataset_copy_1["Document"] = incorrect_passages
dataset_copy_1["Context_Relevance_Label"] = context_relevance_labels
dataset_copy_1 = dataset_copy_1.sample(n=len(dataset_copy_1), random_state=42)

dataset_copy_2["Answer"] = incorrect_answers +  incorrect_answers
dataset_copy_2["Answer_Relevance_Label"] = answer_relevance_labels + answer_relevance_labels
dataset_copy_2 = dataset_copy_2.sample(n=len(dataset_copy_2), random_state=42)

dataset_copy_3["Answer"] = incorrect_language
dataset_copy_3["Language_Consistency_Label"] = language_consistency_labels
dataset_copy_3 = dataset_copy_3.sample(n=len(dataset_copy_3), random_state=42)

dataset_copy_4["Answer"] = unfaithful_answers
dataset_copy_4["Document"] = unfaithful_passages
dataset_copy_4["Answer_Faithfulness_Label"] = answer_faithfulness_labels
dataset_copy_4 = dataset_copy_4.sample(n=len(dataset_copy_4), random_state=42)

# Add labels for positive examples
dataset['Context_Relevance_Label'] = 1
dataset['Answer_Faithfulness_Label'] = 1
dataset['Answer_Relevance_Label'] = 1
dataset['Language_Consistency_Label'] = 1

# Create datasets with different positive/negative ratios
if SPLIT == "test": 
    positive_negative_ratios = [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7]
elif SPLIT == "dev":
    positive_negative_ratios = [0.5]
ids = pd.DataFrame(dataset["id"].unique())
ids_copy_1 = ids.copy()
ids_copy_2 = ids.copy()
ids_copy_3 = ids.copy()
ids_copy_4 = ids.copy()

num_positives = len(ids) // len(positive_negative_ratios)
cap = num_positives > SIZE
for ratio in positive_negative_ratios:
    if SPLIT == "test" and cap:
        num_positives = int(ratio * SIZE)
    negatives_to_add = int((1 - ratio) / ratio * num_positives)
    
    positive_ids = ids.sample(n=num_positives, random_state=42)
    ids = ids.drop(positive_ids.index)
    negative_ids_1 = ids_copy_1.sample(n=negatives_to_add, random_state=42)
    ids_copy_1 = ids_copy_1.drop(negative_ids_1.index)
    negative_ids_2 = ids_copy_2.sample(n=negatives_to_add, random_state=42)
    ids_copy_2 = ids_copy_2.drop(negative_ids_2.index)
    negative_ids_3 = ids_copy_3.sample(n=negatives_to_add, random_state=42)
    ids_copy_3 = ids_copy_3.drop(negative_ids_3.index)
    negative_ids_4 = ids_copy_4.sample(n=negatives_to_add, random_state=42)
    ids_copy_4 = ids_copy_4.drop(negative_ids_4.index)

    split = dataset[dataset["id"].isin(positive_ids[0])]
    split_copy_1 = dataset_copy_1[dataset_copy_1["id"].isin(negative_ids_1[0])]
    split_copy_2 = dataset_copy_2[dataset_copy_2["id"].isin(negative_ids_2[0])]
    split_copy_3 = dataset_copy_3[dataset_copy_3["id"].isin(negative_ids_3[0])]
    split_copy_4 = dataset_copy_4[dataset_copy_4["id"].isin(negative_ids_4[0])]

    dataset_combined = pd.concat([split, split_copy_1, split_copy_2, split_copy_3, split_copy_4], axis=0, ignore_index=True)
    dataset_combined = dataset_combined.sample(n=len(dataset_combined), random_state=42)

    file_path = f"multilingual_data/mlqa_({LANGS[1]})_{SPLIT}_ratio_{ratio}_all.tsv"
    # For every id, only keep a single combination of qa_lang and doc_lang, randomly
    dataset_reduced = dataset_combined.drop_duplicates(subset=["id", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label"])
    dataset_reduced.to_csv(file_path, sep="\t", index=False)

    for lang1, lang2 in product(LANGS, repeat=2):
        file_path = f"multilingual_data/mlqa_({LANGS[1]})_{SPLIT}_ratio_{ratio}_{lang1}_{lang2}.tsv"
        dataset_filtered = dataset_combined[(dataset_combined["doc_lang"] == lang1) & (dataset_combined["qa_lang"] == lang2)]
        
        dataset_filtered.to_csv(file_path, sep="\t", index=False)