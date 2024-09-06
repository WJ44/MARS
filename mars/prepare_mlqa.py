import json
import os
import random
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

#Note: Then negative Answer_Faithfulness samples are not a good representation of hallucinated answers. They are just random answers from the dataset.

random.seed(42)

# Constants for file paths
EN_INDEX_PATH = "multilingual_data/mlqa_index_en_test.json"
DE_INDEX_PATH = "multilingual_data/mlqa_index_de_test.json"

# Load external information about MLQA dataset
with open(EN_INDEX_PATH, "r") as f:
    indexes_en = json.load(f)
with open(DE_INDEX_PATH, "r") as f:
    indexes_de = json.load(f)

indexes = {"en": indexes_en, "de": indexes_de}

def load_and_process_dataset(language_code):
    mlqa_total = load_dataset("facebook/mlqa", name=f"mlqa.{language_code}.{language_code}")
    mlqa_test = mlqa_total["test"].to_pandas()
    
    wikipedia_answers = [mlqa_test.iloc[row]["answers"]["text"][0] for row in tqdm(range(len(mlqa_test)))]
    
    dataset = pd.DataFrame()
    dataset[f"Document_{language_code}"] = mlqa_test["context"]
    dataset[f"Answer_{language_code}"] = wikipedia_answers
    dataset[f"Query_{language_code}"] = mlqa_test["question"]
    dataset["id"] = mlqa_test["id"]
    dataset[f"article_{language_code}"] = [indexes[language_code][id] for id in mlqa_test["id"]]
    
    return dataset

# Load and process datasets
dataset_en_en = load_and_process_dataset("en")
dataset_de_de = load_and_process_dataset("de")

# Combine English and German datasets
dataset_merged = pd.merge(dataset_en_en, dataset_de_de, on="id")

# Sample few-shot examples
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
        "mlqa_test_few_shot_en_en.tsv": few_shot[["Document_en", "Answer_en", "Query_en", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]],
        "mlqa_test_few_shot_de_de.tsv": few_shot[["Document_de", "Answer_de", "Query_de", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]],
        "mlqa_test_few_shot_en_de.tsv": few_shot[["Document_en", "Answer_de", "Query_de", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]],
        "mlqa_test_few_shot_de_en.tsv": few_shot[["Document_de", "Answer_en", "Query_en", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]],
    }

    few_shot_en_de_wrong = few_shot_files["mlqa_test_few_shot_en_de.tsv"].copy()
    few_shot_de_en_wrong = few_shot_files["mlqa_test_few_shot_de_en.tsv"].copy()
    few_shot_en_de_wrong["Answer_de"] = few_shot_de_en_wrong["Answer_en"]
    few_shot_de_en_wrong["Answer_en"] = few_shot_en_de_wrong["Answer_de"]
    few_shot_en_de_wrong["Language_Consistency_Label"] = "[[No]]"
    few_shot_de_en_wrong["Language_Consistency_Label"] = "[[No]]"

    few_shot_files["mlqa_test_few_shot_en_de.tsv"] = pd.concat([few_shot_files["mlqa_test_few_shot_en_de.tsv"], few_shot_en_de_wrong], axis=0, ignore_index=True)
    few_shot_files["mlqa_test_few_shot_de_en.tsv"] = pd.concat([few_shot_files["mlqa_test_few_shot_de_en.tsv"], few_shot_de_en_wrong], axis=0, ignore_index=True)

    for filename, df in few_shot_files.items():
        df.columns = ["Document", "Answer", "Query", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]
        df.to_csv(f"multilingual_data/{filename}", sep="\t", index=False)

if not os.path.exists("multilingual_data/mlqa_test_few_shot_en_en.tsv"):
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
create_dataset_file(dataset_merged, "en", "en", "multilingual_data/mlqa_test_en_en.tsv")
create_dataset_file(dataset_merged, "de", "de", "multilingual_data/mlqa_test_de_de.tsv")
create_dataset_file(dataset_merged, "de", "en", "multilingual_data/mlqa_test_de_en.tsv")
create_dataset_file(dataset_merged, "en", "de", "multilingual_data/mlqa_test_en_de.tsv")

# Combine all datasets
dataset = pd.concat([
    pd.read_csv("multilingual_data/mlqa_test_en_en.tsv", sep="\t"),
    pd.read_csv("multilingual_data/mlqa_test_de_de.tsv", sep="\t"),
    pd.read_csv("multilingual_data/mlqa_test_de_en.tsv", sep="\t"),
    pd.read_csv("multilingual_data/mlqa_test_en_de.tsv", sep="\t")
], axis=0, ignore_index=True)

# Precompute possible incorrect passages and answers
incorrect_passages_dict = {}
incorrect_answers_dict = {}
for doc_lang in ["en", "de"]:
    incorrect_passages_dict[doc_lang] = dataset_merged[[f"Document_{doc_lang}", f"article_{doc_lang}"]].drop_duplicates(subset=[f"Document_{doc_lang}"])
    incorrect_answers_dict[doc_lang] = dataset_merged[f"Answer_{doc_lang}"].unique()

incorrect_passages = []
context_relevance_labels = []
incorrect_answers = []
answer_faithfulness_labels = []
answer_relevance_labels = []
incorrect_language = []
language_consistency_labels = []

# Generate negative examples
for row in tqdm(range(len(dataset))):
    id = dataset.iloc[row]["id"]
    qa_lang = dataset.iloc[row]["qa_lang"]
    doc_lang = dataset.iloc[row]["doc_lang"]
    article = indexes[doc_lang][id]
    answer = dataset_merged[dataset_merged["id"] == id].iloc[0][f"Answer_{doc_lang}"]

    # Get all passages from other articles
    wiki = [item for item in incorrect_passages_dict[doc_lang][incorrect_passages_dict[doc_lang][f"article_{doc_lang}"] != article][f"Document_{doc_lang}"] if len(item.strip().split(" ")) >= 50]

    # Get all passages from the same article that do not contain the answer
    filtered_list = [item for item in dataset_merged[(dataset_merged[f"article_{doc_lang}"] == article) & (dataset_merged["id"] != id)][f"Document_{doc_lang}"].unique() if answer not in item and len(item.strip().split(" ")) >= 50]

    # For half the negative Context Relevance examples, use a passage from the same article that does not contain the answer, for the other half use a random passage
    if row % 2 == 0 and filtered_list:
        incorrect_passages.append(random.choice(filtered_list))
        context_relevance_labels.append(0)
    else:
        incorrect_passages.append(random.choice(wiki))
        context_relevance_labels.append(0)

    # Sample incorrect answer
    incorrect_answers.append(random.choice([ans for ans in incorrect_answers_dict[doc_lang] if ans != answer]))
    answer_faithfulness_labels.append(0)
    answer_relevance_labels.append(0)

    # Sample incorrect language answer
    incorrect_language.append(dataset_merged.loc[dataset_merged["id"] == id].iloc[0][f"Answer_{'de' if qa_lang == 'en' else 'en'}"])
    language_consistency_labels.append(0)

dataset_copy_1 = dataset.copy()
dataset_copy_2 = dataset.copy()
dataset_copy_3 = dataset.copy()

dataset_copy_1["Document"] = incorrect_passages
dataset_copy_1["Context_Relevance_Label"] = context_relevance_labels
dataset_copy_1 = dataset_copy_1.sample(n=len(dataset_copy_1), random_state=42)

dataset_copy_2["Answer"] = incorrect_answers
dataset_copy_2["Answer_Faithfulness_Label"] = answer_faithfulness_labels
dataset_copy_2["Answer_Relevance_Label"] = answer_relevance_labels
dataset_copy_2 = dataset_copy_2.sample(n=len(dataset_copy_2), random_state=42)

dataset_copy_3["Answer"] = incorrect_language
dataset_copy_3["Language_Consistency_Label"] = language_consistency_labels
dataset_copy_3 = dataset_copy_3.sample(n=len(dataset_copy_3), random_state=42)

# Add labels for positive examples
dataset['Context_Relevance_Label'] = 1
dataset['Answer_Faithfulness_Label'] = 1
dataset['Answer_Relevance_Label'] = 1
dataset['Language_Consistency_Label'] = 1

# Create datasets with different positive/negative ratios
positive_negative_ratios = [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7]
for ratio in positive_negative_ratios:
    negatives_to_add = int((1 - ratio) / ratio * len(dataset_copy_1))

    dataset_combined = pd.concat([dataset, dataset_copy_1[:negatives_to_add], dataset_copy_2[:negatives_to_add], dataset_copy_3[:negatives_to_add]], axis=0, ignore_index=True)
    dataset_combined = dataset_combined.sample(n=len(dataset_combined), random_state=42)

    file_path = f"multilingual_data/mlqa_test_ratio_{ratio}.tsv"
    dataset_combined.to_csv(file_path, sep="\t", index=False)