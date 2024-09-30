import random
import pandas as pd
import os

from tqdm import tqdm

# Requires the XOR-AttribQA dataset from: https://storage.googleapis.com/gresearch/xor_attriqa/xor_attriqa.zip (https://github.com/google-research/google-research/tree/master/xor_attriqa)

# TODO might be a way to get correct answers even if not ais

random.seed(42)

SPLIT = "dev" # Choose between "dev" and "test"

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
dataset_merged["Answer_Faithfulness"] = xor_attriqa_split["ais"].apply(lambda x: 1 if x == True else 0)
dataset_merged["id"] = dataset_merged["Document_en"].astype(str) + dataset_merged["Query_en"].astype(str)
dataset_merged["id"] = dataset_merged["id"].apply(hash)

# Sample few-shot examples
if SPLIT == "test":
    few_shot = dataset_merged[dataset_merged["Answer_Faithfulness"] == 1].sample(n=4, random_state=41)
    dataset_merged = dataset_merged.drop(few_shot.index)

# Function to create few-shot example files
def create_few_shot_files(few_shot):
    few_shot["Context_Relevance_Label"] = "[[Yes]]"
    few_shot["Answer_Faithfulness_Label"] = "[[Yes]]"
    few_shot["Answer_Relevance_Label"] = "[[Yes]]"
    few_shot["Language_Consistency_Label"] = "[[Yes]]"
    few_shot["Contradictory_Answer"] = "TODO"  # Added by hand

    few_shot_files = {
        f"attri_qa_{SPLIT}_few_shot_en_en.tsv": few_shot[["Document_en", "Answer_en", "Query_en", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]],
        f"attri_qa_{SPLIT}_few_shot_ja_ja.tsv": few_shot[["Document_ja", "Answer_ja", "Query_ja", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]],
        f"attri_qa_{SPLIT}_few_shot_en_ja.tsv": few_shot[["Document_en", "Answer_ja", "Query_ja", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]],
        f"attri_qa_{SPLIT}_few_shot_ja_en.tsv": few_shot[["Document_ja", "Answer_en", "Query_en", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]],
    }

    few_shot_en_ja_wrong = few_shot_files[f"attri_qa_{SPLIT}_few_shot_en_ja.tsv"].copy()
    few_shot_ja_en_wrong = few_shot_files[f"attri_qa_{SPLIT}_few_shot_ja_en.tsv"].copy()
    few_shot_en_ja_wrong["Answer_ja"] = few_shot_ja_en_wrong["Answer_en"]
    few_shot_ja_en_wrong["Answer_en"] = few_shot_en_ja_wrong["Answer_ja"]
    few_shot_en_ja_wrong["Language_Consistency_Label"] = "[[No]]"
    few_shot_ja_en_wrong["Language_Consistency_Label"] = "[[No]]"

    few_shot_files[f"attri_qa_{SPLIT}_few_shot_en_ja.tsv"] = pd.concat([few_shot_files[f"attri_qa_{SPLIT}_few_shot_en_ja.tsv"], few_shot_en_ja_wrong], axis=0, ignore_index=True)
    few_shot_files[f"attri_qa_{SPLIT}_few_shot_ja_en.tsv"] = pd.concat([few_shot_files[f"attri_qa_{SPLIT}_few_shot_ja_en.tsv"], few_shot_ja_en_wrong], axis=0, ignore_index=True)

    for filename, df in few_shot_files.items():
        df.columns = ["Document", "Answer", "Query", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]
        df.to_csv(f"multilingual_data/{filename}", sep="\t", index=False)

if SPLIT == "test" and not os.path.exists(f"multilingual_data/attri_qa_{SPLIT}_few_shot_en_en.tsv"):
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
    dataset_copy["Answer_Faithfulness"] = dataset["Answer_Faithfulness"]

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

if SPLIT == "dev":
    dataset = dataset.sample(n=len(dataset), random_state=42)
    dataset.to_csv(f"multilingual_data/attri_qa_{SPLIT}.tsv", sep="\t", index=False)

# Precompute possible incorrect passages and answers
incorrect_passages_dict = {}
incorrect_answers_dict = {}
for doc_lang in ["en", "ja"]:
    incorrect_passages_dict[doc_lang] = dataset_merged[f"Document_{doc_lang}"].unique()
    incorrect_answers_dict[doc_lang] = dataset_merged[f"Answer_{doc_lang}"].unique()

incorrect_passages = []
context_relevance_labels = []
incorrect_answers = []
answer_relevance_labels = []
incorrect_language = []
language_consistency_labels = []


# Get positive and negative Answer_Faithfulness samples
dataset_copy_1 = dataset.copy()
dataset = dataset[dataset["Answer_Faithfulness"] == 1]
dataset_copy_1 = dataset_copy_1[dataset_copy_1["Answer_Faithfulness"] == 0]
dataset_copy_1 = dataset_copy_1.sample(n=len(dataset_copy_1), random_state=42)

# Generate negative examples
for row in tqdm(range(len(dataset))):
    id = dataset.iloc[row]["id"]
    qa_lang = dataset.iloc[row]["qa_lang"]
    doc_lang = dataset.iloc[row]["doc_lang"]
    answer = dataset_merged[dataset_merged["id"] == id].iloc[0][f"Answer_{doc_lang}"]
    document = dataset_merged[dataset_merged["id"] == id].iloc[0][f"Document_{doc_lang}"]

    # Get all other passages
    wiki = [item for item in incorrect_passages_dict[doc_lang] if item != document]

    incorrect_passages.append(random.choice(wiki))
    context_relevance_labels.append(0)

    # Sample incorrect answer
    incorrect_answers.append(random.choice([ans for ans in incorrect_answers_dict[doc_lang] if ans != answer]))
    answer_relevance_labels.append(0)

    # Sample incorrect language answer
    incorrect_language.append(dataset_merged.loc[dataset_merged["id"] == id].iloc[0][f"Answer_{'ja' if qa_lang == 'en' else 'en'}"])
    language_consistency_labels.append(0)

dataset_copy_2 = dataset.copy()
dataset_copy_3 = dataset.copy()
dataset_copy_4 = dataset.copy()
dataset_copy_2 = dataset_copy_2.drop(columns=["Answer_Faithfulness"])
dataset_copy_3 = dataset_copy_3.drop(columns=["Answer_Faithfulness"])
dataset_copy_4 = dataset_copy_4.drop(columns=["Answer_Faithfulness"])

dataset_copy_2["Document"] = incorrect_passages
dataset_copy_2["Context_Relevance_Label"] = context_relevance_labels
dataset_copy_2 = dataset_copy_2.sample(n=len(dataset_copy_2), random_state=42)

dataset_copy_3["Answer"] = incorrect_answers
dataset_copy_3["Answer_Relevance_Label"] = answer_relevance_labels
dataset_copy_3 = dataset_copy_3.sample(n=len(dataset_copy_3), random_state=42)

dataset_copy_4["Answer"] = incorrect_language
dataset_copy_4["Language_Consistency_Label"] = language_consistency_labels
dataset_copy_4 = dataset_copy_4.sample(n=len(dataset_copy_4), random_state=42)

# Add labels for positive examples
dataset['Context_Relevance_Label'] = 1
dataset['Answer_Relevance_Label'] = 1
dataset['Language_Consistency_Label'] = 1

if SPLIT == "test":
    # Create datasets with different positive/negative ratios
    positive_negative_ratios = [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7]
    ids = pd.DataFrame(dataset["id"].unique())
    ids_copy_1 = pd.DataFrame(dataset_copy_1["id"].unique())
    ids_copy_2 = ids.copy()
    ids_copy_3 = ids.copy()
    ids_copy_4 = ids.copy()
    num_positives = len(ids) // len(positive_negative_ratios) ## TODO still do not really have enough samples for this even in test I think, better to only do for negatives, although they are also limited since only negatives wher faithfulness is positive are used
    for ratio in positive_negative_ratios:
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

        file_path = f"multilingual_data/mlqa_{SPLIT}_ratio_{ratio}.tsv"
        dataset_combined.to_csv(file_path, sep="\t", index=False)

        for lang1, lang2 in [("en", "en"), ("de", "de"), ("de", "en"), ("en", "de")]:
            file_path = f"multilingual_data/mlqa_{SPLIT}_ratio_{ratio}_{lang1}_{lang2}.tsv"
            dataset_filtered = dataset_combined[(dataset_combined["doc_lang"] == lang1) & (dataset_combined["qa_lang"] == lang2)]
            
            dataset_filtered.to_csv(file_path, sep="\t", index=False)