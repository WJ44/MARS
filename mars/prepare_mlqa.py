import json
import random
import pandas as pd
from tqdm import tqdm

from datasets import load_dataset

#TODO sample few-shot examples
#TODO test set
#TODO files with just English or German documents for synthetic query generation
#TODO human relevance judgement set

indexes = {
    "en": json.load(open("multilingual_data/mlqa_index_en_dev.json", "r")),
    "de": json.load(open("multilingual_data/mlqa_index_de_dev.json", "r"))
}

mlqa_en_en_total = load_dataset("facebook/mlqa", name="mlqa.en.en")

mlqa_en_en = mlqa_en_en_total["validation"]
mlqa_en_en = mlqa_en_en.to_pandas()

wikipedia_passages_en = []
wikipedia_answers_en = []

for row in tqdm(range(len(mlqa_en_en))):
    wikipedia_passages_en.append(mlqa_en_en.iloc[row]["context"])

    wikipedia_answers_en.append(mlqa_en_en.iloc[row]["answers"]["text"][0])

dataset_en_en = pd.DataFrame()

dataset_en_en["Document_en"] = wikipedia_passages_en
dataset_en_en["Answer_en"] = wikipedia_answers_en
dataset_en_en["Query_en"] = mlqa_en_en["question"]
dataset_en_en["id"] = mlqa_en_en["id"]
dataset_en_en["article_en"] = [indexes["en"][id] for id in mlqa_en_en["id"]]


mlqa_de_de_total = load_dataset("facebook/mlqa", name="mlqa.de.de")

mlqa_de_de = mlqa_de_de_total["validation"]
mlqa_de_de = mlqa_de_de.to_pandas()

wikipedia_passages_de = []
wikipedia_answers_de = []

for row in tqdm(range(len(mlqa_de_de))):
    wikipedia_passages_de.append(mlqa_de_de.iloc[row]["context"])

    wikipedia_answers_de.append(mlqa_de_de.iloc[row]["answers"]["text"][0])

dataset_de_de = pd.DataFrame()

dataset_de_de["Document_de"] = wikipedia_passages_de
dataset_de_de["Answer_de"] = wikipedia_answers_de
dataset_de_de["Query_de"] = mlqa_de_de["question"]
dataset_de_de["id"] = mlqa_de_de["id"]
dataset_de_de["article_de"] = [indexes["de"][id] for id in mlqa_de_de["id"]]


dataset_merged = pd.merge(dataset_en_en, dataset_de_de, on="id")

# few_shot = dataset_merged.sample(n=4, random_state=40)
# dataset_merged = dataset_merged.drop(few_shot.index)

# few_shot["Context_Relevance_Label"] = "[[Yes]]"
# few_shot["Answer_Faithfulness_Label"] = "[[Yes]]"
# few_shot["Answer_Relevance_Label"] = "[[Yes]]"
# few_shot["Language_Consistency_Label"] = "[[Yes]]"
# few_shot["Contradictory_Answer"] = "TODO" # Added by hand

# few_shot_en_en = few_shot[["Document_en", "Answer_en", "Query_en", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]].copy()
# few_shot_de_de = few_shot[["Document_de", "Answer_de", "Query_de", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]].copy()
# few_shot_en_de = few_shot[["Document_en", "Answer_de", "Query_de", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]].copy()
# few_shot_de_en = few_shot[["Document_de", "Answer_en", "Query_en", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]].copy()

# few_shot_en_de_wrong = few_shot[["Document_en", "Answer_en", "Query_de", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]].copy()
# few_shot_de_en_wrong = few_shot[["Document_de", "Answer_de", "Query_en", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]].copy()
# few_shot_en_de_wrong["Language_Consistency_Label"] = "[[No]]"
# few_shot_de_en_wrong["Language_Consistency_Label"] = "[[No]]"

# few_shot_en_en.columns = ["Document", "Answer", "Query", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]
# few_shot_de_de.columns = ["Document", "Answer", "Query", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]
# few_shot_en_de.columns = ["Document", "Answer", "Query", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]
# few_shot_de_en.columns = ["Document", "Answer", "Query", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]
# few_shot_en_de_wrong.columns = ["Document", "Answer", "Query", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]
# few_shot_de_en_wrong.columns = ["Document", "Answer", "Query", "Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label", "Language_Consistency_Label", "Contradictory_Answer"]

# few_shot_en_de = pd.concat([few_shot_en_de, few_shot_en_de_wrong], axis=0, ignore_index=True)
# few_shot_de_en = pd.concat([few_shot_de_en, few_shot_de_en_wrong], axis=0, ignore_index=True)

# few_shot_en_en.to_csv("multilingual_data/few_shot_en_en.tsv", sep="\t", index=False)
# few_shot_de_de.to_csv("multilingual_data/few_shot_de_de.tsv", sep="\t", index=False)
# few_shot_en_de.to_csv("multilingual_data/few_shot_en_de.tsv", sep="\t", index=False)
# few_shot_de_en.to_csv("multilingual_data/few_shot_de_en.tsv", sep="\t", index=False)


dataset = pd.DataFrame()

dataset["Document"] = dataset_merged["Document_en"]
dataset["Answer"] = dataset_merged["Answer_en"]
dataset["Query"] = dataset_merged["Query_en"]
dataset["id"] = dataset_merged["id"]
dataset["doc_lang"] = "en"
dataset["qa_lang"] = "en"

# dataset.to_csv("multilingual_data/mlqa_test_en_en.tsv", sep="\t", index=False)

dataset2 = pd.DataFrame()

dataset2["Document"] = dataset_merged["Document_de"]
dataset2["Answer"] = dataset_merged["Answer_de"]
dataset2["Query"] = dataset_merged["Query_de"]
dataset2["id"] = dataset_merged["id"]
dataset2["doc_lang"] = "de"
dataset2["qa_lang"] = "de"

# dataset2.to_csv("multilingual_data/mlqa_test_de_de.tsv", sep="\t", index=False)

dataset3 = pd.DataFrame()

dataset3["Document"] = dataset_merged["Document_de"]
dataset3["Answer"] = dataset_merged["Answer_en"]
dataset3["Query"] = dataset_merged["Query_en"]
dataset3["id"] = dataset_merged["id"]
dataset3["doc_lang"] = "de"
dataset3["qa_lang"] = "en"

# dataset3.to_csv("multilingual_data/mlqa_test_de_en.tsv", sep="\t", index=False)

dataset4 = pd.DataFrame()

dataset4["Document"] = dataset_merged["Document_en"]
dataset4["Answer"] = dataset_merged["Answer_de"]
dataset4["Query"] = dataset_merged["Query_de"]
dataset4["id"] = dataset_merged["id"]
dataset4["doc_lang"] = "en"
dataset4["qa_lang"] = "de"

# dataset4.to_csv("multilingual_data/mlqa_test_en_de.tsv", sep="\t", index=False)

dataset = pd.concat([dataset, dataset2, dataset3, dataset4], axis=0, ignore_index=True)

incorrect_passages = []
context_relevance_labels = []

incorrect_answers = []
answer_faithfulness_labels = []
answer_relevance_labels = []

incorrect_language = []
language_consistency_labels = []


for row in tqdm(range(len(dataset))):
    id = dataset.iloc[row]["id"]
    qa_lang = dataset.iloc[row]["qa_lang"]
    doc_lang = dataset.iloc[row]["doc_lang"]
    article = indexes[doc_lang][id]
    answer = dataset_merged[dataset_merged["id"] == id].iloc[0][f"Answer_{doc_lang}"]
    wiki = dataset_merged[f"Document_{doc_lang}"].unique()

    filtered_list = dataset_merged[dataset_merged[f"article_{doc_lang}"] == article][f"Document_{doc_lang}"].unique()
    filtered_list = [item for item in filtered_list if answer not in item]
    filtered_list = [item for item in filtered_list if len(item.strip().split(" ")) >= 50]

    if row % 2 == 0 and len(filtered_list) > 0:
        incorrect_passages.append(random.choice(filtered_list))
        context_relevance_labels.append(0)
    else:
        random_int = random.randint(0, len(wiki) - 1)
        while (
            wiki[random_int] != dataset.iloc[row]["Document"]
            and wiki[random_int] not in filtered_list
            and len(wiki[random_int]) < 50
        ):
            random_int = random.randint(0, len(wiki) - 1)

        incorrect_passages.append(wiki[random_int])
        context_relevance_labels.append(0)

    random_int = random.randint(0, len(dataset) - 1)
    while random_int == row:
        random_int = random.randint(0, len(dataset) - 1)
    random_answer = dataset.iloc[random_int]["Answer"]
    incorrect_answers.append(random_answer)
    answer_faithfulness_labels.append(0)
    answer_relevance_labels.append(0)

    if qa_lang == "en":
        incorrect_language.append(dataset_merged.loc[dataset_merged["id"] == id].iloc[0]["Answer_de"])
        language_consistency_labels.append(0)
    elif qa_lang == "de":
        incorrect_language.append(dataset_merged.loc[dataset_merged["id"] == id].iloc[0]["Answer_en"])
        language_consistency_labels.append(0)


dataset_copy_1 = dataset.copy()
dataset_copy_2 = dataset.copy()
dataset_copy_3 = dataset.copy()

dataset_copy_1["Document"] = incorrect_passages
dataset_copy_1["Context_Relevance_Label"] = context_relevance_labels

dataset_copy_2["Answer"] = incorrect_answers
# dataset_copy_2["Answer_Faithfulness_Label"] = answer_faithfulness_labels
dataset_copy_2["Answer_Relevance_Label"] = answer_relevance_labels

dataset_copy_3["Answer"] = incorrect_language
dataset_copy_3["Language_Consistency_Label"] = language_consistency_labels

dataset['Context_Relevance_Label'] = [1 for _ in range(len(dataset))]
# dataset['Answer_Faithfulness_Label'] = [1 for _ in range(len(dataset))]
dataset['Answer_Relevance_Label'] = [1 for _ in range(len(dataset))]
dataset['Language_Consistency_Label'] = [1 for _ in range(len(dataset))]

positive_negative_ratios = [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7]

for ratio in positive_negative_ratios:
    negatives_to_add = (1 - ratio) / ratio
    negatives_to_add = int(negatives_to_add * len(dataset_copy_1))

    dataset_combined = pd.concat([dataset, dataset_copy_1[:negatives_to_add], dataset_copy_2[:negatives_to_add], dataset_copy_3[:negatives_to_add]], axis=0, ignore_index=True)
    dataset_combined = dataset_combined.sample(n=len(dataset_combined), random_state=42)

    file_path = "multilingual_data/mlqa_ratio" + str(ratio) + ".tsv"
    
    dataset_combined.to_csv(file_path, sep="\t", index=False)