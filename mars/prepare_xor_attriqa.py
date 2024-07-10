import pandas as pd

#TODO sample few-shot examples
#TODO train and validation set
#TODO files with just English or German documents for synthetic query generation
#TODO human relevance judgement set

df = pd.read_json("multilingual_data/xor_attriqa/in-language/fi.jsonl", lines=True)
# df = df[df.query_language == "fi"]

dataset = pd.DataFrame()

dataset["Document"] = df["passage_en"]
dataset["Answer"] = df["prediction_translated_en"]
dataset["Query"] = df["query_translated_en"]
dataset["doc_lang"] = "en"
dataset["qa_lang"] = "en"
dataset["Answer_Faithfulness"] = df["ais"]


dataset2 = pd.DataFrame()

dataset2["Document"] = df["passage_in_language"]
dataset2["Answer"] = df["prediction"]
dataset2["Query"] = df["query"]
dataset2["doc_lang"] = "fi"
dataset2["qa_lang"] = "fi"
dataset2["Answer_Faithfulness"] = df["ais"]

dataset3 = pd.DataFrame()

dataset3["Document"] = df["passage_in_language"]
dataset3["Answer"] = df["prediction_translated_en"]
dataset3["Query"] = df["query_translated_en"]
dataset3["doc_lang"] = "fi"
dataset3["qa_lang"] = "en"
dataset3["Answer_Faithfulness"] = df["ais"]

dataset4 = pd.DataFrame()

dataset4["Document"] = df["passage_en"]
dataset4["Answer"] = df["prediction"]
dataset4["Query"] = df["query"]
dataset4["doc_lang"] = "en"
dataset4["qa_lang"] = "fi"
dataset4["Answer_Faithfulness"] = df["ais"]

dataset = pd.concat([dataset, dataset2, dataset3, dataset4], axis=0, ignore_index=True)

dataset.head()

file_path = "multilingual_data/xor_attriqa.tsv"

dataset.to_csv(file_path, sep="\t", index=False)