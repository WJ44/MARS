import pandas as pd
import json
from ares import ARES

xorqa = pd.read_json("baselines/mia_2022_dev_xorqa.jsonl", lines=True, dtype=False)

xorqa_output = json.load(open("baselines/submission.json"))["xor"]
xorqa_output = pd.DataFrame([xorqa_output.keys(), xorqa_output.values()]).T
xorqa_output.columns = ["id", "answer"]

xorqa = xorqa.merge(xorqa_output, on="id")

retrieval = json.load(open("baselines/mia_shared_xorqa_development_dpr_retrieval_results.json"))
retrieval = [{"id": x["q_id"], "context": x["ctxs"][0]["text"]} for x in retrieval]
retrieval = pd.DataFrame(retrieval)

xorqa = xorqa.merge(retrieval, on="id")

cora_xorqa = pd.DataFrame()
cora_xorqa["Query"] = xorqa["question"]
cora_xorqa["Answer"] = xorqa["answer"]
cora_xorqa["Document"] = xorqa["context"]
cora_xorqa["qa_lang"] = xorqa["lang"]

cora_xorqa.to_csv("baselines/cora_xorqa.tsv", sep="\t", index=False)

for lang in cora_xorqa["qa_lang"].unique():
    cora_xorqa[cora_xorqa["qa_lang"] == lang].to_csv(f"baselines/cora_xorqa_{lang}.tsv", sep="\t", index=False)


ppi_config = {
    "evaluation_datasets": ["baselines/cora_xorqa_ja.tsv"],
    "checkpoints": ["checkpoints/microsoft-mdeberta-v3-base/Context_Relevance_Label_mlqa_dev_ratio_0.7_2024-09-09_09:47:37.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Relevance_Label_mlqa_dev_ratio_0.7_2024-09-09_14:52:34.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Relevance_Label_mlqa_dev_ratio_0.7_2024-09-09_14:52:34.pt", "checkpoints/microsoft-mdeberta-v3-base/Language_Consistency_Label_mlqaqa_dev_ratio_0.7_2024-09-10_10:31:55.pt"],
    "rag_type": "question_answering",
    "labels": ["Context_Relevance_Label", "Answer_Relevance_Label", "Answer_Faithfulness_Label", "Language_Consistency_Label"],
    "gold_label_paths": ["multilingual_data/xor_attri_qa_dev.tsv"],
    "model_choice": "microsoft/mdeberta-v3-base",
    "assigned_batch_size": 1,
    "prediction_filepaths": ["baselines/cora_xorqa_ja_preds.tsv"],
}

ares_module = ARES(ppi=ppi_config)
results = ares_module.evaluate_RAG()
print(results)
json.dump(results, open("baselines/results_cora_xorqa.json", "w"))
