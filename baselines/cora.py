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
    "checkpoints": ["checkpoints/microsoft-mdeberta-v3-base/Context_Relevance_Label_mlqa_dev_ratio_0.5_2024-09-30_15:27:39.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Relevance_Label_mlqa_dev_ratio_0.5_2024-10-01_07:58:16.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Faithfulness_Label_mlqa_dev_ratio_0.5_2024-10-02_05:48:52.pt", "checkpoints/microsoft-mdeberta-v3-base/Language_Consistency_Label_mlqa_dev_ratio_0.5_2024-10-02_13:18:21.pt"],
    "rag_type": "question_answering",
    "labels": ["Context_Relevance_Label", "Answer_Relevance_Label", "Answer_Faithfulness_Label", "Language_Consistency_Label"],
    "gold_label_paths": ["multilingual_data/mlqa_dev_ratio_0.5.tsv"],
    "model_choice": "microsoft/mdeberta-v3-base",
    "assigned_batch_size": 32,
    "prediction_filepaths": ["baselines/cora_xorqa_ja_preds_Context_Relevance_Label.tsv", "baselines/cora_xorqa_ja_preds_Answer_Relevance_Label.tsv", "baselines/cora_xorqa_ja_preds_Answer_Faithfulness_Label.tsv", "baselines/cora_xorqa_ja_preds_Language_Consistency_Label.tsv"],
}

ares_module = ARES(ppi=ppi_config)
results = ares_module.evaluate_RAG()
print(results)
json.dump(results, open("baselines/results_cora_xorqa.json", "w"))
