from ares import ARES
import json

ppi_config = {
    "evaluation_datasets": ["datasets/eval_datasets/nq/nq_ratio_0.5.tsv", "datasets/eval_datasets/nq/nq_ratio_0.525.tsv", "datasets/eval_datasets/nq/nq_ratio_0.55.tsv", "datasets/eval_datasets/nq/nq_ratio_0.575.tsv", "datasets/eval_datasets/nq/nq_ratio_0.6.tsv", "datasets/eval_datasets/nq/nq_ratio_0.625.tsv", "datasets/eval_datasets/nq/nq_ratio_0.65.tsv", "datasets/eval_datasets/nq/nq_ratio_0.675.tsv", "datasets/eval_datasets/nq/nq_ratio_0.7.tsv"],
    "checkpoints": ["checkpoints/microsoft-mdeberta-v3-base/Context_Relevance_Label_mlqa_dev_ratio_0.7_2024-09-09_09:47:37.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Relevance_Label_mlqa_dev_ratio_0.7_2024-09-09_14:52:34.pt", "checkpoints/microsoft-mdeberta-v3-base/Language_Consistency_Label_mlqa_dev_ratio_0.7_2024-09-10_10:31:55.pt"],
    "rag_type": "question_answering",
    "labels": ["Context_Relevance_Label", "Answer_Relevance_Label", "Language_Consistency_Label"],
    "gold_label_paths": ["datasets/example_files/nq_labeled_output.tsv"],
    "model_choice": "microsoft/mdeberta-v3-base",
    "assigned_batch_size": 32,
}

ares_module = ARES(ppi=ppi_config)
results = ares_module.evaluate_RAG()
print(results)
json.dump(results, open("results_nq.json", "w"))
