from ares import ARES

ppi_config = {
    "evaluation_datasets": ["multilingual_data/mlqa_dev_ratio_0.5.tsv"],
    "checkpoints": ["checkpoints/microsoft-mdeberta-v3-base/Context_Relevance_Label_mlqa_dev_ratio_0.7_2024-09-09_094737.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Relevance_Label_mlqa_dev_ratio_0.7_2024-09-09_145234.pt", "checkpoints/microsoft-mdeberta-v3-base/Language_Consistency_Label_mlqa_dev_ratio_0.7_2024-09-10_103155.pt"],
    "rag_type": "question_answering",
    "labels": ["Context_Relevance_Label", "Answer_Relevance_Label", "Language_Consistency_Label"],
    "gold_label_paths": ["multilingual_data/mlqa_dev_ratio_0.5.tsv"],
    "model_choice": "microsoft/mdeberta-v3-base",
    "assigned_batch_size": 8,
}

ares_module = ARES(ppi=ppi_config)
results = ares_module.evaluate_RAG()
print(results)