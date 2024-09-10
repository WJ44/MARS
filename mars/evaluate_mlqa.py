from ares import ARES

ppi_config = {
    "evaluation_datasets": ["multilingual_data/mlqa_test_ratio_0.7.tsv"],
    "checkpoints": ["checkpoints/microsoft-mdeberta-v3-base/Context_Relevance_Label_mlqa_dev_ratio_0.7_2024-09-09_094737.pt"],
    "rag_type": "question_answering",
    "labels": ["Context_Relevance_Label"],
    "gold_label_paths": ["multilingual_data/mlqa_test_ratio_0.7.tsv"],
    "model_choice": "microsoft/mdeberta-v3-base",
}

ares_module = ARES(ppi=ppi_config)
results = ares_module.evaluate_RAG()
print(results)