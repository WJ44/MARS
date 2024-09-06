from ares import ARES

classifier_config = {
    "training_dataset": ["multilingual_data/synthetic_queries_mlqa_en-en.tsv", "multilingual_data/synthetic_queries_mlqa_en-de.tsv", "multilingual_data/synthetic_queries_mlqa_de-en.tsv", "multilingual_data/synthetic_queries_mlqa_de-de.tsv"],
    "training_dataset_path": "multilingual_data/synthetic_queries_mlqa.tsv",
    "validation_set": ["multilingual_data/mlqa_test_ratio_0.7.tsv"],
    "label_column": ["Context_Relevance_Label", "Answer_Relevance_Label", "Language_Consitency_Label"],
    "model_choice": "microsoft/mdeberta-v3-base",
    "num_epochs": 10,
    "patience_value": 3,
    "learning_rate": 5e-6,
    "assigned_batch_size": 1,
    "gradient_accumulation_multiplier": 32,
}

ares = ARES(classifier_model=classifier_config)
results = ares.train_classifier()
print(results)