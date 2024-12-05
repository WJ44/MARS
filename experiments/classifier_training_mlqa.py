from ares import ARES

classifier_config = {
    "training_dataset": ["multilingual_data/synthetic_queries_mlqa_test_en-en.tsv", "multilingual_data/synthetic_queries_mlqa_test_en-de.tsv", "multilingual_data/synthetic_queries_mlqa_test_de-en.tsv", "multilingual_data/synthetic_queries_mlqa_test_de-de.tsv"],
    "training_dataset_path": "multilingual_data/synthetic_queries_mlqa_test.tsv",
    "validation_set": ["multilingual_data/mlqa_dev_ratio_0.5.tsv"],
    "label_column": ["Context_Relevance_Label", "Answer_Relevance_Label", "Answer_Faithfulness_Label", "Language_Consistency_Label"],
    "model_choice": "microsoft/mdeberta-v3-base",
    "num_epochs": 10,
    "patience_value": 3,
    "learning_rate": 5e-6,
    "assigned_batch_size": 32,
    "gradient_accumulation_multiplier": 32,
}

ares = ARES(classifier_model=classifier_config)
results = ares.train_classifier()
print(results)