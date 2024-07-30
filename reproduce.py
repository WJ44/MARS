from ares import ARES

classifier_config = {
    "training_dataset": ["nq_synthetic_queries.tsv"],
    "validation_set": ["datasets/example_files/nq_labeled_output.tsv"],
    "label_column": ["Context_Relevance_Label", "Answer_Relevance_Label"],
    "num_epochs": 10,
    "patience_value": 3,
    "learning_rate": 5e-6,
    "assigned_batch_size": 1,
    "gradient_accumulation_multiplier": 32,
    "model_choice": "microsoft/deberta-v3-xsmall",
}

ares = ARES(classifier_model=classifier_config)
results = ares.train_classifier()
print(results)