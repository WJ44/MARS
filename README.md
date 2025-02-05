# MARS #
MARS is an evaluation tool that lets you score your multilingual RAG system on four different metrics: context relevance, answer relevance, answer faithfulness and language consistency. MARS works by using your own knowledge base to make sure its scores reflect your specific use case and only needs little labelled examples, reducing the need for you to create extensive test datasets. While the initial setup requires a GPU with considerable VRAM, using the system in practice can be done easily on conventional GPUs with just a few GBs of VRAM. MARS works in three steps: synthetic data generation, LLM judge training and RAG system evaluation. In order to use MARS to evaluate RAG systems, you will have to perform these three steps.

## Data requirements ##
Before you start, you should ensure you have access to the right data to use MARS. MARS requires an in-domain corpus of passages, this would be the knowledge base you use for your RAG system. Secondly, MARS needs a few examples of questions that might be asked to your RAG system, as well as answers the RAG system would give. Additionally, it requires a labelled set of examples for each of the metrics from MARS you want to use. Ideally, this set would reflect the real-world distribution of questions asked to the RAG system and answers given by the RAG system, labelled as true or false for each metric. This set should contain at least around 150 examples, with a few hundred leading to better performance. You could construct this set by collecting data from your RAG system in practice and then human-labelling them. Lastly, to actually score your RAG system, MARS requires a set of (unlabelled) responses from your RAG system (including the question and context retrieved). Ideally, these are real-world questions, but benchmark questions can also be used.

## Synthetic data generation ##
When you are sure you have the data you need, you first need to generate a synthetic dataset based on your own corpus. This requires access to a machine with considerable VRAM, as it includes running an LLM locally; we suggest using a VM with an 80GB A100 GPU, as this was used when developing MARS, so it is guaranteed to work. MARS requires a dataset of a few thousand questions, which should take a few hours to generate. A code example for the synthetic generation is shown below:
```
from mars import MARS

synth_config = {
    "document_filepaths": ["multilingual_data/mlqa_(de)_test_en_de.tsv"],
    "few_shot_prompt_filename": "multilingual_data/mlqa_(de)_test_few_shot_en_de.tsv",
    "synthetic_queries_filenames": ["multilingual_data/synthetic_queries_mlqa_(de)_test_en-de.tsv"],
    "documents_sampled": 3000,
    "model_choice": "CohereForAI/aya-23-35B",
    "document_language": "English",
    "query_language": "German",
}

mars = MARS(synthetic_query_generator=synth_config)
mars.generate_synthetic_data()
```
Currently, synthetic datasets have to be generated for each language pair separately.

## LLM judge training ##
Now that you have your synthetic dataset, you can train your own LLM judges for each metric you want to use. Again, this requires a bit of compute and should take a few hours on an A100 GPU. A code example is shown below:
```
from mars import MARS

classifier_config = {
    "training_dataset": [
        "multilingual_data/synthetic_queries_mlqa_test_en-en.tsv",
        "multilingual_data/synthetic_queries_mlqa_test_en-de.tsv",
        "multilingual_data/synthetic_queries_mlqa_test_de-en.tsv",
        "multilingual_data/synthetic_queries_mlqa_test_de-de.tsv",
    ],
    "training_dataset_path": "multilingual_data/synthetic_queries_mlqa_test.tsv",
    "validation_set": ["multilingual_data/mlqa_(de)_dev_ratio_0.5_all.tsv"],
    "label_column": [
        "Context_Relevance_Label",
        "Answer_Relevance_Label",
        "Answer_Faithfulness_Label",
        "Language_Consistency_Label",
    ],
    "model_choice": "microsoft/mdeberta-v3-base",
    "num_epochs": 10,
    "patience_value": 3,
    "learning_rate": 5e-6,
    "assigned_batch_size": 32,
    "gradient_accumulation_multiplier": 32,
}

mars = MARS(classifier_model=classifier_config)
mars.train_classifier()
```

## RAG system evaluation ##
With your LLM judges ready to go, you can now score your RAG system. This requires the sets of labelled and unlabelled responses. A code example is shown below:
```
from mars import MARS

ppi_config = {
    "evaluation_datasets": [
        "multilingual_data/mlqa_(de)_test_ratio_0.7_all.tsv",
    ],
    "checkpoints": [
        "checkpoints/microsoft-mdeberta-v3-base/Context_Relevance_Label_mlqa_dev_ratio_0.5_2024-09-30.pt",
        "checkpoints/microsoft-mdeberta-v3-base/Answer_Relevance_Label_mlqa_dev_ratio_0.5_2024-10-01.pt",
        "checkpoints/microsoft-mdeberta-v3-base/Answer_Faithfulness_Label_mlqa_dev_ratio_0.5_2024-10-02.pt",
        "checkpoints/microsoft-mdeberta-v3-base/Language_Consistency_Label_mlqa_dev_ratio_0.5_2024-10-02.pt",
    ],
    "labels": [
        "Context_Relevance_Label",
        "Answer_Relevance_Label",
        "Answer_Faithfulness_Label",
        "Language_Consistency_Label",
    ],
    "gold_label_paths": ["multilingual_data/mlqa_(de)_dev_ratio_0.5_all.tsv"],
    "model_choice": "microsoft/mdeberta-v3-base",
    "assigned_batch_size": 8,
    "prediction_filepaths": [
        "mlqa_(de)_test_ratio_0.7_all_output.tsv",
    ],
}

mars = MARS(ppi=ppi_config)
results = mars.evaluate_RAG()
print(results)
```
MARS will now label each sample, both labelled and unlabelled, and combine this information to compute a score for your RAG system. While these scores can be used as is to get an idea about the performance of your RAG system, they are best used for comparison between different RAG systems.
