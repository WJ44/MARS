import json
import sys
from itertools import product

from mars.mars import MARS

LANGS = ["en", "ar"]

for lang1, lang2 in product(LANGS, repeat=2):
    ppi_config = {
        "evaluation_datasets": [
            f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.5_{lang1}_{lang2}.tsv",
            f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.525_{lang1}_{lang2}.tsv",
            f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.55_{lang1}_{lang2}.tsv",
            f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.575_{lang1}_{lang2}.tsv",
            f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.6_{lang1}_{lang2}.tsv",
            f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.625_{lang1}_{lang2}.tsv",
            f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.65_{lang1}_{lang2}.tsv",
            f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.675_{lang1}_{lang2}.tsv",
            f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.7_{lang1}_{lang2}.tsv",
        ],
        "checkpoints": [
            "checkpoints/microsoft-mdeberta-v3-base/Context_Relevance_Label_mlqa_dev_ratio_0.5_2024-09-30_15:27:39.pt",
            "checkpoints/microsoft-mdeberta-v3-base/Answer_Relevance_Label_mlqa_dev_ratio_0.5_2024-10-01_07:58:16.pt",
            "checkpoints/microsoft-mdeberta-v3-base/Answer_Faithfulness_Label_mlqa_dev_ratio_0.5_2024-10-02_05:48:52.pt",
            "checkpoints/microsoft-mdeberta-v3-base/Language_Consistency_Label_mlqa_dev_ratio_0.5_2024-10-02_13:18:21.pt",
        ],
        "rag_type": "question_answering",
        "labels": [
            "Context_Relevance_Label",
            "Answer_Relevance_Label",
            "Answer_Faithfulness_Label",
            "Language_Consistency_Label",
        ],
        "gold_label_paths": [f"multilingual_data/mlqa_({LANGS[1]})_dev_ratio_0.5_{lang1}_{lang2}.tsv"],
        "model_choice": "microsoft/mdeberta-v3-base",
        "assigned_batch_size": 8,
        "prediction_filepaths": [
            f"mlqa_({LANGS[1]})_test_ratio_0.5_{lang1}_{lang2}_output.tsv",
            f"mlqa_({LANGS[1]})_test_ratio_0.525_{lang1}_{lang2}_output.tsv",
            f"mlqa_({LANGS[1]})_test_ratio_0.55_{lang1}_{lang2}_output.tsv",
            f"mlqa_({LANGS[1]})_test_ratio_0.575_{lang1}_{lang2}_output.tsv",
            f"mlqa_({LANGS[1]})_test_ratio_0.6_{lang1}_{lang2}_output.tsv",
            f"mlqa_({LANGS[1]})_test_ratio_0.625_{lang1}_{lang2}_output.tsv",
            f"mlqa_({LANGS[1]})_test_ratio_0.65_{lang1}_{lang2}_output.tsv",
            f"mlqa_({LANGS[1]})_test_ratio_0.675_{lang1}_{lang2}_output.tsv",
            f"mlqa_({LANGS[1]})_test_ratio_0.7_{lang1}_{lang2}_output.tsv",
        ],
        "azure_openai_config": {},
    }

    mars = MARS(ppi=ppi_config)
    results = mars.evaluate_RAG()
    print(results)
    json.dump(results, open(f"results/results_MLQA_({LANGS[1]})_{lang1}-{lang2}.json", "w"))

ppi_config = {
    "evaluation_datasets": [
        f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.5_all.tsv",
        f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.525_all.tsv",
        f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.55_all.tsv",
        f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.575_all.tsv",
        f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.6_all.tsv",
        f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.625_all.tsv",
        f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.65_all.tsv",
        f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.675_all.tsv",
        f"multilingual_data/mlqa_({LANGS[1]})_test_ratio_0.7_all.tsv",
    ],
    "checkpoints": [
        "checkpoints/microsoft-mdeberta-v3-base/Context_Relevance_Label_mlqa_dev_ratio_0.5_2024-09-30_15:27:39.pt",
        "checkpoints/microsoft-mdeberta-v3-base/Answer_Relevance_Label_mlqa_dev_ratio_0.5_2024-10-01_07:58:16.pt",
        "checkpoints/microsoft-mdeberta-v3-base/Answer_Faithfulness_Label_mlqa_dev_ratio_0.5_2024-10-02_05:48:52.pt",
        "checkpoints/microsoft-mdeberta-v3-base/Language_Consistency_Label_mlqa_dev_ratio_0.5_2024-10-02_13:18:21.pt",
    ],
    "rag_type": "question_answering",
    "labels": [
        "Context_Relevance_Label",
        "Answer_Relevance_Label",
        "Answer_Faithfulness_Label",
        "Language_Consistency_Label",
    ],
    "gold_label_paths": [f"multilingual_data/mlqa_({LANGS[1]})_dev_ratio_0.5_all.tsv"],
    "model_choice": "microsoft/mdeberta-v3-base",
    "assigned_batch_size": 8,
    "prediction_filepaths": [
        f"mlqa_({LANGS[1]})_test_ratio_0.5_all_output.tsv",
        f"mlqa_({LANGS[1]})_test_ratio_0.525_all_output.tsv",
        f"mlqa_({LANGS[1]})_test_ratio_0.55_all_output.tsv",
        f"mlqa_({LANGS[1]})_test_ratio_0.575_all_output.tsv",
        f"mlqa_({LANGS[1]})_test_ratio_0.6_all_output.tsv",
        f"mlqa_({LANGS[1]})_test_ratio_0.625_all_output.tsv",
        f"mlqa_({LANGS[1]})_test_ratio_0.65_all_output.tsv",
        f"mlqa_({LANGS[1]})_test_ratio_0.675_all_output.tsv",
        f"mlqa_({LANGS[1]})_test_ratio_0.7_all_output.tsv",
    ],
    "azure_openai_config": {},
}

mars = MARS(ppi=ppi_config)
results = mars.evaluate_RAG()
print(results)
json.dump(results, open(f"results/results_MLQA_({LANGS[1]})_all.json", "w"))
