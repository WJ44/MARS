from ares import ARES
import json
from itertools import product

LANGS = ["en", "ar"]

for lang1, lang2 in product(LANGS, repeat=2):
    ppi_config = {
        "evaluation_datasets": [f"multilingual_data/mlqa_test_ratio_0.5_{lang1}_{lang2}.tsv", f"multilingual_data/mlqa_test_ratio_0.525_{lang1}_{lang2}.tsv", f"multilingual_data/mlqa_test_ratio_0.55_{lang1}_{lang2}.tsv", f"multilingual_data/mlqa_test_ratio_0.575_{lang1}_{lang2}.tsv", f"multilingual_data/mlqa_test_ratio_0.6_{lang1}_{lang2}.tsv", f"multilingual_data/mlqa_test_ratio_0.625_{lang1}_{lang2}.tsv", f"multilingual_data/mlqa_test_ratio_0.65_{lang1}_{lang2}.tsv", f"multilingual_data/mlqa_test_ratio_0.675_{lang1}_{lang2}.tsv", f"multilingual_data/mlqa_test_ratio_0.7_{lang1}_{lang2}.tsv"],
        "checkpoints": ["checkpoints/microsoft-mdeberta-v3-base/Context_Relevance_Label_mlqa_dev_ratio_0.5_2024-09-30_15:27:39.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Relevance_Label_mlqa_dev_ratio_0.5_2024-10-01_07:58:16.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Faithfulness_Label_mlqa_dev_ratio_0.5_2024-10-02_05:48:52.pt", "checkpoints/microsoft-mdeberta-v3-base/Language_Consistency_Label_mlqa_dev_ratio_0.5_2024-10-02_13:18:21.pt"],
        "rag_type": "question_answering",
        "labels": ["Context_Relevance_Label", "Answer_Relevance_Label", "Answer_Faithfulness_Label", "Language_Consistency_Label"],
        "gold_label_paths": [f"multilingual_data/mlqa_dev_ratio_0.5_{lang1}_{lang2}.tsv"],
        "model_choice": "microsoft/mdeberta-v3-base",
        "assigned_batch_size": 8,
        "prediction_filepaths": [f"results_output_ratio_0.5_{lang1}_{lang2}.json", f"results_output_ratio_0.525_{lang1}_{lang2}.json", f"results_output_ratio_0.55_{lang1}_{lang2}.json", f"results_output_ratio_0.575_{lang1}_{lang2}.json", f"results_output_ratio_0.6_{lang1}_{lang2}.json", f"results_output_ratio_0.625_{lang1}_{lang2}.json", f"results_output_ratio_0.65_{lang1}_{lang2}.json", f"results_output_ratio_0.675_{lang1}_{lang2}.json", f"results_output_ratio_0.7_{lang1}_{lang2}.json"],
        "azure_openai_config": {},
    }

    ares_module = ARES(ppi=ppi_config)
    results = ares_module.evaluate_RAG()
    print(results)
    json.dump(results, open(f"results_{lang1}_{lang2}.json", "w"))


ppi_config = {
    "evaluation_datasets": [f"multilingual_data/mlqa_test_ratio_0.5_{LANGS[1]}.tsv", f"multilingual_data/mlqa_test_ratio_0.525_{LANGS[1]}.tsv", f"multilingual_data/mlqa_test_ratio_0.55_{LANGS[1]}.tsv", f"multilingual_data/mlqa_test_ratio_0.575_{LANGS[1]}.tsv", f"multilingual_data/mlqa_test_ratio_0.6_{LANGS[1]}.tsv", f"multilingual_data/mlqa_test_ratio_0.625_{LANGS[1]}.tsv", f"multilingual_data/mlqa_test_ratio_0.65_{LANGS[1]}.tsv", f"multilingual_data/mlqa_test_ratio_0.675_{LANGS[1]}.tsv", f"multilingual_data/mlqa_test_ratio_0.7_{LANGS[1]}.tsv"],
    "checkpoints": ["checkpoints/microsoft-mdeberta-v3-base/Context_Relevance_Label_mlqa_dev_ratio_0.5_2024-09-30_15:27:39.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Relevance_Label_mlqa_dev_ratio_0.5_2024-10-01_07:58:16.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Faithfulness_Label_mlqa_dev_ratio_0.5_2024-10-02_05:48:52.pt", "checkpoints/microsoft-mdeberta-v3-base/Language_Consistency_Label_mlqa_dev_ratio_0.5_2024-10-02_13:18:21.pt"],
    "rag_type": "question_answering",
    "labels": ["Context_Relevance_Label", "Answer_Relevance_Label", "Answer_Faithfulness_Label", "Language_Consistency_Label"],
    "gold_label_paths": [f"multilingual_data/mlqa_dev_ratio_0.5_{LANGS[1]}.tsv"],
    "model_choice": "microsoft/mdeberta-v3-base",
    "assigned_batch_size": 8,
    "prediction_filepaths": [f"results_output_ratio_0.5_{LANGS[1]}.json", f"results_output_ratio_0.525_{LANGS[1]}.json", f"results_output_ratio_0.55_{LANGS[1]}.json", f"results_output_ratio_0.575_{LANGS[1]}.json", f"results_output_ratio_0.6_{LANGS[1]}.json", f"results_output_ratio_0.625_{LANGS[1]}.json", f"results_output_ratio_0.65_{LANGS[1]}.json", f"results_output_ratio_0.675_{LANGS[1]}.json", f"results_output_ratio_0.7_{LANGS[1]}.json"],
    "azure_openai_config": {},
}

ares_module = ARES(ppi=ppi_config)
results = ares_module.evaluate_RAG()
print(results)
json.dump(results, open(f"results_{LANGS[1]}.json", "w"))
0