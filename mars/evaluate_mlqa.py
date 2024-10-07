from ares import ARES
import json

for lang1, lang2 in [("en", "en"), ("de", "de"), ("de", "en"), ("en", "de")]:
    ppi_config = {
        "evaluation_datasets": [f"multilingual_data/mlqa_test_ratio_0.5_{lang1}_{lang2}.tsv", f"multilingual_data/mlqa_test_ratio_0.525_{lang1}_{lang2}.tsv", f"multilingual_data/mlqa_test_ratio_0.55_{lang1}_{lang2}.tsv", f"multilingual_data/mlqa_test_ratio_0.575_{lang1}_{lang2}.tsv", f"multilingual_data/mlqa_test_ratio_0.6_{lang1}_{lang2}.tsv", f"multilingual_data/mlqa_test_ratio_0.625_{lang1}_{lang2}.tsv", f"multilingual_data/mlqa_test_ratio_0.65_{lang1}_{lang2}.tsv", f"multilingual_data/mlqa_test_ratio_0.675_{lang1}_{lang2}.tsv", f"multilingual_data/mlqa_test_ratio_0.7_{lang1}_{lang2}.tsv"],
        "checkpoints": ["checkpoints/microsoft-mdeberta-v3-base/Context_Relevance_Label_mlqa_dev_ratio_0.5_2024-09-30_15:27:39.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Relevance_Label_mlqa_dev_ratio_0.5_2024-10-01_07:58:16.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Faithfulness_Label_mlqa_dev_ratio_0.5_2024-10-02_05:48:52.pt", "checkpoints/microsoft-mdeberta-v3-base/Language_Consistency_Label_mlqa_dev_ratio_0.5_2024-10-02_13:18:21.pt"],
        "rag_type": "question_answering",
        "labels": ["Context_Relevance_Label", "Answer_Relevance_Label", "Answer_Faithfulness_Label", "Language_Consistency_Label"],
        "gold_label_paths": [f"multilingual_data/mlqa_dev_ratio_0.5_{lang1}_{lang2}.tsv"],
        "model_choice": "microsoft/mdeberta-v3-base",
        "assigned_batch_size": 32,
        "prediction_filepaths": [f"results_output_{lang1}_{lang2}_Context_Relevance.json", f"results_output_{lang1}_{lang2}_Answer_Relevance.json", f"results_output_{lang1}_{lang2}_Answer_Faithfulness.json", f"results_output_{lang1}_{lang2}_Language_Consistency.json"],
    }

    ares_module = ARES(ppi=ppi_config)
    results = ares_module.evaluate_RAG()
    print(results)
    json.dump(results, open(f"results_{lang1}_{lang2}.json", "w"))


ppi_config = {
    "evaluation_datasets": ["multilingual_data/mlqa_test_ratio_0.5.tsv", "multilingual_data/mlqa_test_ratio_0.525.tsv", "multilingual_data/mlqa_test_ratio_0.55.tsv", "multilingual_data/mlqa_test_ratio_0.575.tsv", "multilingual_data/mlqa_test_ratio_0.6.tsv", "multilingual_data/mlqa_test_ratio_0.625.tsv", "multilingual_data/mlqa_test_ratio_0.65.tsv", "multilingual_data/mlqa_test_ratio_0.675.tsv", "multilingual_data/mlqa_test_ratio_0.7.tsv"],
    "checkpoints": ["checkpoints/microsoft-mdeberta-v3-base/Context_Relevance_Label_mlqa_dev_ratio_0.5_2024-09-30_15:27:39.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Relevance_Label_mlqa_dev_ratio_0.5_2024-10-01_07:58:16.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Faithfulness_Label_mlqa_dev_ratio_0.5_2024-10-02_05:48:52.pt", "checkpoints/microsoft-mdeberta-v3-base/Language_Consistency_Label_mlqa_dev_ratio_0.5_2024-10-02_13:18:21.pt"],
    "rag_type": "question_answering",
    "labels": ["Context_Relevance_Label", "Answer_Relevance_Label", "Answer_Faithfulness_Label", "Language_Consistency_Label"],
    "gold_label_paths": ["multilingual_data/mlqa_dev_ratio_0.5.tsv"],
    "model_choice": "microsoft/mdeberta-v3-base",
    "assigned_batch_size": 32,
    "prediction_filepaths": ["results_output_Context_Relevance.json", "results_output_Answer_Relevance.json", "results_output_Answer_Faithfulness.json", "results_output_Language_Consistency.json"],
}

ares_module = ARES(ppi=ppi_config)
results = ares_module.evaluate_RAG()
print(results)
json.dump(results, open("results.json", "w"))
