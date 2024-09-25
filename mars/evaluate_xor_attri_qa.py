from ares import ARES
import json

for lang1, lang2 in [("en", "en"), ("ja", "ja"), ("ja", "en"), ("en", "ja")]:
    ppi_config = {
        "evaluation_datasets": [f"multilingual_data/xor_attri_qa_test_ratio_0.5_{lang1}_{lang2}.tsv", f"multilingual_data/xor_attri_qa_test_ratio_0.525_{lang1}_{lang2}.tsv", f"multilingual_data/xor_attri_qa_test_ratio_0.55_{lang1}_{lang2}.tsv", f"multilingual_data/xor_attri_qa_test_ratio_0.575_{lang1}_{lang2}.tsv", f"multilingual_data/xor_attri_qa_test_ratio_0.6_{lang1}_{lang2}.tsv", f"multilingual_data/xor_attri_qa_test_ratio_0.625_{lang1}_{lang2}.tsv", f"multilingual_data/xor_attri_qa_test_ratio_0.65_{lang1}_{lang2}.tsv", f"multilingual_data/xor_attri_qa_test_ratio_0.675_{lang1}_{lang2}.tsv", f"multilingual_data/xor_attri_qa_test_ratio_0.7_{lang1}_{lang2}.tsv"],
        "checkpoints": ["checkpoints/microsoft-mdeberta-v3-base/Context_Relevance_Label_mlqa_dev_ratio_0.7_2024-09-09_09:47:37.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Relevance_Label_mlqa_dev_ratio_0.7_2024-09-09_14:52:34.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Relevance_Label_mlqa_dev_ratio_0.7_2024-09-09_14:52:34.pt", "checkpoints/microsoft-mdeberta-v3-base/Language_Consistency_Label_mlqaqa_dev_ratio_0.7_2024-09-10_10:31:55.pt"],
        "rag_type": "question_answering",
        "labels": ["Context_Relevance_Label", "Answer_Relevance_Label", "Answer_Faithfulness_Label", "Language_Consistency_Label"],
        "gold_label_paths": [f"multilingual_data/xor_attri_qa_dev_{lang1}_{lang2}.tsv"],
        "model_choice": "microsoft/mdeberta-v3-base",
        "assigned_batch_size": 32,
    }

    ares_module = ARES(ppi=ppi_config)
    results = ares_module.evaluate_RAG()
    print(results)
    json.dump(results, open(f"results_xor_attri_qa_{lang1}_{lang2}.json", "w"))


ppi_config = {
    "evaluation_datasets": ["multilingual_data/xor_attri_qa_test_ratio_0.5.tsv", "multilingual_data/xor_attri_qa_test_ratio_0.525.tsv", "multilingual_data/xor_attri_qa_test_ratio_0.55.tsv", "multilingual_data/xor_attri_qa_test_ratio_0.575.tsv", "multilingual_data/xor_attri_qa_test_ratio_0.6.tsv", "multilingual_data/xor_attri_qa_test_ratio_0.625.tsv", "multilingual_data/xor_attri_qa_test_ratio_0.65.tsv", "multilingual_data/xor_attri_qa_test_ratio_0.675.tsv", "multilingual_data/xor_attri_qa_test_ratio_0.7.tsv"],
    "checkpoints": ["checkpoints/microsoft-mdeberta-v3-base/Context_Relevance_Label_mlqa_dev_ratio_0.7_2024-09-09_09:47:37.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Relevance_Label_mlqa_dev_ratio_0.7_2024-09-09_14:52:34.pt", "checkpoints/microsoft-mdeberta-v3-base/Answer_Relevance_Label_mlqa_dev_ratio_0.7_2024-09-09_14:52:34.pt", "checkpoints/microsoft-mdeberta-v3-base/Language_Consistency_Label_mlqaqa_dev_ratio_0.7_2024-09-10_10:31:55.pt"],
    "rag_type": "question_answering",
    "labels": ["Context_Relevance_Label", "Answer_Relevance_Label", "Answer_Faithfulness_Label", "Language_Consistency_Label"],
    "gold_label_paths": ["multilingual_data/xor_attri_qa_dev.tsv"],
    "model_choice": "microsoft/mdeberta-v3-base",
    "assigned_batch_size": 32,
}

ares_module = ARES(ppi=ppi_config)
results = ares_module.evaluate_RAG()
print(results)
json.dump(results, open("results_xor_attri_qa.json", "w"))
