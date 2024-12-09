import json
import sys
from itertools import product

from mars.mars import MARS

for lang in ["bn", "fi", "ja", "ru", "te"]:

    LANGS = ["en", lang]

    for lang1, lang2 in product(LANGS, repeat=2):
        ppi_config = {
            "evaluation_datasets": [f"multilingual_data/attri_qa_({LANGS[1]})_test_{lang1}_{lang2}.tsv"],
            "checkpoints": [
                "checkpoints/microsoft-mdeberta-v3-base/Answer_Faithfulness_Label_mlqa_dev_ratio_0.5_2024-10-02_05:48:52.pt"
            ],
            "rag_type": "question_answering",
            "labels": ["Answer_Faithfulness_Label"],
            "gold_label_paths": [f"multilingual_data/attri_qa_({LANGS[1]})_dev_{lang1}_{lang2}.tsv"],
            "model_choice": "microsoft/mdeberta-v3-base",
            "assigned_batch_size": 8,
            "prediction_file_paths": [f"attri_qa_({LANGS[1]})_test_{lang1}_{lang2}_output.tsv"],
            "azure_openai_config": {},
        }

        mars = MARS(ppi=ppi_config)
        results = mars.evaluate_RAG()
        print(results)
        json.dump(results, open(f"results/results_attri_qa_({LANGS[1]})_{lang1}-{lang2}.json", "w"))

    ppi_config = {
        "evaluation_datasets": [f"multilingual_data/attri_qa_({LANGS[1]})_test_all.tsv"],
        "checkpoints": [
            "checkpoints/microsoft-mdeberta-v3-base/Answer_Faithfulness_Label_mlqa_dev_ratio_0.5_2024-10-02_05:48:52.pt"
        ],
        "rag_type": "question_answering",
        "labels": ["Answer_Faithfulness_Label"],
        "gold_label_paths": [f"multilingual_data/attri_qa_({LANGS[1]})_dev_all.tsv"],
        "model_choice": "microsoft/mdeberta-v3-base",
        "assigned_batch_size": 8,
        "prediction_file_paths": [f"attri_qa_({LANGS[1]})_test_all_output.tsv"],
        "azure_openai_config": {},
    }

    mars = MARS(ppi=ppi_config)
    results = mars.evaluate_RAG()
    print(results)
    json.dump(results, open(f"results/results_attri_qa_({LANGS[1]})_all.json", "w"))
