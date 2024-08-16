from ares import ARES

synth_config = {
    "document_filepaths": ["multilingual_data/mlqa_test_en_en.tsv"],
    "few_shot_prompt_filename": "multilingual_data/few_shot_en_en.tsv",
    "synthetic_queries_filenames": ["multilingual_data/synthetic_queries_mlqa_en-en.tsv"],
    "documents_sampled": 10, # 4513
    "model_choice": "CohereForAI/aya-23-35B",
    "document_language": "English",
    "query_language": "English"
}

ares = ARES(synthetic_query_generator=synth_config)
results = ares.generate_synthetic_data()
print(results)

synth_config = {
    "document_filepaths": ["multilingual_data/mlqa_test_en_de.tsv"],
    "few_shot_prompt_filename": "multilingual_data/few_shot_en_de.tsv",
    "synthetic_queries_filenames": ["multilingual_data/synthetic_queries_mlqa_en-de.tsv"],
    "documents_sampled": 10,
    "model_choice": "CohereForAI/aya-23-35B",
    "document_language": "English",
    "query_language": "German"
}

ares = ARES(synthetic_query_generator=synth_config)
results = ares.generate_synthetic_data()
print(results)

synth_config = {
    "document_filepaths": ["multilingual_data/mlqa_test_de_en.tsv"],
    "few_shot_prompt_filename": "multilingual_data/few_shot_de_en.tsv",
    "documents_sampled": 10,
    "synthetic_queries_filenames": ["multilingual_data/synthetic_queries_mlqa_de-en.tsv"],
    "model_choice": "CohereForAI/aya-23-35B",
    "document_language": "German",
    "query_language": "English"
}

ares = ARES(synthetic_query_generator=synth_config)
results = ares.generate_synthetic_data()
print(results)

synth_config = {
    "document_filepaths": ["multilingual_data/mlqa_test_de_de.tsv"],
    "few_shot_prompt_filename": "multilingual_data/few_shot_de_de.tsv",
    "documents_sampled": 10,
    "synthetic_queries_filenames": ["multilingual_data/synthetic_queries_mlqa_de-de.tsv"],
    "model_choice": "CohereForAI/aya-23-35B",
    "document_language": "German",
    "query_language": "German"
}

ares = ARES(synthetic_query_generator=synth_config)
results = ares.generate_synthetic_data()
print(results)