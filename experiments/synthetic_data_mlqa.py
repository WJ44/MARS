from mars import MARS

synth_config = {
    "document_filepaths": ["multilingual_data/mlqa_(de)_test_en_en.tsv"],
    "few_shot_prompt_filenames": ["multilingual_data/mlqa_(de)_test_few_shot_de_de.tsv"],
    "synthetic_queries_filenames": ["multilingual_data/synthetic_queries_mlqa_(de)_test_en-en.tsv"],
    "documents_sampled": 2,  # 4513
    "model_choice": "google/flan-t5-small",
    "document_language": "English",
    "query_language": "English",
    "second_language": "German",
}

mars = MARS(synthetic_query_generator=synth_config)
mars.generate_synthetic_data()

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

synth_config = {
    "document_filepaths": ["multilingual_data/mlqa_(de)_test_de_en.tsv"],
    "few_shot_prompt_filename": "multilingual_data/mlqa_(de)_test_few_shot_de_en.tsv",
    "documents_sampled": 3000,
    "synthetic_queries_filenames": ["multilingual_data/synthetic_queries_mlqa_(de)_test_de-en.tsv"],
    "model_choice": "CohereForAI/aya-23-35B",
    "document_language": "German",
    "query_language": "English",
}

mars = MARS(synthetic_query_generator=synth_config)
mars.generate_synthetic_data()

synth_config = {
    "document_filepaths": ["multilingual_data/mlqa_(de)_test_de_de.tsv"],
    "few_shot_prompt_filename": "multilingual_data/mlqa_(de)_test_few_shot_de_de.tsv",
    "documents_sampled": 3000,
    "synthetic_queries_filenames": ["multilingual_data/synthetic_queries_mlqa_(de)_test_de-de.tsv"],
    "model_choice": "CohereForAI/aya-23-35B",
    "document_language": "German",
    "query_language": "German",
    "second_language": "English",
}

mars = MARS(synthetic_query_generator=synth_config)
mars.generate_synthetic_data()
