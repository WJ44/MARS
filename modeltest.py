from ares import ARES

synth_config = {
    "document_filepaths": ["datasets/example_files/nq_labeled_output.tsv"],
    "few_shot_prompt_filename": "datasets/example_files/nq_few_shot_prompt_for_synthetic_query_generation.tsv",
    "synthetic_queries_filenames": ["nq_0.6_synthetic_queries.tsv"],
    "documents_sampled": 10,
    "model_choice": "CohereForAI/aya-23-8B"
}

ares = ARES(synthetic_query_generator=synth_config)
results = ares.generate_synthetic_data()
print(results)