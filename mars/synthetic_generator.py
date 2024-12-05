from LLM_as_a_Judge_Adaptation.Generate_Synthetic_Queries_and_Answers import (
    load_model,
    load_documents,
    load_few_shot_prompt,
    generate_few_shot_prompts,
    generate_wrong_language_few_shot_prompts,
    generate_synthetic_data,
)


def synthetic_generator_config(
    document_filepaths: list,
    few_shot_prompt_filenames: list,
    synthetic_queries_filenames: list,
    documents_sampled: int,
    model_choice: str = "CohereForAI/aya-23-35B",
    percentiles: list = [0.05, 0.25, 0.5, 0.95],
    document_language: str = "English",
    query_language: str = "English",
    second_language: str = None,
) -> None:
    """
    Configures and generates synthetic queries and answers based on the provided parameters.

    Args:
        document_filepaths (list): List of file paths to the documents.
        few_shot_prompt_filenames (list): List of filenames for the few-shot prompts.
        synthetic_queries_filenames (list): List of filenames for the synthetic queries.
        documents_sampled (int): Number of documents to sample.
        model_choice (str, optional): Model choice for the generation. Defaults to "google/flan-t5-xxl".
        percentiles (list, optional): List of percentiles for the generation. Defaults to [0.05, 0.25, 0.5, 0.95].
        document_language (str, optional): Language of the documents. Defaults to "English".
        query_language (str, optional): Language of the queries. Defaults to "English".
        second_language (str, optional): Second language for the generation. Defaults to None.
    Raises:
        ValueError: If the lengths of document_filepaths, few_shot_prompt_filenames,
            and synthetic_queries_filenames do not match.
    """

    print("=" * 40)
    print("Saving synthetic queries to: ", synthetic_queries_filenames)
    print("=" * 40)

    model, tokenizer, device = load_model(model_choice)

    if not len(document_filepaths) == len(few_shot_prompt_filenames) == len(synthetic_queries_filenames):
        raise ValueError(
            "document_filepaths, few_shot_prompt_filenames, and synthetic_queries_filenames lists must be of the same length."
        )

    for document_filepath, few_shot_prompt_filename, synthetic_queries_filename in zip(
        document_filepaths, few_shot_prompt_filenames, synthetic_queries_filenames
    ):
        documents = load_documents(document_filepath, documents_sampled)

        few_shot_examples, length_of_fewshot_prompt = load_few_shot_prompt(
            few_shot_prompt_filename, document_language, query_language
        )

        answer_gen_few_shot_examples, length_of_fewshot_prompt_answer_gen = generate_few_shot_prompts(
            few_shot_prompt_filename, document_language, query_language
        )

        wrong_language_answer_gen_few_shot_examples, length_of_fewshot_prompt_wrong_language_answer_gen = (
            generate_wrong_language_few_shot_prompts(
                few_shot_prompt_filename, document_language, query_language, second_language
            )
        )

        settings = {
            "few_shot_examples": few_shot_examples,
            "length_of_fewshot_prompt": length_of_fewshot_prompt,
            "answer_gen_few_shot_examples": answer_gen_few_shot_examples,
            "length_of_fewshot_prompt_answer_gen": length_of_fewshot_prompt_answer_gen,
            "device": device,
            "tokenizer": tokenizer,
            "model": model,
            "model_choice": model_choice,
            "percentiles": percentiles,
            "synthetic_queries_filename": synthetic_queries_filename,
            "document_language": document_language,
            "query_language": query_language,
            "second_language": second_language,
            "wrong_language_answer_gen_few_shot_examples": wrong_language_answer_gen_few_shot_examples,
            "length_of_fewshot_prompt_wrong_language_answer_gen": length_of_fewshot_prompt_wrong_language_answer_gen,
        }

        generate_synthetic_data(documents, synthetic_queries_filename, settings)
