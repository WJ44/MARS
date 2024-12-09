import json
import os
import re
import sys

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from mars.LLM_as_a_Judge_Adaptation.Filter_Synthetic_Queries import (
    filter_synthetic_queries,
    generate_index,
)
from mars.LLM_as_a_Judge_Adaptation.LLM_Generation_Functions import (
    check_generated_answer,
    generate_answer_llm_approach,
    generate_synthetic_query_llm_approach,
    generate_wrong_language_answer_llm_approach,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)


def clean_document(document: str) -> str:
    """
    Cleans the input document by removing unnecessary whitespace characters and replacing certain punctuation.

    Args:
        document (str): The original document text that needs to be cleaned.

    Returns:
        str: The cleaned document text.
    """
    # Replace carriage returns and tabs with a space, and reduce multiple newlines to a single newline
    cleaned_document = re.sub(r"\n+", "\n", document.replace("\r", " ").replace("\t", " ")).strip()
    # Replace equals signs and hyphens with spaces
    cleaned_document = cleaned_document.replace("=", " ").replace("-", " ")
    # Reduce multiple spaces to a single space
    cleaned_document = re.sub(r"\s+", " ", cleaned_document).strip()
    # Join words with a single space (this line seems redundant and could be removed if confirmed)
    cleaned_document = (" ").join(
        cleaned_document.split(" ")
    )  # [:512] - this part is commented out and can be ignored or removed
    return cleaned_document


def validate_input_file(df: pd.DataFrame, required_columns: list[str]) -> bool:
    """
    Validates that the DataFrame contains all required columns. Exits the program if any are missing.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        required_columns (List[str]): A list of strings representing the column names that are required in the DataFrame.

    Returns:
        bool: True if the DataFrame contains all required columns, otherwise the program will exit with an error.
    """
    # Identify any missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    # Exit the program with an error message if there are missing columns
    if missing_columns:
        sys.exit(f"Error: The DataFrame is missing the following required column(s): {', '.join(missing_columns)}.")
    return True


def load_model(model_choice: str) -> tuple:
    """
    Loads the specified model and tokenizer, and sets the model to evaluation mode on the appropriate device.

    Args:
        model_choice (str): The model identifier to load from the Hugging Face model hub.

    Returns:
        tuple: A tuple containing the model, tokenizer, and device.
    """
    if "Llama" in model_choice:
        tokenizer = AutoTokenizer.from_pretrained(model_choice)
        model = AutoModelForCausalLM.from_pretrained(model_choice)
    if "aya" in model_choice:
        tokenizer = AutoTokenizer.from_pretrained(model_choice)
        model = AutoModelForCausalLM.from_pretrained(model_choice, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_choice)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_choice)

    # Disable gradient calculations and set the model to evaluation mode
    torch.no_grad()
    model.eval()

    # Set the device to CUDA if available
    device = torch.device("cuda:0")
    model.to(device)

    return model, tokenizer, device


def load_documents(document_filepath: str, documents_sampled: int) -> pd.DataFrame:
    """
    Loads and processes documents for synthetic query and answer generation.

    Args:
        document_filepath (str): The path to the document file.
        documents_sampled (int): The number of documents to sample.

    Returns:
        pd.DataFrame: A DataFrame containing the processed documents.
    """
    documents = []
    # required_columns = ['Query', 'Document', 'Answer']
    required_columns = ["Document"]

    if "docs_aws" in document_filepath:
        with open(document_filepath, "r") as json_file:
            json_data = json.load(json_file)
            documents = [x["text"] for x in json_data]

        documents = pd.DataFrame(documents, columns=["document"])
    else:
        if not document_filepath.endswith(".tsv"):
            sys.exit(f"Error: The file {document_filepath} is not a TSV file.")
        try:
            documents = pd.read_csv(document_filepath, sep="\t")
            validate_input_file(documents, required_columns)
            documents.rename(columns={"Document": "document"}, inplace=True)
            documents["document"] = documents["document"].str.strip()
        except Exception as e:
            sys.exit(f"Error reading the file {document_filepath}: {e}")

    initial_count = len(documents)
    documents = documents[
        documents["document"].str.split().apply(len) >= 50
    ]  # Filter documents with less than 50 words. #TODO does not work for Japanese
    after_filter_count = len(documents)

    count = initial_count - after_filter_count

    if after_filter_count == 0:
        sys.exit(
            "All documents were less than 50 words, please provide a dataset with documents containing more than 50 words."
        )

    if documents_sampled > initial_count:
        print(
            f"\nThe `documents_sampled` parameter ({documents_sampled}) exceeds the available number of documents ({initial_count}). Sampling will be adjusted to the maximum available documents ({initial_count}).\n"
        )
        documents_sampled = initial_count

    if count > 0:
        print(f"Filtered out {count} documents because they had less than 50 words.")
        if documents_sampled > after_filter_count:
            print(
                f"Document sample is greater than document count. Sampling will be adjusted to {after_filter_count} documents\n"
            )
            documents_sampled = after_filter_count

    documents = documents.sample(n=documents_sampled)

    return documents


def load_few_shot_prompt(
    few_shot_prompt_filename: str, document_language: str, query_language: str
) -> tuple[str, int]:
    """
    Loads and processes a few-shot prompt from a TSV file.

    Args:
        few_shot_prompt_filename (str): The filename of the TSV file containing the few-shot prompts.

    Returns:
        tuple[str, int]: A tuple containing the few-shot examples as a string and the length of the few-shot prompt.
    """
    few_shot_prompt = pd.read_csv(few_shot_prompt_filename, sep="\t")
    few_shot_prompt = few_shot_prompt[few_shot_prompt["Context_Relevance_Label"] == "[[Yes]]"]

    if "Query" not in few_shot_prompt:
        few_shot_prompt["Query"] = few_shot_prompt["Question"]

    length_of_fewshot_prompt = len(few_shot_prompt)
    few_shot_examples = ""

    for row in range(len(few_shot_prompt)):
        few_shot_examples += f"Example {row + 1}:\n"
        few_shot_examples += (
            f"Document ({document_language}): {clean_document(few_shot_prompt.iloc[row]['Document'])}\n"
        )
        few_shot_examples += f"Question ({query_language}): {few_shot_prompt.iloc[row]['Query']}\n\n"

    return few_shot_examples, length_of_fewshot_prompt


def generate_few_shot_prompts(
    few_shot_prompt_filename: str, document_language: str, query_language: str
) -> tuple[str, int]:
    """
    Generates few-shot prompts for answer generation based on the provided dataset.

    Args:
        few_shot_prompt_filename (str): The filename of the TSV file containing the few-shot prompts.
        for_fever_dataset (bool): Flag indicating if the prompts are for the FEVER dataset.
        for_wow_dataset (bool): Flag indicating if the prompts are for the WoW dataset.

    Returns:
        tuple: A tuple containing the few-shot examples string and the length of the few-shot prompt.
    """
    # Load the few-shot prompt data
    answer_gen_few_shot_prompt = pd.read_csv(few_shot_prompt_filename, sep="\t")

    # Filter the prompts based on relevance and faithfulness labels
    answer_gen_few_shot_prompt = answer_gen_few_shot_prompt[
        (answer_gen_few_shot_prompt["Answer_Relevance_Label"] == "[[Yes]]")
        & (answer_gen_few_shot_prompt["Answer_Faithfulness_Label"] == "[[Yes]]")
        & (answer_gen_few_shot_prompt["Language_Consistency_Label"] == "[[Yes]]")
    ]

    # Get the length of the few-shot prompt
    length_of_fewshot_prompt_answer_gen = len(answer_gen_few_shot_prompt)

    # Rename 'Query' column to 'Question' if it exists
    if "Query" in answer_gen_few_shot_prompt.columns:
        answer_gen_few_shot_prompt["Question"] = answer_gen_few_shot_prompt["Query"]

    # Initialize the few-shot examples string
    answer_gen_few_shot_examples = ""

    # Construct the few-shot examples
    for row in range(len(answer_gen_few_shot_prompt)):
        answer_gen_few_shot_examples += f"Example {row + 1}:\n"
        answer_gen_few_shot_examples += (
            f"Document ({document_language}): {answer_gen_few_shot_prompt.iloc[row]['Document']}\n"
        )
        answer_gen_few_shot_examples += (
            f"Question ({query_language}): {answer_gen_few_shot_prompt.iloc[row]['Query']}\n"
        )
        answer_gen_few_shot_examples += (
            f"Answer ({query_language}): {answer_gen_few_shot_prompt.iloc[row]['Answer']}\n\n"
        )

    return answer_gen_few_shot_examples, length_of_fewshot_prompt_answer_gen


def generate_wrong_language_few_shot_prompts(
    few_shot_prompt_filename: str, document_language: str, query_language: str, second_language: str
) -> tuple[str, int]:
    """
    Generates few-shot prompts for answer generation based on the provided dataset.

    Args:
        few_shot_prompt_filename (str): The filename of the TSV file containing the few-shot prompts.
        for_fever_dataset (bool): Flag indicating if the prompts are for the FEVER dataset.
        for_wow_dataset (bool): Flag indicating if the prompts are for the WoW dataset.

    Returns:
        tuple: A tuple containing the few-shot examples string and the length of the few-shot prompt.
    """
    # Load the few-shot prompt data
    answer_gen_few_shot_prompt = pd.read_csv(few_shot_prompt_filename, sep="\t")

    # Filter the prompts based on relevance and faithfulness labels
    answer_gen_few_shot_prompt = answer_gen_few_shot_prompt[
        (answer_gen_few_shot_prompt["Answer_Relevance_Label"] == "[[Yes]]")
        & (answer_gen_few_shot_prompt["Answer_Faithfulness_Label"] == "[[Yes]]")
        & (answer_gen_few_shot_prompt["Language_Consistency_Label"] == "[[No]]")
    ]

    # Get the length of the few-shot prompt
    length_of_fewshot_prompt_answer_gen = len(answer_gen_few_shot_prompt)

    # Rename 'Query' column to 'Question' if it exists
    if "Query" in answer_gen_few_shot_prompt.columns:
        answer_gen_few_shot_prompt["Question"] = answer_gen_few_shot_prompt["Query"]

    # Initialize the few-shot examples string
    answer_gen_few_shot_examples = ""

    if query_language != document_language:
        answer_language = document_language
    else:
        answer_language = second_language

    # Construct the few-shot examples
    for row in range(len(answer_gen_few_shot_prompt)):
        answer_gen_few_shot_examples += f"Example {row + 1}:\n"
        answer_gen_few_shot_examples += (
            f"Document ({document_language}): {answer_gen_few_shot_prompt.iloc[row]['Document']}\n"
        )
        answer_gen_few_shot_examples += (
            f"Question ({query_language}): {answer_gen_few_shot_prompt.iloc[row]['Query']}\n"
        )
        answer_gen_few_shot_examples += (
            f"Answer ({answer_language}): {answer_gen_few_shot_prompt.iloc[row]['Answer']}\n\n"
        )

    return answer_gen_few_shot_examples, length_of_fewshot_prompt_answer_gen


def generate_query(document: str, settings: dict) -> list:
    """
    Generates synthetic queries for a given document.

    Args:
        document (str): The document text.
        settings (dict): Dictionary containing various settings and parameters required for generating synthetic queries.

    Returns:
        list: List of generated synthetic queries.
    """

    return generate_synthetic_query_llm_approach(  # LLM_Generation
        document,
        settings["few_shot_examples"],
        settings["length_of_fewshot_prompt"],
        settings["device"],
        settings["tokenizer"],
        settings["model"],
        settings["percentiles"],
        settings["document_language"],
        settings["query_language"],
    )


def generate_positive_synthetic_queries(documents: pd.DataFrame, settings: dict, chunk_size: int) -> pd.DataFrame:
    """
    Processes the documents to generate synthetic queries and remove duplicates.

    Args:
        documents (pd.DataFrame): DataFrame containing the documents.
        settings (dict): Dictionary containing various settings and parameters required for generating synthetic queries.
        chunk_size (int): Number of documents to process in each chunk.

    Returns:
        pd.DataFrame: DataFrame containing the documents with the generated synthetic queries.
    """
    num_documents = len(documents)
    target_queries = num_documents * 2

    all_queries = []
    initial_queries_per_document = 1
    synthetic_queries_filename = settings.get("synthetic_queries_filename", "intermediate_queries.tsv")

    for start in range(0, num_documents, chunk_size):
        end = min(start + chunk_size, num_documents)
        chunk = documents[start:end]

        with tqdm(
            total=len(chunk) * initial_queries_per_document,
            desc=f"Generating positive synthetic queries for documents {start} to {end}...",
        ) as pbar:
            for index, row in chunk.iterrows():
                document = row["document"]
                synthetic_queries = []
                for _ in range(initial_queries_per_document):
                    synthetic_queries.extend(generate_query(document, settings))
                    pbar.update(1)
                all_queries.append((index, document, synthetic_queries))

        all_queries_flat = [(index, document, query) for index, document, queries in all_queries for query in queries]
        synthetic_queries_df = pd.DataFrame(
            all_queries_flat, columns=["document_index", "document", "synthetic_query"]
        )

        print(f"Total queries generated before filtering: {len(synthetic_queries_df)}")

        synthetic_queries_df = synthetic_queries_df[synthetic_queries_df["synthetic_query"].str.len() > 10]
        print(f"Total queries after length filtering: {len(synthetic_queries_df)}")

        synthetic_queries_df = synthetic_queries_df.drop_duplicates(subset=["synthetic_query"])
        print(f"Total queries after deduplication: {len(synthetic_queries_df)}")

        document_index = generate_index(documents)
        synthetic_queries_df = filter_synthetic_queries(synthetic_queries_df, document_index)
        print(f"Total queries after filtering: {len(synthetic_queries_df)}")

        while len(synthetic_queries_df) < target_queries:
            print("Not enough queries. Generating more...")
            counts = synthetic_queries_df["document_index"].value_counts()
            documents_needing_more_queries = counts[counts < 1].index.tolist()

            additional_queries = []
            with tqdm(
                total=len(documents_needing_more_queries) * initial_queries_per_document,
                desc="Generating additional synthetic queries...",
            ) as pbar:
                for index in documents_needing_more_queries:
                    document = documents.loc[index, "document"]
                    for _ in range(initial_queries_per_document):
                        additional_queries.extend(generate_query(document, settings))
                        pbar.update(1)
                    all_queries.append((index, document, additional_queries))

            additional_queries_flat = [
                (index, document, query) for index, document, queries in additional_queries for query in queries
            ]
            additional_queries_df = pd.DataFrame(
                additional_queries_flat, columns=["document_index", "document", "synthetic_query"]
            )

            print(f"Additional queries generated before filtering: {len(additional_queries_df)}")

            additional_queries_df = additional_queries_df[additional_queries_df["synthetic_query"].str.len() > 10]
            print(f"Additional queries after length filtering: {len(additional_queries_df)}")

            synthetic_queries_df = pd.concat([synthetic_queries_df, additional_queries_df]).drop_duplicates(
                subset=["synthetic_query"]
            )
            synthetic_queries_df = filter_synthetic_queries(synthetic_queries_df, document_index)
            print(f"Total queries after adding additional queries and filtering: {len(synthetic_queries_df)}")

        # Save intermediate results
        synthetic_queries_df.to_csv(
            synthetic_queries_filename,
            mode="a",
            header=not os.path.exists(synthetic_queries_filename),
            index=False,
            sep="\t",
        )

    return synthetic_queries_df


def generate_answers(synthetic_queries: pd.DataFrame, answer_generation_settings: dict) -> pd.DataFrame:
    """
    Generate synthetic answers using the FLAN approach.

    Args:
        synthetic_queries (pd.DataFrame): DataFrame containing the synthetic queries.
        answer_generation_settings (dict): Dictionary containing settings and parameters for answer generation.

    Returns:
        pd.DataFrame: DataFrame containing the synthetic queries with generated answers.
    """
    tqdm.pandas(desc="Generating answers... (FLAN)", total=synthetic_queries.shape[0])
    synthetic_queries["generated_answer"] = synthetic_queries.progress_apply(
        lambda x: generate_answer_llm_approach(
            x["document"],
            x["synthetic_query"],
            answer_generation_settings["answer_gen_few_shot_examples"],
            answer_generation_settings["length_of_fewshot_prompt_answer_gen"],
            answer_generation_settings["device"],
            answer_generation_settings["tokenizer"],
            answer_generation_settings["model"],
            answer_generation_settings["document_language"],
            answer_generation_settings["query_language"],
        ),
        axis=1,
    )
    return synthetic_queries


def label_answers(synthetic_queries: pd.DataFrame) -> pd.DataFrame:
    """
    Label the generated answers for faithfulness and relevance.

    This function takes a DataFrame containing synthetic queries and their generated answers,
    and labels each answer for faithfulness and relevance. The labels are added as new columns
    in the DataFrame.

    Args:
        synthetic_queries (pd.DataFrame): DataFrame containing the synthetic queries and their generated answers.

    Returns:
        pd.DataFrame: DataFrame with additional columns for answer faithfulness and relevance labels.
    """

    # Label each generated answer for faithfulness
    synthetic_queries["Answer_Faithfulness_Label"] = [
        check_generated_answer(synthetic_queries.iloc[i]["generated_answer"]) for i in range(len(synthetic_queries))
    ]

    # Label each generated answer for relevance
    synthetic_queries["Answer_Relevance_Label"] = [
        check_generated_answer(synthetic_queries.iloc[i]["generated_answer"]) for i in range(len(synthetic_queries))
    ]

    return synthetic_queries


def generate_wrong_language_answers(synthetic_queries: pd.DataFrame, answer_generation_settings: dict) -> pd.DataFrame:
    """
    Generate synthetic answers using the FLAN approach.

    Args:
        synthetic_queries (pd.DataFrame): DataFrame containing the synthetic queries.
        answer_generation_settings (dict): Dictionary containing settings and parameters for answer generation.

    Returns:
        pd.DataFrame: DataFrame containing the synthetic queries with generated answers.
    """
    tqdm.pandas(desc="Generating answers... (FLAN)", total=synthetic_queries.shape[0])
    synthetic_queries["generated_answer_wrong_language"] = synthetic_queries.progress_apply(
        lambda x: generate_wrong_language_answer_llm_approach(
            x["document"],
            x["synthetic_query"],
            answer_generation_settings["wrong_language_answer_gen_few_shot_examples"],
            answer_generation_settings["length_of_fewshot_prompt_wrong_language_answer_gen"],
            answer_generation_settings["device"],
            answer_generation_settings["tokenizer"],
            answer_generation_settings["model"],
            answer_generation_settings["document_language"],
            answer_generation_settings["query_language"],
            answer_generation_settings["second_language"],
        ),
        axis=1,
    )
    return synthetic_queries


def shuffle_and_save(synthetic_queries: pd.DataFrame, synthetic_queries_filename: str) -> None:
    """
    Shuffle and save the synthetic queries to a specified file.

    This function shuffles the rows of the synthetic queries DataFrame and saves the result to a file in TSV format.

    Args:
        synthetic_queries (pd.DataFrame): The DataFrame containing synthetic queries to be shuffled and saved.
        synthetic_queries_filename (str): The filename where the shuffled synthetic queries will be saved.

    Returns:
        None
    """
    # Shuffle the synthetic queries DataFrame with a fixed random state for reproducibility
    synthetic_queries = synthetic_queries.sample(n=len(synthetic_queries), random_state=42)

    # Save the shuffled DataFrame to a TSV file without the index
    synthetic_queries.to_csv(synthetic_queries_filename, index=False, sep="\t")

    # Print completion messages
    print("Completed synthetic generation!")
    print(f"Saved synthetic queries file to: {synthetic_queries_filename}")


def generate_synthetic_data(documents: pd.DataFrame, synthetic_queries_filename: str, settings: dict) -> pd.DataFrame:
    total_documents = len(documents)

    synthetic_queries = generate_positive_synthetic_queries(documents, settings, total_documents)
    synthetic_queries = synthetic_queries.sample(n=total_documents, random_state=41, ignore_index=True)
    synthetic_queries["Context_Relevance_Label"] = "Yes"

    synthetic_queries = generate_answers(synthetic_queries, settings)
    synthetic_queries = label_answers(synthetic_queries)
    synthetic_queries["Language_Consistency_Label"] = "Yes"

    if settings["query_language"] != settings["document_language"] or settings["second_language"]:
        synthetic_queries = generate_wrong_language_answers(synthetic_queries, settings)

    synthetic_queries_copy_1 = synthetic_queries.copy()
    sampled_documents = documents["document"].sample(
        n=len(synthetic_queries_copy_1), random_state=42, ignore_index=True
    )
    synthetic_queries_copy_1["document"] = sampled_documents
    synthetic_queries_copy_1["Context_Relevance_Label"] = "No"
    # Set "Answer_Faithfulness_Label" to "No" for a random half of the rows
    synthetic_queries_copy_1["Answer_Faithfulness_Label"] = ""
    synthetic_queries_copy_1["Answer_Relevance_Label"] = ""
    half_indices = synthetic_queries_copy_1.sample(frac=0.5, random_state=42).index
    synthetic_queries_copy_1.loc[half_indices, "Answer_Faithfulness_Label"] = "No"
    # If a document is the same as the original document, set context relevance to ""
    same_document_mask = synthetic_queries_copy_1["document"] == synthetic_queries["document"]
    synthetic_queries_copy_1.loc[same_document_mask, "Context_Relevance_Label"] = ""
    synthetic_queries_copy_1.loc[same_document_mask, "Answer_Faithfulness_Label"] = ""

    synthetic_queries_copy_2 = synthetic_queries.copy()
    sampled_answers = synthetic_queries[["generated_answer", "generated_answer_wrong_language"]].sample(
        n=len(synthetic_queries_copy_2), random_state=42, ignore_index=True
    )
    synthetic_queries_copy_2["generated_answer"] = sampled_answers["generated_answer"]
    synthetic_queries_copy_2["generated_answer_wrong_language"] = sampled_answers["generated_answer_wrong_language"]
    synthetic_queries_copy_2["Answer_Faithfulness_Label"] = ""
    half_indices = synthetic_queries_copy_2.sample(frac=0.5, random_state=43).index
    synthetic_queries_copy_2.loc[half_indices, "Answer_Faithfulness_Label"] = "No"
    synthetic_queries_copy_2["Answer_Relevance_Label"] = "No"
    # If the shuffled answer is the same as the original answer, set both labels to ""
    same_answer_mask = synthetic_queries_copy_2["generated_answer"] == synthetic_queries["generated_answer"]
    synthetic_queries_copy_2.loc[same_answer_mask, "Answer_Faithfulness_Label"] = ""
    synthetic_queries_copy_2.loc[same_answer_mask, "Answer_Relevance_Label"] = ""
    synthetic_queries_copy_2["Context_Relevance_Label"] = ""

    synthetic_queries = pd.concat(
        [synthetic_queries, synthetic_queries_copy_1, synthetic_queries_copy_2], ignore_index=True
    )
    synthetic_queries_copy_3 = synthetic_queries.copy()
    synthetic_queries_copy_3["generated_answer"] = synthetic_queries_copy_3["generated_answer_wrong_language"]
    synthetic_queries_copy_3["Language_Consistency_Label"] = "No"

    synthetic_queries = pd.concat([synthetic_queries, synthetic_queries_copy_3], ignore_index=True)

    shuffle_and_save(synthetic_queries, synthetic_queries_filename)
