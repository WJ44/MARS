import ast
import copy
import csv
import json
import math
import random
import re
import time
from typing import Union

import numpy as np
import openai
import pandas as pd
import requests
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from mars.LLM_as_a_Judge_Adaptation.LLM_Synthetic_Generation import (
    generate_synthetic_contradictory_answers_api_approach,
)


def generate_synthetic_query_llm_approach(
    document: str,
    prompt: str,
    length_of_fewshot_prompt: int,
    device: torch.device,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    percentiles: list,
    document_language=None,
    query_language=None,
) -> list:
    """
    Generates synthetic queries based on a document using a language model.

    This function constructs a prompt with the document and generates queries based on the specified dataset type.
    It handles token length limitations by truncating the document if necessary.

    Args:
        document (str): The document to base the query on.
        prompt (str): The initial prompt to the language model.
        length_of_fewshot_prompt (int): The number of few-shot examples already included in the prompt.
        device: The device to run the model on (e.g., 'cuda' or 'cpu').
        tokenizer: The tokenizer for encoding the text.
        model: The language model used for generating text.
        percentiles (list): A list of percentiles for sampling during generation.
        for_fever_dataset (bool, optional): Flag to indicate if the function is used for the FEVER dataset. Defaults to False.
        for_wow_dataset (bool, optional): Flag to indicate if the function is used for the WoW dataset. Defaults to False.

    Returns:
        list: A list of synthetic queries generated by the model.
    """

    synthetic_queries = []

    # Construct the prompt without the document based on the dataset type
    prompt_without_document = prompt + "Example " + str(length_of_fewshot_prompt + 1) + ":\n"
    prompt_without_document += f"Document ({document_language}): \nQuestion ({query_language}): "

    # Calculate the length of tokens for the prompt and document
    prompt_tokens_length = tokenizer.encode(prompt_without_document, return_tensors="pt").to(device).shape[1]
    document_length = tokenizer.encode(document, return_tensors="pt").to(device).shape[1]

    # Check if the total length exceeds the model's maximum input size and truncate if necessary
    if prompt_tokens_length + document_length + 100 >= 2048:
        encoded_input = tokenizer(
            document, max_length=2048 - prompt_tokens_length - 100, truncation=True, return_tensors="pt"
        )
        truncated_document = tokenizer.decode(encoded_input["input_ids"][0][: 2048 - prompt_tokens_length - 100])
        document = truncated_document.replace("</s>", "")

    # Append the document to the prompt
    prompt += "Example " + str(length_of_fewshot_prompt + 1) + ":\n"
    prompt += f"Document ({document_language}): " + document + "\n"
    prompt += f"Question ({query_language}): "

    # Encode the complete prompt
    if model.config.model_type == "cohere":
        prompt = [{"role": "user", "content": prompt}]

        input_ids = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_promt=True, return_tensors="pt"
        ).to(model.device)
        prompt_len = len(input_ids[0])
        # input_ids = tokenizer.encode(prompt, max_length=2048, truncation=True, return_tensors='pt').to(device)

        # Set the maximum length for the generated text based on the dataset
        max_length = 32

        # Generate queries for each percentile
        for percentile in percentiles:
            if input_ids.shape[0] != 1 or input_ids.shape[1] >= 2048:
                print("Length of problematic input ids: " + str(input_ids.shape))
                print("Length of problematic document: " + str(len(encoded_input["input_ids"][0])))
                assert False
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_length,
                do_sample=True,
                top_p=percentile,
                num_return_sequences=1,
            )

            query = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

            synthetic_queries.append(query)
    else:
        input_ids = tokenizer.encode(prompt, max_length=2048, truncation=True, return_tensors="pt").to(device)

        # Set the maximum length for the generated text based on the dataset
        max_length = 32

        # Generate queries for each percentile
        for percentile in percentiles:
            if input_ids.shape[0] != 1 or input_ids.shape[1] >= 2048:
                print("Length of problematic input ids: " + str(input_ids.shape))
                print("Length of problematic document: " + str(len(encoded_input["input_ids"][0])))
                assert False
            outputs = model.generate(
                input_ids=input_ids, max_length=max_length, do_sample=True, top_p=percentile, num_return_sequences=1
            )

            query = tokenizer.decode(outputs[0], skip_special_tokens=True)

            synthetic_queries.append(query)
    return synthetic_queries


def generate_answer_llm_approach(
    document: str,
    question: str,
    prompt: str,
    length_of_fewshot_prompt: int,
    device: torch.device,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    document_language=None,
    query_language=None,
) -> str:
    """
    Generates an answer using a language model based on the provided document and question.

    This function constructs a prompt for the language model by appending the document and question to a base prompt.
    It then encodes the prompt and checks if the total token length exceeds the model's maximum input size.
    If it does, the document is truncated to fit. The function finally generates an answer using the model.

    Args:
        document (str): The document text to be used for generating the answer.
        question (str): The question text based on the document.
        prompt (str): The initial prompt text to which the document and question will be appended.
        length_of_fewshot_prompt (int): The ordinal number of the current example in the context of few-shot learning.
        device: str: The device (CPU/GPU) on which the tokenizer and model are loaded.
        tokenizer: The tokenizer used for encoding the text.
        model: The model used for generating the answer.
        for_fever_dataset (bool, optional): Flag to indicate if the function is being used for the FEVER dataset. Defaults to False.
        for_wow_dataset (bool, optional): Flag to indicate if the function is being used for the WoW dataset. Defaults to False.

    Returns:
        str: The generated answer text.
    """
    # Construct the prompt without the document based on the dataset type
    prompt_without_document = prompt + "Example " + str(length_of_fewshot_prompt + 1) + ":\n"
    prompt_without_document += (
        f"Document ({document_language}): \nQuestion ({query_language}): \nAnswer ({query_language}): "
    )

    # Calculate the token lengths for the prompt, document, and question
    prompt_tokens_length = tokenizer.encode(prompt_without_document, return_tensors="pt").to(device).shape[1]
    document_length = tokenizer.encode(document, return_tensors="pt").to(device).shape[1]
    question_length = tokenizer.encode(question, return_tensors="pt").to(device).shape[1]

    # Check if the total length exceeds the model's maximum input size and truncate if necessary
    if prompt_tokens_length + document_length + question_length + 100 >= 2048:
        reduction_length = prompt_tokens_length + question_length + 100
        encoded_input = tokenizer(document, max_length=2048 - reduction_length, truncation=True, return_tensors="pt")
        truncated_document = tokenizer.decode(encoded_input["input_ids"][0][: 2048 - reduction_length])
        document = truncated_document.replace("</s>", "")

    # Append the document and question to the prompt
    prompt += "Example " + str(length_of_fewshot_prompt + 1) + ":\n"
    prompt += f"Document ({document_language}): " + document + "\n"
    prompt += f"Question ({query_language}): " + question + "\n"
    prompt += f"Answer ({query_language}): "

    # Encode the complete prompt
    if model.config.model_type == "cohere":
        prompt = [{"role": "user", "content": prompt}]

        input_ids = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_promt=True, return_tensors="pt"
        ).to(model.device)
        prompt_len = len(input_ids[0])
        # input_ids = tokenizer.encode(prompt, max_length=2048, truncation=True, return_tensors='pt').to(device)

        # Check for encoding issues and generate the answer
        if input_ids.shape[0] != 1 or input_ids.shape[1] >= 2048:
            print("Length of problematic input ids: " + str(input_ids.shape))
            print("Length of problematic document: " + str(len(encoded_input["input_ids"][0])))
            assert False
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=256, do_sample=True, top_p=0.05, num_return_sequences=1
        )

        answer = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    else:
        input_ids = tokenizer.encode(prompt, max_length=2048, truncation=True, return_tensors="pt").to(device)

        # Check for encoding issues and generate the answer
        if input_ids.shape[0] != 1 or input_ids.shape[1] >= 2048:
            print("Length of problematic input ids: " + str(input_ids.shape))
            print("Length of problematic document: " + str(len(encoded_input["input_ids"][0])))
            assert False
        outputs = model.generate(
            input_ids=input_ids, max_length=256, do_sample=True, top_p=0.05, num_return_sequences=1
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


def generate_synthetic_query_openai_approach(
    document: str, system_prompt: str, few_shot_examples: str, temperatures: list, length_of_fewshot_prompt: int
) -> list:
    """
    Generates synthetic queries using the OpenAI API with different temperatures.

    This function takes a document and a system prompt, combines them with few-shot examples,
    and queries the OpenAI API to generate synthetic queries at various temperatures.

    Args:
        document (str): The document text based on which the query is to be generated.
        system_prompt (str): The prompt that guides the AI on how to respond.
        few_shot_examples (str): Preceding examples to prime the model for consistent responses.
        temperatures (list): A list of temperature settings to use for generating diverse outputs.
        length_of_fewshot_prompt (int): The number of few-shot examples provided.

    Returns:
        list: A list of synthetic documents generated at different temperatures.
    """

    time.sleep(1)

    # Initialize a list to store the generated documents
    synth_documents_generated = []

    # Iterate over each temperature value provided in the temperatures list
    for temp in temperatures:
        try:
            # Construct the user prompt by appending the document to the few-shot examples
            user_prompt = few_shot_examples
            user_prompt += "Document: " + document + "\n"
            user_prompt += "Question: "

            # Define the message format for the API request
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

            # Make a request to the OpenAI API with the specified model and temperature
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",  # Consider using "gpt-4" for potentially better results
                messages=messages,
                temperature=temp,
            )

            # Extract the content from the response and add it to the list of generated documents
            final_response = response["choices"][0]["message"]["content"]
            synth_documents_generated.append(final_response)
        except Exception as e:
            # Handle exceptions by logging the error and pausing before retrying
            print("Error with OpenAI! Waiting 30 seconds...")
            print("Error: " + str(e))
            time.sleep(30)

    # Ensure that at least one document has been generated
    assert len(synth_documents_generated) >= 1, "No documents were generated."

    # Return the list of generated synthetic documents
    return synth_documents_generated


def generate_answer_from_context(document: str, synth_question: str) -> str:
    """
    Generates an answer from a given document context and a synthetic question using the OpenAI API.

    This function attempts to generate an answer up to five times if an error occurs during the API request.
    It introduces a delay before making the request to avoid rapid requests that might lead to API rate limits.

    Args:
    document (str): The document context from which the answer needs to be generated.
    synth_question (str): The synthetic question based on the document context.

    Returns:
    str: The generated answer from the API or an error message if unable to generate after retries.

    Raises:
    Exception: Outputs an error message if the API request fails continuously.
    """

    # Introduce a slight delay before making the API request
    time.sleep(1)

    # Attempt to generate an answer up to 5 times in case of failures
    for attempt in range(5):
        try:
            # Define the system prompt that guides the model's response
            system_prompt = (
                "You are a helpful assistant built by Databricks, you are not human, you are good at helping to answer a query based on the context step by step, the context is a document. "
                "If the query doesn't form a complete question, just say I don't know. "
                "If there is a good answer from the context, try to summarize the context as the answer. "
                "If you don't know the answer, just say I don't know. "
                "If there is no enough information to determine the answer, just say I don't know. "
                "If the context is irrelevant to the question, just say I don't know."
            )
            # Construct the user prompt that includes the synthetic question and the document context
            user_prompt = f"Here is the question: {synth_question}\nHere is the context: {document}?"

            # Prepare the message payload for the API request
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

            # Make the API request to OpenAI
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=messages, temperature=0.0)

            # Extract the response content from the API response
            final_response = response["choices"][0]["message"]["content"]
            return final_response

        except Exception as e:
            # Log the error and attempt again if less than 5 attempts have been made
            print(f"Error querying OpenAI on attempt {attempt + 1}: {str(e)}. Attempting again...")

    # If all attempts fail, raise an exception
    raise Exception("Failed to generate an answer from the context after multiple attempts.")


def generate_contradictory_answer_from_context(document: str, synth_question: str) -> str:
    """
    Generates an answer that intentionally contradicts the information provided in the document.

    This function attempts to generate a contradictory answer by instructing the model to create
    false information that disagrees with the content of the document. It tries up to 5 times to get
    a valid response from the OpenAI API.

    Args:
        document (str): The document context from which the answer needs to contradict.
        synth_question (str): The synthetic question based on the document.

    Returns:
        str: The generated contradictory answer from the model or raises an exception after retries.

    Raises:
        Exception: If the API fails to generate an answer after multiple attempts.
    """

    # Introduce a slight delay before making the API request to avoid rapid repeated requests
    time.sleep(1)

    # Attempt to generate an answer up to 5 times in case of failures
    for attempt in range(5):
        try:
            # Construct the system prompt that instructs the model to generate a contradictory answer
            system_prompt = (
                "Create an answer for the given question that contradicts the provided document. "
                "You should create false information that disagrees with what exists within the content of the document."
            )
            # Construct the user prompt that includes the synthetic question and the document context
            user_prompt = f"Question: {synth_question}\nDocument: {document}"
            # Combine system and user prompts
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"

            # Prepare the message payload for the API request
            messages = [{"role": "user", "content": combined_prompt}]

            # Make the API request to OpenAI
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=messages)

            # Extract the response content from the API response
            final_response = response["choices"][0]["message"]["content"]
            return final_response

        except Exception as e:
            # Log the error and attempt again if less than 5 attempts have been made
            print(f"Error querying OpenAI on attempt {attempt + 1}: {str(e)}. Attempting again...")

    # If all attempts fail, raise an exception
    raise Exception("Failed to generate a contradictory answer from the context after multiple attempts.")


def check_generated_answer(answer: str) -> str:
    """
    Checks if the provided answer contains any problematic phrases that indicate uncertainty or lack of knowledge.

    This function iterates through a list of predefined problematic phrases. If any of these phrases are found
    in the answer, it returns "No", indicating the presence of such phrases. If none of the phrases are found,
    it returns "Yes", indicating the answer does not contain problematic content.

    Args:
        answer (str): The answer text to be checked for problematic phrases.

    Returns:
        str: "No" if any problematic phrases are found in the answer, otherwise "Yes".
    """
    # List of phrases that are considered problematic because they indicate uncertainty
    problematic_phrases = ["I don't know", "don't know", "i don't know"]

    # Check each phrase in the list to see if it is present in the answer
    for phrase in problematic_phrases:
        if phrase in answer.lower():  # Convert answer to lowercase to ensure the check is case-insensitive
            return "No"

    # If no problematic phrases are found, return "Yes"
    return "Yes"


def generate_contradictory_answer_examples(
    queries_dataset: pd.DataFrame,
    number_of_contradictory_answers_to_generate: int,
    few_shot_examples_for_contradictory_answers=None,
    api_model=False,
    synthetic_contradictory_answer_prompt=None,
    device=None,
    tokenizer=None,
    model=None,
    for_fever_dataset=None,
    for_wow_dataset=None,
    document_language=None,
    query_language=None,
) -> pd.DataFrame:
    """
    Generates a specified number of contradictory answers from a given dataset of queries.

    This function processes a dataset to generate contradictory answers using either a direct approach or a language model-based approach, depending on the availability of few-shot examples. It also removes problematic phrases from the generated answers to ensure the quality of contradictions. The function then merges the original dataset with the newly generated contradictory answers and shuffles the dataset to randomize the order of entries.

    Args:
        queries_dataset (pd.DataFrame): The dataset containing queries for which contradictory answers are to be generated.
        number_of_contradictory_answers_to_generate (int): The number of contradictory answers to generate.
        few_shot_examples_for_contradictory_answers (Optional): Predefined examples to guide the generation process if using a language model.
        device: The device on which the model is loaded (e.g., 'cuda:0').
        tokenizer: The tokenizer used for processing text for the model.
        model: The model used for generating answers.
        for_fever_dataset (Optional): Flag to indicate if the generation is for the FEVER dataset.
        for_wow_dataset (Optional): Flag to indicate if the generation is for the WoW dataset.

    Returns:
        pd.DataFrame: The updated dataset with generated contradictory answers, labels, and shuffled entries.
    """

    def remove_problematic_contradictory_phrases(text: str) -> str:
        """
        Removes predefined problematic phrases from the generated text.

        Args:
            text (str): The text from which problematic phrases are to be removed.

        Returns:
            str: The cleaned text after removing problematic phrases.
        """
        if text is None:
            return text

        problematic_phrases = ["Contradictory Answer:", "The false information created is:", "Incorrect Answer: "]
        text_split = text.split(":")
        if len(text_split) > 1:
            return text_split[1].strip()
        else:
            return text

    # Prepare the dataset for processing
    queries_dataset_copy = queries_dataset.copy()
    queries_dataset_copy = queries_dataset_copy.drop_duplicates(subset=["synthetic_query"])

    # Limit the number of generations to the minimum of the requested number or the dataset size
    number_of_contradictory_answers_to_generate = min(
        number_of_contradictory_answers_to_generate, len(queries_dataset_copy)
    )
    queries_dataset_copy = queries_dataset_copy.sample(n=number_of_contradictory_answers_to_generate, random_state=42)

    contradictory_answers = []
    contradictory_labels = []

    # Generate contradictory answers for each query in the dataset
    for i in tqdm(range(len(queries_dataset_copy))):
        if api_model:
            contradictory_answer_generated = generate_synthetic_contradictory_answers_api_approach(
                queries_dataset_copy.iloc[i]["document"],
                queries_dataset_copy.iloc[i]["synthetic_query"],
                synthetic_contradictory_answer_prompt,
                few_shot_examples_for_contradictory_answers,
                model,
                for_fever_dataset=for_fever_dataset,
                for_wow_dataset=for_wow_dataset,
            )
        else:
            contradictory_answer_generated = generate_contradictory_answer_llm_approach(
                queries_dataset_copy.iloc[i]["document"],
                queries_dataset_copy.iloc[i]["synthetic_query"],
                few_shot_examples_for_contradictory_answers,
                device,
                tokenizer,
                model,
                for_fever_dataset=for_fever_dataset,
                for_wow_dataset=for_wow_dataset,
                document_language=document_language,
                query_language=query_language,
            )

        contradictory_answer_generated = remove_problematic_contradictory_phrases(contradictory_answer_generated)

        contradictory_answers.append(contradictory_answer_generated)
        contradictory_labels.append("No")

    queries_dataset_copy["generated_answer"] = contradictory_answers
    queries_dataset_copy["Answer_Faithfulness_Label"] = contradictory_labels
    queries_dataset_copy["Answer_Relevance_Label"] = contradictory_labels

    # Additional Generation Method: Answer Randomization
    queries_dataset_copy_2 = queries_dataset.copy()
    queries_dataset_copy_2 = queries_dataset_copy_2.drop_duplicates(subset=["synthetic_query"])
    queries_dataset_copy_2 = queries_dataset_copy_2.sample(
        n=number_of_contradictory_answers_to_generate, random_state=42
    )

    total_answers = queries_dataset[queries_dataset["Answer_Relevance_Label"] == "Yes"]["generated_answer"].tolist()
    total_answers = [answer for answer in total_answers if isinstance(answer, str)]
    total_answers = [str(answer) for answer in total_answers]
    total_answers = [answer for answer in total_answers if len(answer) > 5]

    contradictory_answers_2 = []
    contradictory_labels_2 = []
    for i in tqdm(range(len(queries_dataset_copy_2))):

        random_answer = random.choice(total_answers)
        contradictory_answers_2.append(random_answer)
        contradictory_labels_2.append("No")

    queries_dataset_copy_2["generated_answer"] = contradictory_answers_2
    queries_dataset_copy_2["Answer_Relevance_Label"] = contradictory_labels_2
    queries_dataset_copy_2["Answer_Faithfulness_Label"] = contradictory_labels_2

    # Combine the original dataset with the two copied datasets
    queries_dataset = pd.concat(
        [queries_dataset, queries_dataset_copy, queries_dataset_copy_2], axis=0, ignore_index=True
    )

    # Shuffle the combined dataframe to randomize the order of entries
    queries_dataset = queries_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    return queries_dataset


def generate_contradictory_answer_llm_approach(
    document: str,
    question: str,
    prompt: str,
    device: torch.device,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    for_fever_dataset: bool = False,
    for_wow_dataset: bool = False,
    document_language=None,
    query_language=None,
) -> str:
    """
    Generates a contradictory answer using a language model approach based on the provided document and question.

    This function constructs a prompt dynamically based on whether it is being used for the FEVER or WoW dataset,
    or a general dataset. It then checks if the total token length exceeds the model's maximum input size and truncates
    the document if necessary. Finally, it generates a contradictory answer using the model.

    Args:
        document (str): The document text based on which the contradictory answer is to be generated.
        question (str): The question text based on the document.
        prompt (str): The initial prompt text to which the document and question will be appended.
        device: The device (CPU/GPU) on which the tokenizer and model are loaded.
        tokenizer: The tokenizer used for encoding the text.
        model: The model used for generating the answer.
        for_fever_dataset (bool, optional): Flag to indicate if the function is being used for the FEVER dataset. Defaults to False.
        for_wow_dataset (bool, optional): Flag to indicate if the function is being used for the WoW dataset. Defaults to False.

    Returns:
        str: The generated contradictory answer text.
    """

    # Construct the initial part of the prompt without the document based on the dataset type
    prompt_without_document = prompt + "Example " + str(prompt.count("Example") + 1) + ":\n"
    if for_fever_dataset:
        prompt_without_document += (
            f"Document ({document_language}): \nStatement ({query_language}): \nIncorrect Answer ({query_language}): "
        )
    elif for_wow_dataset:
        prompt_without_document += (
            f"Document ({document_language}): \nDialogue ({query_language}): \nIncorrect Response ({query_language}): "
        )
    else:
        prompt_without_document += (
            f"Document ({document_language}): \nQuestion ({query_language}): \nIncorrect Answer ({query_language}): "
        )

    # Calculate the token lengths for the prompt, document, and question
    prompt_tokens_length = tokenizer.encode(prompt_without_document, return_tensors="pt").to(device).shape[1]
    document_length = tokenizer.encode(document, return_tensors="pt").to(device).shape[1]
    question_length = tokenizer.encode(question, return_tensors="pt").to(device).shape[1]

    # Check if the total length exceeds the model's maximum input size and truncate if necessary
    if prompt_tokens_length + document_length + question_length + 100 >= 2048:
        reduction_length = prompt_tokens_length + question_length + 100
        encoded_input = tokenizer(document, max_length=2048 - reduction_length, truncation=True, return_tensors="pt")
        truncated_document = tokenizer.decode(encoded_input["input_ids"][0][: 2048 - reduction_length])
        document = truncated_document.replace("</s>", "")

    # Append the document and question to the prompt
    prompt += "Example " + str(prompt.count("Example") + 1) + ":\n"
    prompt += f"Document ({document_language}): " + document + "\n"
    if for_fever_dataset:
        prompt += f"Statement ({query_language}): " + question + "\n"
        prompt += f"Incorrect Answer ({query_language}): "
    elif for_wow_dataset:
        prompt += f"Dialogue ({query_language}): " + question + "\n"
        prompt += f"Incorrect Response ({query_language}): "
    else:
        prompt += f"Question ({query_language}): " + question + "\n"
        prompt += f"Incorrect Answer ({query_language}): "

    # Encode the complete prompt
    input_ids = tokenizer.encode(prompt, max_length=2048, truncation=True, return_tensors="pt").to(device)

    # Check for encoding issues and generate the answer
    if input_ids.shape[0] != 1 or input_ids.shape[1] >= 2048:
        print("Length of problematic input ids: " + str(input_ids.shape))
        print("Length of problematic document: " + str(len(encoded_input["input_ids"][0])))
        assert False

    outputs = model.generate(input_ids=input_ids, max_length=256, do_sample=True, top_p=1.0, num_return_sequences=1)

    query = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return query


def generate_wrong_language_answer_llm_approach(
    document: str,
    question: str,
    prompt: str,
    length_of_fewshot_prompt: int,
    device: torch.device,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    document_language=None,
    query_language=None,
    second_language=None,
) -> str:
    """
    Generates an answer using a language model based on the provided document and question.

    This function constructs a prompt for the language model by appending the document and question to a base prompt.
    It then encodes the prompt and checks if the total token length exceeds the model's maximum input size.
    If it does, the document is truncated to fit. The function finally generates an answer using the model.

    Args:
        document (str): The document text to be used for generating the answer.
        question (str): The question text based on the document.
        prompt (str): The initial prompt text to which the document and question will be appended.
        length_of_fewshot_prompt (int): The ordinal number of the current example in the context of few-shot learning.
        device: str: The device (CPU/GPU) on which the tokenizer and model are loaded.
        tokenizer: The tokenizer used for encoding the text.
        model: The model used for generating the answer.
        for_fever_dataset (bool, optional): Flag to indicate if the function is being used for the FEVER dataset. Defaults to False.
        for_wow_dataset (bool, optional): Flag to indicate if the function is being used for the WoW dataset. Defaults to False.

    Returns:
        str: The generated answer text.
    """
    if query_language != document_language:
        answer_language = document_language
    elif second_language:
        answer_language = second_language

    # Construct the prompt without the document based on the dataset type
    prompt_without_document = prompt + "Example " + str(length_of_fewshot_prompt + 1) + ":\n"
    prompt_without_document += (
        f"Document ({document_language}): \nQuestion ({query_language}): \nAnswer ({answer_language}): "
    )

    # Calculate the token lengths for the prompt, document, and question
    prompt_tokens_length = tokenizer.encode(prompt_without_document, return_tensors="pt").to(device).shape[1]
    document_length = tokenizer.encode(document, return_tensors="pt").to(device).shape[1]
    question_length = tokenizer.encode(question, return_tensors="pt").to(device).shape[1]

    # Check if the total length exceeds the model's maximum input size and truncate if necessary
    if prompt_tokens_length + document_length + question_length + 100 >= 2048:
        reduction_length = prompt_tokens_length + question_length + 100
        encoded_input = tokenizer(document, max_length=2048 - reduction_length, truncation=True, return_tensors="pt")
        truncated_document = tokenizer.decode(encoded_input["input_ids"][0][: 2048 - reduction_length])
        document = truncated_document.replace("</s>", "")

    # Append the document and question to the prompt
    prompt += "Example " + str(length_of_fewshot_prompt + 1) + ":\n"
    prompt += f"Document ({document_language}): " + document + "\n"
    prompt += f"Question ({query_language}): " + question + "\n"
    prompt += f"Answer ({answer_language}): "

    # Encode the complete prompt
    if model.config.model_type == "cohere":
        prompt = [{"role": "user", "content": prompt}]

        input_ids = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_promt=True, return_tensors="pt"
        ).to(model.device)
        prompt_len = len(input_ids[0])
        # input_ids = tokenizer.encode(prompt, max_length=2048, truncation=True, return_tensors='pt').to(device)

        # Check for encoding issues and generate the answer
        if input_ids.shape[0] != 1 or input_ids.shape[1] >= 2048:
            print("Length of problematic input ids: " + str(input_ids.shape))
            print("Length of problematic document: " + str(len(encoded_input["input_ids"][0])))
            assert False
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=256, do_sample=True, top_p=0.05, num_return_sequences=1
        )

        answer = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    else:
        input_ids = tokenizer.encode(prompt, max_length=2048, truncation=True, return_tensors="pt").to(device)

        # Check for encoding issues and generate the answer
        if input_ids.shape[0] != 1 or input_ids.shape[1] >= 2048:
            print("Length of problematic input ids: " + str(input_ids.shape))
            print("Length of problematic document: " + str(len(encoded_input["input_ids"][0])))
            assert False
        outputs = model.generate(
            input_ids=input_ids, max_length=256, do_sample=True, top_p=0.05, num_return_sequences=1
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer