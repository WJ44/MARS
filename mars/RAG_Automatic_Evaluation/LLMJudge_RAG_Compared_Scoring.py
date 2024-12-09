import os
import random
import re

import datasets
import evaluate
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    MptForSequenceClassification,
)

tqdm.pandas()

# Set random seed for reproducibility
random_state = 44

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ["PYTHONHASHSEED"] = str(random_state)
os.environ["HUGGINGFACE_HUB_DISABLE_DOWNLOAD_PROGRESS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from mars.RAG_Automatic_Evaluation.ppi import pp_mean_iid_asymptotic


class CustomBERTModel(nn.Module):
    def __init__(self, number_of_labels: int, model_choice: str):
        """
        Initializes the CustomBERTModel with the specified number of labels and model choice.

        Args:
            number_of_labels (int): The number of labels for the classification task.
            model_choice (str): The model choice for the encoder.
        """
        self.model_choice = model_choice

        super(CustomBERTModel, self).__init__()

        if model_choice in ["mosaicml/mpt-1b-redpajama-200b"]:
            model_encoding = MptForSequenceClassification.from_pretrained(
                "mosaicml/mpt-1b-redpajama-200b", trust_remote_code=True
            )
            embedding_size = 2048
            self.encoderModel = model_encoding

        elif model_choice in ["google/t5-large-lm-adapt", "google/t5-xl-lm-adapt"]:
            model_encoding = AutoModelForSequenceClassification.from_pretrained(model_choice)
            embedding_size = 1024
            self.encoderModel = model_encoding

        elif model_choice in ["roberta-large", "microsoft/deberta-v3-large"]:
            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 1024
            self.encoderModel = model_encoding

        elif model_choice in ["microsoft/deberta-v2-xlarge", "microsoft/deberta-v2-xxlarge"]:
            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 1536
            self.encoderModel = model_encoding
        elif "electra" in model_choice.lower():
            config = AutoConfig.from_pretrained(model_choice)
            self.encoderModel = AutoModel.from_pretrained(model_choice)
            embedding_size = config.hidden_size
            self.embedding_size = embedding_size
        elif model_choice in ["microsoft/deberta-v3-xsmall"]:
            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 384
            self.encoderModel = model_encoding

        else:
            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 768
            self.encoderModel = model_encoding

        self.encoderModel.eval()
        self.classifier = nn.Sequential(nn.Linear(embedding_size, 256), nn.Linear(256, number_of_labels))
        self.embedding_size = embedding_size

    def forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for the model.

        Parameters:
        ids (torch.Tensor): Input tensor containing token ids.
        mask (torch.Tensor): Attention mask tensor.
        labels (torch.Tensor, optional): Labels tensor for supervised training. Defaults to None.
        decoder_input_ids (torch.Tensor, optional): Decoder input ids for models that require them. Defaults to None.

        Returns:
        torch.Tensor: The output logits or classifier output depending on the model choice.
        """
        model_choice = self.model_choice

        # For specific models, use the encoder model to get the logits directly
        if model_choice in [
            "t5-small",
            "google/t5-xl-lm-adapt",
            "google/t5-large-lm-adapt",
            "mosaicml/mpt-1b-redpajama-200b",
        ]:
            total_output = self.encoderModel(input_ids=ids, attention_mask=mask)
            return total_output["logits"]
        elif "electra" in self.model_choice.lower():
            outputs = self.encoderModel(input_ids=ids, attention_mask=mask)
            pooled_output = outputs.last_hidden_state[:, 0]
            return self.classifier(pooled_output)
        else:
            # For other models, process the output through the classifier
            total_output = self.encoderModel(ids, attention_mask=mask)
            sequence_output = total_output["last_hidden_state"]

            # Format the last hidden state and pass it through the classifier
            last_hidden_state_formatted = sequence_output[:, 0, :].view(-1, self.embedding_size)
            linear2_output = self.classifier(last_hidden_state_formatted)

            return linear2_output


def combine_query_document(query: str, document: str = None, answer: str = None) -> str:
    """
    Combines a query and a document into a single string, optionally including an answer.

    Parameters:
    query (str): The query string.
    document (str): The document string.
    answer (str, optional): The answer string. Defaults to None.

    Returns:
    str: A combined string of the query, cleaned document, and optionally the answer.
    """
    # Clean the document by removing extra newlines, carriage returns, and tabs
    if document:
        cleaned_document = re.sub(r"\n+", "\n", document.replace("\r", " ").replace("\t", " ")).strip()
        cleaned_document = cleaned_document.replace("=", " ").replace("-", " ")
        cleaned_document = re.sub(r"\s+", " ", cleaned_document).strip()
        cleaned_document = " ".join(cleaned_document.split(" ")[:512])  # TODO does not work for Japanese

    # Truncate the query if it is too long
    if len(query.split(" ")) > 100:  # TODO does not work for Japanese
        query = " ".join(query.split(" ")[:30])

    # Combine query and cleaned document, optionally including the answer
    if answer is None:
        return query + " | " + cleaned_document
    elif document is None:
        return query + " | " + answer
    else:
        try:
            return query + " | " + cleaned_document + " | " + answer
        except Exception as e:
            print("Error with combine_query_document")
            print("Query: " + str(query))
            print("Cleaned Document: " + str(cleaned_document))
            print("Answer: " + str(answer))
            return str(query) + " | " + str(cleaned_document) + " | " + str(answer)


def tokenize_function(tokenizer, examples: dict) -> dict:
    """
    Tokenizes the input examples using the provided tokenizer.

    Parameters:
    tokenizer (Tokenizer): The tokenizer to be used for tokenizing the text.
    examples (dict): A dictionary containing the text to be tokenized.
                     It should have a key "text" with the text data as its value.

    Returns:
    dict: A dictionary containing the tokenized text with padding and truncation applied.
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def prepare_dataset_for_evaluation(
    dataframe: pd.DataFrame, label_column: str, text_column: str, assigned_batch_size: int, tokenizer
) -> DataLoader:
    """
    Prepares a dataset for evaluation by tokenizing the text and creating a DataLoader.

    Parameters:
    dataframe (pd.DataFrame): The input dataframe containing the data.
    label_column (str): The name of the column containing the labels.
    text_column (str): The name of the column containing the text data.
    assigned_batch_size (int): The batch size to be used for the DataLoader.
    tokenizer: The tokenizer to be used for tokenizing the text.

    Returns:
    DataLoader: A DataLoader object for the tokenized dataset.
    """
    from datasets.utils.logging import disable_progress_bar

    disable_progress_bar()

    # Extract text and labels from the dataframe
    test_set_text = [dataframe.iloc[i][text_column] for i in range(len(dataframe))]

    if label_column in dataframe.columns:
        test_set_label = dataframe[label_column].tolist()
        # Create a pandas DataFrame with the extracted text and labels
        test_dataset_pandas = pd.DataFrame({"label": test_set_label, "text": test_set_text})
    else:
        # Create a pandas DataFrame with only the text data
        test_dataset_pandas = pd.DataFrame({"text": test_set_text})

    # Convert the pandas DataFrame to an Arrow Table and then to a Hugging Face Dataset
    test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
    test_dataset_arrow = datasets.Dataset(test_dataset_arrow)

    # Create a DatasetDict with the test dataset
    classification_dataset = datasets.DatasetDict({"test": test_dataset_arrow})

    # Tokenize the dataset
    tokenized_datasets = classification_dataset.map(
        lambda examples: tokenize_function(tokenizer, examples), batched=True
    )

    # Remove the original text column and rename the label column to "labels"
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    if "label" in tokenized_datasets["test"].column_names:
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Set the format of the dataset to PyTorch tensors
    tokenized_datasets.set_format("torch")

    # Create a DataLoader for the tokenized dataset
    eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=assigned_batch_size)

    return eval_dataloader


def calculate_ppi(
    Y_labeled: np.ndarray,
    Yhat_labeled: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    alpha: float,
) -> tuple:
    """
    Calculate prediction-powered inference (PPI) and classical inference intervals.

    Parameters:
    Y_labeled (np.ndarray): Labeled ground truth values.
    Yhat_labeled (np.ndarray): Predictions for the labeled data.
    Yhat_unlabeled (np.ndarray): Predictions for the unlabeled data.
    alpha (float): Significance level for the confidence intervals.
    num_trials (int): Number of trials to run for the inference.

    Returns:
    tuple: A tuple containing the average PPI confidence interval, the average classical confidence interval, and the imputed-only confidence interval.
    """

    return pp_mean_iid_asymptotic(Y_labeled, Yhat_labeled, Yhat_unlabeled, alpha)


def begin(evaluation_datasets: list, checkpoints: list, labels: list) -> pd.DataFrame:
    """
    Begin the evaluation process by printing the evaluation datasets, checkpoints, and labels.
    If a few-shot examples file path is provided, read the file and return the few-shot examples.

    Parameters:
    evaluation_datasets (list): List of evaluation datasets.
    checkpoints (list): List of checkpoints.
    labels (list): List of labels.
    """
    print("--------------------------------------------------------")
    print("Evaluation Sets: " + str(evaluation_datasets))
    print("Checkpoints: " + str(checkpoints))
    print("Labels: " + str(labels))
    print("--------------------------------------------------------")


def preprocess_data(test_set_selection: str, label_column: str, labels: list):
    """
    Preprocesses the data for evaluation.

    Parameters:
    - test_set_selection (str): The file path to the test set selection in CSV format.
    - label_column (str): The column name in the test set that contains the labels.
    - labels (list): A list of labels to be used for filtering the test set.

    Returns:
    - Tuple[pd.DataFrame, str]: A tuple containing the preprocessed test set DataFrame and the name of the text column.

    Raises:
    - ValueError: If the dataset has fewer than 10 rows after filtering.
    """

    # Read the test set from a CSV file
    test_set = pd.read_csv(test_set_selection, sep="\t")

    # Define the text column name
    text_column = "concat_text"

    if label_column in test_set.columns:
        test_set = test_set[test_set[label_column].notna()]

    # Combine query and document (and answer if applicable) into the text column
    if "Context" in label_column:
        test_set[text_column] = [
            combine_query_document(query=test_set.iloc[i]["Query"], document=test_set.iloc[i]["Document"])
            for i in range(len(test_set))
        ]
    elif "Answer_Relevance" in label_column:
        test_set[text_column] = [
            combine_query_document(query=test_set.iloc[i]["Query"], answer=test_set.iloc[i]["Answer"])
            for i in range(len(test_set))
        ]
    else:
        test_set[text_column] = [
            combine_query_document(
                query=test_set.iloc[i]["Query"],
                document=test_set.iloc[i]["Document"],
                answer=test_set.iloc[i]["Answer"],
            )
            for i in range(len(test_set))
        ]

    # Filter out rows where the text column has the value "Error"
    test_set = test_set[test_set[text_column] != "Error"]

    # Check if the dataset has fewer than 10 rows after filtering
    if len(test_set) < 10:
        raise ValueError("Insufficient Data: Dataset has fewer than 10 rows after filtering!")

    return test_set, text_column

    ############################################################


def load_tokenizer_and_model(model_identifier: str, number_of_labels: int, checkpoint: str = None) -> tuple:
    """
    Loads a tokenizer and model based on the provided model identifier and number of labels.

    Parameters:
    - model_identifier (str): The identifier of the model to load.
    - number_of_labels (int): The number of labels for the model.
    - checkpoint (str, optional): The path to a checkpoint file to load the model state from.

    Returns:
    - tuple: A tuple containing the model, tokenizer, and device.

    Raises:
    - FileNotFoundError: If the checkpoint file is not found.
    """
    max_token_length = 512 if "electra" in model_identifier.lower() else 2048
    tokenizer = AutoTokenizer.from_pretrained(model_identifier, model_max_length=max_token_length)
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")

    model = CustomBERTModel(number_of_labels, model_identifier)
    model.to(device)

    if checkpoint:
        checkpoint_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
        model_dict = model.state_dict()

        # Print some information about the checkpoint and model
        print(f"Checkpoint keys: {len(checkpoint_dict)}")
        print(f"Model keys: {len(model_dict)}")

        # Filter out unnecessary keys
        pretrained_dict = {
            k: v for k, v in checkpoint_dict.items() if k in model_dict and v.shape == model_dict[k].shape
        }

        # Print information about matched and unmatched keys
        print(f"Matched keys: {len(pretrained_dict)}")
        print(f"Unmatched keys: {len(checkpoint_dict) - len(pretrained_dict)}")

        if len(pretrained_dict) == 0:
            print("Warning: No keys matched between the checkpoint and the model!")
            print("Checkpoint keys (first 10):", list(checkpoint_dict.keys())[:10])
            print("Model keys (first 10):", list(model_dict.keys())[:10])
            raise ValueError("No matching keys found between checkpoint and model")

        # Update model state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

        print(f"Loaded checkpoint: {checkpoint}")
        print(f"Matched keys: {len(pretrained_dict)}/{len(model_dict)}")
    else:
        print("Loaded model based on model identifier:", model_identifier)

    return model, tokenizer, device

    ############################################################


def evaluate_model(params: dict) -> tuple:
    """
    Evaluates a model based on the provided parameters.

    Parameters:
    - params (dict): A dictionary containing the following keys:
        - test_set (pd.DataFrame): The test dataset.
        - label_column (str): The column name for labels in the test set.
        - text_column (str): The column name for text in the test set.
        - device (str): The device to run the model on (e.g., 'cuda:0').
        - checkpoint (str): The path to a checkpoint file to load the model state from.
        - tokenizer (AutoTokenizer): The tokenizer to use.
        - model (CustomBERTModel): The model to evaluate.
        - assigned_batch_size (int): The batch size for evaluation.
        - model_choice (str): The choice of model.

    Returns:
    - tuple: A tuple containing total_predictions, results, and metric.
    """
    test_set = params["test_set"]
    label_column = params["label_column"]
    text_column = params["text_column"]
    device = params["device"]
    checkpoint = params["checkpoint"]
    tokenizer = params["tokenizer"]
    model = params["model"]
    assigned_batch_size = params["assigned_batch_size"]
    model_choice = params["model_choice"]

    metric = evaluate.load("accuracy")

    if checkpoint:
        total_predictions = torch.FloatTensor([]).to(device)
        total_references = torch.FloatTensor([]).to(device)
        total_logits = torch.FloatTensor([]).to(device)
        eval_dataloader = prepare_dataset_for_evaluation(
            test_set, label_column, text_column, assigned_batch_size, tokenizer
        )
        model.eval()
        with tqdm(eval_dataloader, desc="Evaluating", leave=False) as progress_bar:
            for batch in progress_bar:
                with torch.no_grad():
                    if model_choice in ["mosaicml/mpt-1b-redpajama-200b"]:
                        new_batch = {
                            "ids": batch["input_ids"].to(device),
                            "mask": batch["attention_mask"].bool().to(device),
                        }
                    else:
                        new_batch = {"ids": batch["input_ids"].to(device), "mask": batch["attention_mask"].to(device)}

                    if model_choice in ["t5-small", "google/t5-xl-lm-adapt", "google/t5-large-lm-adapt"]:
                        new_batch["decoder_input_ids"] = (
                            batch["labels"].reshape(batch["labels"].shape[0], 1).to(device)
                        )

                    outputs = model(**new_batch)

                    logits = outputs
                    predictions = torch.argmax(logits, dim=-1)

                    if "labels" in batch:
                        # Add the batch to the metric
                        metric.add_batch(predictions=predictions, references=batch["labels"].to(device))

                        # Concatenate the references for later use
                        total_references = torch.cat((total_references, batch["labels"].to(device)), 0)

                    total_predictions = torch.cat((total_predictions, predictions), 0)
                    total_logits = torch.cat((total_logits, logits), 0)

                    progress_bar.update(1)

    if total_references.nelement() > 0:
        results = metric.compute(references=total_references, predictions=total_predictions)
    else:
        results = None

    return total_predictions, results, metric


def post_process_predictions(params: dict):
    checkpoint = params["checkpoint"]
    test_set = params["test_set"]
    label_column = params["label_column"]
    total_predictions = params["total_predictions"]
    labels = params["labels"]
    gold_label_path = params["gold_label_path"]
    tokenizer = params["tokenizer"]
    assigned_batch_size = params["assigned_batch_size"]
    device = params["device"]
    unlabeled_eval_set = params["test_set"]

    prediction_column = label_column + "_MARS_Predictions"
    test_set[prediction_column] = (
        total_predictions if isinstance(total_predictions, list) else total_predictions.tolist()
    )

    if label_column in test_set.columns:
        test_set = test_set[test_set[label_column].notna()]

    for label in labels:
        if label in test_set.columns:
            if label != label_column:
                test_set = test_set[test_set[label] != 0]

    Y_labeled_dataset = pd.read_csv(gold_label_path, sep="\t")
    Y_labeled_dataset = Y_labeled_dataset[Y_labeled_dataset[label_column].notna()].head(300)

    text_column = "concat_text"
    if "Context" in label_column:
        Y_labeled_dataset[text_column] = [
            combine_query_document(
                query=Y_labeled_dataset.iloc[i]["Query"], document=Y_labeled_dataset.iloc[i]["Document"]
            )
            for i in range(len(Y_labeled_dataset))
        ]
    elif "Answer_Relevance" in label_column:
        Y_labeled_dataset[text_column] = [
            combine_query_document(
                query=Y_labeled_dataset.iloc[i]["Query"], answer=Y_labeled_dataset.iloc[i]["Answer"]
            )
            for i in range(len(Y_labeled_dataset))
        ]
    else:
        Y_labeled_dataset[text_column] = [
            combine_query_document(
                query=Y_labeled_dataset.iloc[i]["Query"],
                document=Y_labeled_dataset.iloc[i]["Document"],
                answer=Y_labeled_dataset.iloc[i]["Answer"],
            )
            for i in range(len(Y_labeled_dataset))
        ]

    Y_labeled_dataset = Y_labeled_dataset[Y_labeled_dataset[text_column] != "Error"]

    if checkpoint:
        Y_labeled_dataloader = prepare_dataset_for_evaluation(
            Y_labeled_dataset, label_column, text_column, assigned_batch_size, tokenizer
        )
    else:
        Y_labeled_dataloader = None

    Y_labeled_predictions = torch.FloatTensor([]).to(device)
    Yhat_unlabeled_dataset = test_set

    return (
        test_set,
        Y_labeled_dataset,
        Y_labeled_dataloader,
        Y_labeled_predictions,
        Yhat_unlabeled_dataset,
        prediction_column,
    )


def evaluate_and_scoring_data(params: dict):
    # Extract parameters
    test_set = params["test_set"]
    Y_labeled_predictions = params["Y_labeled_predictions"]
    Y_labeled_dataset = params["Y_labeled_dataset"]
    Y_labeled_dataloader = params["Y_labeled_dataloader"]
    Yhat_unlabeled_dataset = params["Yhat_unlabeled_dataset"]
    alpha = params["alpha"]
    model = params["model"]
    device = params["device"]
    model_choice = params["model_choice"]
    metric = params["metric"]
    prediction_column = params["prediction_column"]
    label_column = params["label_column"]
    test_set_selection = params["test_set_selection"]
    LLM_judge_ratio_predictions = params["LLM_judge_ratio_predictions"]
    validation_set_lengths = params["validation_set_lengths"]
    validation_set_ratios = params["validation_set_ratios"]
    ppi_confidence_intervals = params["ppi_confidence_intervals"]
    accuracy_scores = params["accuracy_scores"]
    results = params["results"]
    checkpoint = params["checkpoint"]
    prediction_filepath = params["prediction_filepath"]

    failed_extraction_count = {"failed": 0}  # Reset failed extraction count

    if checkpoint:
        model.eval()
        with tqdm(Y_labeled_dataloader, desc="Scoring", leave=False) as progress_bar:
            for batch in progress_bar:
                with torch.no_grad():
                    new_batch = {
                        "ids": batch["input_ids"].to(device),
                        "mask": (
                            batch["attention_mask"].bool().to(device)
                            if model_choice in ["mosaicml/mpt-1b-redpajama-200b"]
                            else batch["attention_mask"].to(device)
                        ),
                    }
                    if model_choice in ["t5-small", "google/t5-xl-lm-adapt", "google/t5-large-lm-adapt"]:
                        new_batch["decoder_input_ids"] = (
                            batch["labels"].reshape(batch["labels"].shape[0], 1).to(device)
                        )

                    outputs = model(**new_batch)

                    logits = outputs
                    predictions = torch.argmax(logits, dim=-1)

                    if "labels" in batch:
                        labels = batch["labels"].to(device)
                        metric.add_batch(predictions=predictions, references=labels)

                    Y_labeled_predictions = torch.cat((Y_labeled_predictions, predictions), 0)
                    progress_bar.update(1)

        Y_labeled_dataset[prediction_column] = Y_labeled_predictions.detach().cpu().numpy().tolist()
        Yhat_unlabeled_dataset = test_set

        # Compute the metric after all batches are processed
        results_metric = metric.compute()
        # results['accuracy'] = results_metric.get('accuracy', None)

    # Convert predictions and labels to integer type
    Y_labeled = Y_labeled_dataset[label_column].values.astype(int)
    Yhat_labeled = Y_labeled_dataset[prediction_column].values.astype(int)
    Yhat_unlabeled = Yhat_unlabeled_dataset[prediction_column].values.astype(int)

    # Debugging: Print a sample of the ground truth and predictions
    print("Sample Ground Truth Labels:", Y_labeled[:10])
    print("Sample LLM Judge Predictions:", Yhat_labeled[:10])

    # Calculate PPI metrics
    avg_ci = calculate_ppi(Y_labeled, Yhat_labeled, Yhat_unlabeled, alpha)

    # Update metrics lists
    LLM_judge_ratio_predictions.append(sum(avg_ci) / len(avg_ci))
    validation_set_lengths.append(len(test_set))
    ppi_confidence_intervals.append([round(value, 3) for value in avg_ci])

    # Compute Ground Truth Performance
    ground_truth_available = False
    if label_column in Yhat_unlabeled_dataset.columns and not Yhat_unlabeled_dataset[label_column].isnull().all():
        ground_truth_performance = round(
            Yhat_unlabeled_dataset[label_column].tolist().count(1) / len(Yhat_unlabeled_dataset), 3
        )
        validation_set_ratios.append(ground_truth_performance)
        ground_truth_available = True
        # Calculate accuracy separately
        accuracy = (
            Yhat_unlabeled_dataset[label_column].values.astype(int)
            == Yhat_unlabeled_dataset[prediction_column].values.astype(int)
        ).mean()
        accuracy_scores.append(round(accuracy, 3) if accuracy is not None else None)
    else:
        accuracy_scores.append(None)
        validation_set_ratios.append(None)

    pre_ppi_score = round(Yhat_unlabeled_dataset[prediction_column].tolist().count(1) / len(Yhat_unlabeled_dataset), 3)

    # Build the results dictionary
    results = {
        "Label_Column": label_column,
        "Evaluation_Set": test_set_selection,
        "ARES_Prediction": LLM_judge_ratio_predictions[-1] if LLM_judge_ratio_predictions else None,
        "ARES_Confidence_Interval": ppi_confidence_intervals[-1] if ppi_confidence_intervals else None,
        "Number_of_Examples_in_Evaluation_Set": validation_set_lengths[-1] if validation_set_lengths else None,
        "Ground_Truth_Performance": validation_set_ratios[-1],
        "ARES_LLM_Judge_Accuracy_on_Ground_Truth_Labels": accuracy_scores[-1],
        "Annotated_Examples_used_for_PPI": len(Y_labeled),
        "Pre_PPI_Score": pre_ppi_score,
    }

    # Save the labeled dataset with predictions to a new TSV file
    if prediction_filepath != "None":
        # Update the prediction column name based on the label column
        prediction_column_mapping = {
            "Context_Relevance_Label": "ARES_Context_Relevance_Prediction",
            "Answer_Relevance_Label": "ARES_Answer_Relevance_Prediction",
            "Answer_Faithfulness_Label": "ARES_Answer_Faithfulness_Prediction",
            "Language_Consistency_Label": "ARES_Language_Consistency_Prediction",
        }
        prediction_column_name = prediction_column_mapping.get(label_column, prediction_column)

        Yhat_unlabeled_dataset.rename(columns={prediction_column: prediction_column_name}, inplace=True)
        if os.path.exists(prediction_filepath):
            # TODO better done with join
            existing_predictions = pd.read_csv(prediction_filepath, sep="\t")
            if ground_truth_available:
                existing_predictions = pd.concat([existing_predictions, Yhat_unlabeled_dataset], ignore_index=True)
                # remove duplicates
                existing_predictions = existing_predictions.drop_duplicates(
                    subset=["Query", "Document", "Answer", "doc_lang", "qa_lang"]
                )
            else:
                existing_predictions[prediction_column_name] = Yhat_unlabeled_dataset[prediction_column_name]
        else:
            existing_predictions = Yhat_unlabeled_dataset

        existing_predictions.to_csv(prediction_filepath, sep="\t", index=False)
        print("--------------------------------------------------")
        print(f"Labeled dataset with predictions saved to {prediction_filepath}")
        print("--------------------------------------------------")

    # Print the results
    print("--------------------------------------------------")
    print(f"{label_column} Scoring")
    print("ARES Ranking")
    print(f"Evaluation_Set: {test_set_selection}")
    print(f"Checkpoint: {checkpoint}")
    print(f"ARES Prediction: {LLM_judge_ratio_predictions[-1] if LLM_judge_ratio_predictions else None}")
    print(f"ARES Confidence Interval: {ppi_confidence_intervals[-1] if ppi_confidence_intervals else None}")
    print(f"Number of Examples in Evaluation Set: {validation_set_lengths[-1] if validation_set_lengths else None}")
    if ground_truth_available:
        print(f"Ground Truth Performance: {validation_set_ratios[-1]}")
    if accuracy_scores[-1] is not None:
        print(f"ARES LLM Judge Accuracy on Ground Truth Labels: {accuracy_scores[-1]}")
    print(f"Annotated Examples used for PPI: {len(Y_labeled)}")
    print(f"Pre-PPI Score: {pre_ppi_score}")
    print("--------------------------------------------------\n")

    return results
