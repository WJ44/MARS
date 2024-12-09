import os

import torch

from .RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import (
    begin,
    evaluate_and_scoring_data,
    evaluate_model,
    load_tokenizer_and_model,
    post_process_predictions,
    preprocess_data,
)


def rag_scoring_config(
    alpha,
    evaluation_datasets,
    checkpoints,
    labels,
    model_choice,
    assigned_batch_size,
    number_of_labels,
    gold_label_paths,
    prediction_filepaths,
):
    """
    Configures and runs the RAG scoring process.

    Parameters:
    - alpha: The alpha value for the scoring process.
    - evaluation_datasets: List of datasets to evaluate.
    - checkpoints: List of model checkpoints.
    - labels: List of labels.
    - model_choice: Choice of model.
    - assigned_batch_size: Batch size to use.
    - number_of_labels: Number of labels.
    - gold_label_paths: List of paths to the gold labels.
    - prediction_filepaths: List of file paths to save predictions.
    """

    # Validate if either gold_label_paths or gold_machine_label_path is provided
    if gold_label_paths == ["None"]:
        raise ValueError("'gold_label_paths'' must be provided.")

    # Validate inputs and determine model loading strategy
    if checkpoints:
        model_loader = lambda chk: load_tokenizer_and_model(model_choice, number_of_labels, chk)
    else:
        raise ValueError("No valid model or checkpoint provided.")

    # Use zip only if checkpoints are not empty, otherwise assume only llm_judge is used
    if checkpoints:
        # Here we assume that the length of checkpoints and labels is the same
        pairings = zip(checkpoints, labels)

    all_evaluation_results = []

    # remove prediction filepath files
    for prediction_filepath in prediction_filepaths:
        if os.path.exists(prediction_filepath):
            os.remove(prediction_filepath)

    for _, (checkpoint, label_column) in enumerate(pairings):

        chekpoint_results = []

        LLM_judge_ratio_predictions = []
        validation_set_lengths = []
        validation_set_ratios = []
        ppi_confidence_intervals = []
        accuracy_scores = []
        for test_set_idx, test_set_selection in enumerate(evaluation_datasets):

            begin(evaluation_datasets, checkpoints, labels)

            test_set, text_column = preprocess_data(test_set_selection, label_column, labels)

            loaded_model = model_loader(checkpoint)
            if isinstance(loaded_model, tuple):
                model, tokenizer, device = loaded_model
            else:
                model = loaded_model
                tokenizer = None
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            eval_model_settings = {
                "test_set": test_set,
                "label_column": label_column,
                "text_column": text_column,
                "device": device,
                "checkpoint": checkpoint,
                "tokenizer": tokenizer,
                "model": model,
                "assigned_batch_size": assigned_batch_size,
                "model_choice": model_choice,
            }

            total_predictions, results, metric = evaluate_model(eval_model_settings)

            post_process_settings = {
                "checkpoint": checkpoint,
                "test_set": test_set,
                "label_column": label_column,
                "total_predictions": total_predictions,
                "labels": labels,
                "gold_label_path": (
                    gold_label_paths[test_set_idx] if test_set_idx < len(gold_label_paths) else gold_label_paths[-1]
                ),
                "tokenizer": tokenizer,
                "assigned_batch_size": assigned_batch_size,
                "device": device,
            }

            (
                test_set,
                Y_labeled_dataset,
                Y_labeled_dataloader,
                Y_labeled_predictions,
                Yhat_unlabeled_dataset,
                prediction_column,
            ) = post_process_predictions(post_process_settings)

            evaluate_scoring_settings = {
                "test_set": test_set,
                "Y_labeled_predictions": Y_labeled_predictions,
                "Y_labeled_dataset": Y_labeled_dataset,
                "Y_labeled_dataloader": Y_labeled_dataloader,
                "Yhat_unlabeled_dataset": Yhat_unlabeled_dataset,
                "alpha": alpha,
                "model": model,
                "device": device,
                "model_choice": model_choice,
                "metric": metric,
                "prediction_column": prediction_column,
                "label_column": label_column,
                "test_set_selection": test_set_selection,
                "LLM_judge_ratio_predictions": LLM_judge_ratio_predictions,
                "validation_set_lengths": validation_set_lengths,
                "validation_set_ratios": validation_set_ratios,
                "ppi_confidence_intervals": ppi_confidence_intervals,
                "accuracy_scores": accuracy_scores,
                "results": results,
                "checkpoint": checkpoint,
                "prediction_filepath": (
                    prediction_filepaths[test_set_idx]
                    if test_set_idx < len(prediction_filepaths)
                    else prediction_filepaths[-1]
                ),
            }
            dataset_results = evaluate_and_scoring_data(evaluate_scoring_settings)
            chekpoint_results.append(dataset_results)

        all_evaluation_results.append(chekpoint_results)

    return all_evaluation_results
