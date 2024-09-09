import torch
from torch.utils.data import DataLoader

from ares.LLM_as_a_Judge_Adaptation.General_Binary_Classifier import CustomBERTModel, analyze_and_report_data, evaluate_model, initalize_dataset_for_tokenization, load_model, prepare_dataset, print_and_save_model, split_dataset, transform_data

device = torch.device("cuda:0")
model_choice = "microsoft/mdeberta-v3-base"
checkpoint_path = "checkpoints/microsoft-mdeberta-v3-base/Context_Relevance_Label_mlqa_dev_ratio_0.7_2024-09-09_09:47:37.pt"

model = CustomBERTModel(2, model_choice)
model.to(device)
model.eval()

inference_times = []
assigned_batch_size = 1

training_dataset_path = "multilingual_data/synthetic_queries_mlqa.tsv"
label = "Context_Relevance_Label"
validation_dataset_path = "multilingual_data/mlqa_dev_ratio_0.7.tsv"
validation_set_scoring = True

tokenizer, max_token_length = load_model(model_choice)


synth_queries = analyze_and_report_data(training_dataset_path, label, tokenizer, max_token_length)

train_df, test_set = transform_data(synth_queries, validation_dataset_path, label)

train_set_text, train_set_label, dev_set_text, dev_set_label, test_set_text, text_set_label_, labels_list = split_dataset(train_df, training_dataset_path, test_set, label)

training_dataset_pandas, training_dataset_arrow, validation_dataset_arrow, test_dataset_arrow, test_dataset_pandas = prepare_dataset(validation_set_scoring, train_set_label, train_set_text, dev_set_label, dev_set_text)

tokenized_datasets = initalize_dataset_for_tokenization(tokenizer, training_dataset_arrow, validation_dataset_arrow, test_dataset_arrow)

eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)

total_predictions, total_references, metric = evaluate_model(model, model_choice, checkpoint_path, device, eval_dataloader, inference_times)

print_and_save_model(total_predictions, total_references, checkpoint_path, metric)