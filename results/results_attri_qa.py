import json
import os
import pandas as pd
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import tikzplotlib

path = "results"

langs = []
pairs = []

accuracies = []
scores = []
truths = []
confidences = []
scores_pre_ppi = []

for filename in sorted(os.listdir(path)):
    if filename.endswith(".json") and "attri_qa" in filename:
        experiment_name = filename.replace("results_", "").replace(".json", "").replace("_", " ")
        lang = experiment_name.split(" ")[2][1:3]
        pair = experiment_name.split(" ")[3]

        langs.append(lang)
        pairs.append(pair)

        file_path = os.path.join(path, filename)

        with open(file_path, 'r') as file:
            results = json.load(file)

        for experiment in results:
            accuracy = experiment[0]["ARES_LLM_Judge_Accuracy_on_Ground_Truth_Labels"]
            score = experiment[0]["ARES_Prediction"]
            truth = experiment[0]["Ground_Truth_Performance"]
            confidence = experiment[0]["ARES_Confidence_Interval"]
            score_pre_ppi = experiment[0]["Pre_PPI_Score"]

            accuracies.append(accuracy)
            scores.append(score)
            truths.append(truth)
            confidences.append(confidence)
            scores_pre_ppi.append(score_pre_ppi)

index = pd.MultiIndex.from_tuples(zip(langs, pairs), names=["Language", "Pair"])
df = pd.DataFrame({"Accuracy": accuracies, "Ground truth": truths, "Pre-PPI score": scores_pre_ppi, "MARS Score": scores, "Confidence interval": confidences}, index=index)

print(df.to_latex(formatters={
    "Accuracy": lambda x: "{:.1f}\\%".format(x * 100), 
    "Ground truth": "{:.2f}".format, 
    "Pre-PPI score": "{:.2f}".format, 
    "MARS Score": "{:.2f}".format, 
    "Confidence interval": lambda x: f"[{x[0]:.2f}, {x[1]:.2f}]"
}))