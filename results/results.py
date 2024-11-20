import json
import os
import pandas as pd
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import tikzplotlib

path = "results"

accuracy_table = pd.DataFrame(columns=["Experiment", "Context Relevance", "Answer Relevance", "Answer Faithfulness", "Language Consistency"])
accuracy_table.set_index("Experiment", inplace=True)

tau_table = accuracy_table.copy()

for filename in sorted(os.listdir(path)):
    if filename.endswith(".json"):
        experiment_name = filename.replace("results_", "").replace(".json", "").replace("_", " ")
        file_path = os.path.join(path, filename)

        accuracy_table.loc[experiment_name] = [0.0, 0.0, 0.0, 0.0]
        tau_table.loc[experiment_name] = [0.0, 0.0, 0.0, 0.0]

        with open(file_path, 'r') as file:
            results = json.load(file)

        for experiment in results:
            accuracies = []
            scores = []
            truths = []
            confidences = []
            for split in experiment:
                accuracies.append(split["ARES_LLM_Judge_Accuracy_on_Ground_Truth_Labels"])
                scores.append(split["ARES_Prediction"])
                truths.append(split["Ground_Truth_Performance"])
                confidences.append(split["ARES_Confidence_Interval"])
            tau, p_value = kendalltau(scores, truths)
            accuracy = sum(accuracies) / len(accuracies)
            print("Metric: ", experiment[0]["Label_Column"])
            print("Average accuracy: ", accuracy)
            print("Kendall's tau: ", tau)

            # Transpose confidences
            confidences = list(map(list, zip(*confidences)))
            confidences = [[abs(a - b) for a, b in list(zip(x, scores))] for x in confidences]
            print("Confidence intervals: ", confidences)

            label = experiment[0]["Label_Column"].replace("_", " ").replace(" Label", "")

            plt.figure(figsize=(10, 6))

            # Plot scores vs truths
            plt.errorbar(truths, scores, yerr=confidences, fmt='o', ecolor='r', capthick=2, label='Scores with Confidence Intervals')
            plt.scatter(truths, truths, color='g', marker='o', label='Ground Truth Performance')
            plt.xlabel('Ground Truth Performance')
            plt.ylabel('ARES Prediction')
            plt.title(f'{experiment_name} - {label}')
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(True)
            # plt.show()

            plot_filename = f"plots/results_{experiment_name}_{experiment[0]['Label_Column']}.tex"
            tikzplotlib.save(plot_filename)
            print(f"Saved plot to {plot_filename}")

            accuracy_table.loc[experiment_name, label] = accuracy
            tau_table.loc[experiment_name, label] = tau

print(accuracy_table.to_latex(float_format="{:.1%}".format))
print(tau_table.to_latex(float_format="%.2f"))