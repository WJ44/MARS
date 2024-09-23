import json
from scipy.stats import kendalltau

file_path = "results.json"

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
    print("Metric: ", experiment[0]["Label_Column"])
    print("Average accuracy: ", sum(accuracies) / len(accuracies))
    print("Kendall's tau: ", tau)

    # Transpose confidences
    confidences = list(map(list, zip(*confidences)))
    confidences = [[abs(a - b) for a,b in list(zip(x, scores))] for x in confidences]
    print("Confidence intervals: ", confidences)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    # Plot scores vs truths
    plt.errorbar(truths, scores, yerr=confidences, fmt='o', ecolor='r', capthick=2, label='Scores with Confidence Intervals')
    plt.scatter(truths, truths, color='g', marker='o', label='Ground Truth Performance')
    plt.xlabel('Ground Truth Performance')
    plt.ylabel('ARES Prediction')
    plt.title('ARES Prediction vs Ground Truth Performance')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()
