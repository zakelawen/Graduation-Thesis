import numpy as np
from sklearn.metrics import precision_recall_curve

def f1_picker(correctness: list, truth_values: list):
    truth_values = np.array(truth_values)
    truth_values = np.nan_to_num(truth_values)
    (precisions, recalls, thresholds) = precision_recall_curve(correctness, truth_values)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    max_index = np.argmax(f1_scores)
    threshold = thresholds[max_index - 1] if max_index > 0 else thresholds[0]
    return threshold

def precision_picker(correctness: list, truth_values: list, precision: float):
    (precisions, recalls, thresholds) = precision_recall_curve(correctness, truth_values)
    index = np.where(precisions >= precision)[0][0]
    threshold = thresholds[index - 1] if index > 0 else thresholds[0]
    return threshold

def recall_picker(correctness: list, truth_values: list, recall: float):
    (precisions, recalls, thresholds) = precision_recall_curve(correctness, truth_values)
    index = np.where(recalls >= recall)[0][0]
    threshold = thresholds[index - 1] if index > 0 else thresholds[0]
    return threshold

def accuracy_picker(correctness: list, truth_values: list):
    accuracies = [np.mean((truth_values > threshold) == correctness) for threshold in np.unique(truth_values)]
    threshold = np.unique(truth_values)[np.argmax(accuracies)]
    return threshold
