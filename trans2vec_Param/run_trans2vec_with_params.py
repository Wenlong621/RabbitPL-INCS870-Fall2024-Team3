import matplotlib.pyplot as plt
from copy import deepcopy
from trans2vec_params import run_trans2vec
import argparse

#  Convert dictionary to Namespace
def dict_to_namespace(param_dict):
    return argparse.Namespace(**param_dict)

# Assuming run_trans2vec_with_params is your defined function
def run_trans2vec_with_params(params, times):
    # Convert parameter dictionary to Namespace
    args = dict_to_namespace(params)
    
    # Call run_trans2vec from trans2vec.py
    results = run_trans2vec(args, times)
    
    return results

# origin params
original_params = {
    "seed": 2022,
    "dimensions": 64,
    "num_walks": 5,
    "walk_length": 2,
    "window_size": 2,
    "workers": 1,
    "train_size": 0.8,
    "alpha": 0.5
}

# Parameter ranges (define parameter variation ranges based on your suggestions)
param_ranges = {
    "dimensions": [32, 128, 256],
    "num_walks": [3, 10, 15],
    "walk_length": [1, 5, 7],
    "window_size": [1, 5, 10],
    "workers": [2, 4],
    "train_size": [0.7, 0.9],
    "alpha": [0.1, 0.2, 0.3, 0.8]
}

# Experiment and plot for each parameter and its possible values
for param_name, values in param_ranges.items():
    f1_scores = []
    precision_scores = []
    recall_scores = []
    experiment_labels = []  # Save labels for each experiment (i.e., parameter values)

    for value in values:
        new_params = deepcopy(original_params)  # Create a copy of the parameters
        new_params[param_name] = value  # Modify the current parameter
        print(f"Changing {param_name} to {value}")
        
        # Run the experiment and save results
        results = run_trans2vec_with_params(new_params, 10)
        f1_scores.append(results[2])  # The third value is the F1-score
        precision_scores.append(results[0])  # The first value is Precision
        recall_scores.append(results[1])  # The second value is Recall
        experiment_labels.append(f"{param_name}={value}")
       

    # Plot the graph for this parameter
    plt.figure(figsize=(10, 6))

    # Plot F1-score
    plt.plot(experiment_labels, f1_scores, label='F1 Score', marker='o', color='b', linestyle='-')  # Adjusted F1 Score, solid line
    plt.axhline(results[2], color='b', linestyle='--', label='Original F1 Score')  # Original F1 Score, dashed line


    # Plot Precision
    plt.plot(experiment_labels, precision_scores, label='Precision', marker='o', color='orange', linestyle='-')  # Adjusted Precision, solid line
    plt.axhline(results[0], color='orange', linestyle='--', label='Original Precision')  # Original Precision, dashed line

    # Plot Recall
    plt.plot(experiment_labels, recall_scores, label='Recall', marker='o', color='g', linestyle='-')  # Adjusted Recall, solid line
    plt.axhline(results[1], color='g', linestyle='--', label='Original Recall')  # Original Recall, dashed line


    plt.xlabel(f'{param_name} Values')
    plt.ylabel('Score')
    plt.title(f'Comparison of {param_name} Changes vs. Original')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate X-axis labels to avoid overlap
    plt.tight_layout()  # Automatically adjust layout to prevent labels from being cut off
    plt.show()

