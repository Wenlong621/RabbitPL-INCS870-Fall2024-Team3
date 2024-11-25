# Trans2Vec Parameter Tuning and Analysis

This project includes implementations for generating parameter tuning results and comparing the performance of the model with original parameters versus optimal parameters. Follow the steps below to understand the process.

## Files Overview

- **`trans2vec.py`**: The core implementation of the Trans2Vec model, used to generate embeddings and classification results.
- **`run_trans2vec_with_params.py`**: A script for running experiments by varying parameters over a defined range and visualizing the impact on model performance.
- **`trans2vec_params.py`**: Provides helper functions for running Trans2Vec with specified parameters and ranges.
- **`utils.py`**: Includes helper functions for data processing.
- **`walker.py`**: Implements the BiasedRandomWalker for generating biased random walks.


## Steps to Execute

**Note:** All the following commands must be executed from the `trans2vec_Param` directory.

### 1. Parameter Tuning with Visualization
This step explores the impact of varying model parameters on performance metrics (Precision, Recall, F1-score). The parameter ranges are:
- **dimensions**: `[32, 128, 256]`
- **num_walks**: `[3, 10, 15]`
- **walk_length**: `[1, 5, 7]`
- **window_size**: `[1, 5, 10]`
- **workers**: `[2, 4]`
- **train_size**: `[0.7, 0.9]`
- **alpha**: `[0.1, 0.2, 0.3, 0.8]`

#### Command to Run:
```bash
cd trans2vec_Param
python run_trans2vec_with_params.py

```
#### What It Does
-Iterates over the defined parameter ranges.
- Generates visualizations (graphs) for each parameter showing the impact of tuning on Precision, Recall, and F1-score.
- **Note: You need to close each graph for the script to proceed to the next.**

### 2. Running with Optimal Parameters
After determining the best parameters through run_trans2vec_with_params.py, execute the model with the optimal parameters and compare results with the original parameter setup.

#### Optimal Parameters:
```json
    {
        "seed": 2022,
        "dimensions": 128,
        "num_walks": 10,
        "walk_length": 5,
        "window_size": 5,
        "workers": 4,
        "train_size": 0.9,
        "alpha": 0.3
    }
```
#### Commands:
    
- Original Parameters: 
```bash
    python trans2vec.py
```
This uses the default parameters as defined in the script.
- Optimal Parameters:
```bash
    python trans2vec.py --seed 2022 --dimensions 128 --num_walks 10 --walk_length 5 --window_size 5 --workers 4 --train_size 0.9 --alpha 0.3
```
#### What It Does:
- Compares the performance of the model using the original parameters versus the optimal parameters.
- Displays the results of Precision, Recall, and F1-score in the terminal.

### Output
#### Parameter Tuning Results
- Visualizations for each parameter showing how performance metrics change.
#### Comparison Results
- Classification reports for both the original and optimal parameter setups.

### Notes 
- Ensure that the dataset is placed correctly in the dataset directory.
- All necessary dependencies should be installed, including gensim, scikit-learn, and matplotlib.