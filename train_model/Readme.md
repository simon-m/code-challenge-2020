# Train Model task

This script will train (fit) a model on the input dataset and serializes the transformer pipeline (feature generator) and the estimator (regressor) separately.
Model selection can be performed instead of simple fitting. In this case, a table containing the results is output in file cv_results.csv

```
Usage: python train_model.py [OPTIONS]

  Trains a model on the input and serialized the results.

  Parameters
  ----------
    dataset: path to the dataset used for training the model.
    out_dir: path to the directory where the results should be saved to.
    n_iter: number of model selection steps
    select_model: "True" to run model selection, "False" if not. Errors otherwise.

  Returns
  ------- 
  None

Options:
  --dataset TEXT
  --out-dir TEXT
  --n-iter NUMBER
  --select-model "True" or "False"
  --help          Show this message and exit.
```
