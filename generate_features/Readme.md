# Generate Features task

This script will apply a fitted serialized transformer or a pipeline thereof to a dataset.
It will then split the resulting dataset in features (X) and outcome (y) and write them to the output directory.
Extra information about the generated features will also be written out.
Namely, a dictionary containing feature names to be used for model interpretation.
Topic words are also included for text variables procesed with a decomposition method (NMF, LDA).

```
Usage: python generate_features.py [OPTIONS]

  Applies a fitted serialized transformer to a dataset and splits the result in X and y.

  Parameters
  ----------
    dataset: path to the parquet file containing data to be processed.
    feature_generator: path to the the joblib file containing a serialized processor (e.g. a pipeline or a transformer).
    out_dir: path to the directory where the results should be saved to.
    feature info: "True" if additional feature info should be written, "False" otherwise.


  Returns
  ------- 
  None

Options:
  --dataset TEXT
  --feature-generator TEXT
  --out-dir TEXT
  --feature-info TEXT
  --help          Show this message and exit.
```
