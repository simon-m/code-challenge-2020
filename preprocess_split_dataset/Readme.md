# Preprocess Split Dataset task

This script will preprocesses a csv dataset and a randomly split it in a training and a test set.
It will then write the resulting datasets in separate files.

```
Usage: python preprocess_split_dataset.py [OPTIONS]

  Preprocesses a csv dataset and randomly splits it in a training and a test set.

  Parameters
  ----------
    in_csv: path to the csv file to be processed.
    out_dir: path to the directory where the results should be saved to.
    test_size: test set size relative to the full dataset


  Returns
  ------- 
  None

Options:
  --in-csv TEXT
  --out-dir TEXT
  --test-size NUMBER (between 0 and 1 excluded)
  --help          Show this message and exit.
```
