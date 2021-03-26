# This is a short description of the submission.

## Results
The rendered report is report.html
There are two logs: one with and one without model selection. See details below.

The comments in the report may no longer be valid after the pipeline has been rerun because of the random nature 
of splitting the dataset, selecting the features etc. They should not be too far off however.

## Running the submission
There are two ways to run the pipeline. As submitted, it will fit a predictor with pre-tuned parameters.
Changing the select_model parameter in orchestrator/task.py will allow to run model selection.

I used your project base so you must already know the commands to use, but for the sake of completeness, here they are:
```
$ ./build-task-images.sh
$ docker-compose up orchestrator
```

## Results structure
In execution order:
- download_data writes the raw data in data_root/data/raw
- preprocess_split_dataset writes the preprocessed train and test sets in data_root/data/intermediate
- train_model writes the serialized pipelines and model selection information in data_root/models
- generate_features write the features and outcome in data_root/data/processed
- report_evaluation write the report in data_root/results

## Other remarks
Disclaimer: I knowingly put some code that had to be shared between containers in data_root/utils.
That is because I could not get docker-compose to mount another shared directory in time.

Many improvements are possible:
- running model and feature selection for a much larger and finer grid
- repeated cross-validation for model selection
- using other predictors (e.g. XGBoost) or do some model stacking
- using LDA or LSI in place of NMF. Or Word2Vec and the likes.

