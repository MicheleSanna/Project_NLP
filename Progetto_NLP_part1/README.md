*classification_run_bert*
Contains the model's parameter saves from the run of 5 epochs for the classification task's fine tuning of bert and its tensorboard logs.
# classification_run_bert
Contains the model's parameter saves from the run of 5 epochs for the classification task's fine tuning of gpt-2 and its tensorboard logs.
# regression_scripts
Contains files concerning a test with a regression task. The results of this expirement are not included in the report
# runs
Tensorboard logs
# Successful runs
ORFEO logs to test the distributed training
# JsonDatasets.py
Classes used to access in an efficient way the dataset (that is indeed in a json extension)
# conf.py
Configuration variables
# models.py
File containing custom models. Those models were used for tests, not for projects's experiment (for wich i used straight HuggingFace's models)
# test.py
Functions used for testing and debugging
# trainer.py
The script contaning the class that manages the training process
# train.py
The script for launching the training.
# gpt_cosine.out
ORFEO log for the 1 epoch training experiment with gpt (cosine stands for "cosine annealing with warm restarts')
# bert_cosine.out
ORFEO log for the 1 epoch training experiment with bert (cosine stands for "cosine annealing with warm restarts')
