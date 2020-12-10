# hard-ood

## Overview of important files
`setup.sh`: run to install dependencies for the project on a new machine.

`main.py`: contains logic for data loading, setting the hyperparameter configuration and training.

`eval.py`: obtains and stores the predicted posterior values for a model.

`eval_ensembles.py`: loads the stored posteriors for the models in an ensemble and computes ensemble metrics e.g. AUROC, FPR@80 etc
