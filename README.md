# hard-ood

## Overview of important files
`setup.sh`: run to install dependencies for the project on a new machine.

`main.py`: contains logic for data loading, setting the hyperparameter configuration and training.

`eval.py`: obtains and stores the predicted posterior values for a model.

`eval_ensembles.py`: loads the stored posteriors for the models in an ensemble and computes ensemble metrics e.g. AUROC, FPR@80 etc

## Running RETO

### Environment setup

We'll set up a root folder for storing RETO experiment data, for example `~/reto/`.

Firt we need to export some environment variables:

```bash
export RANDOM_LABELS_PROJECT_ROOT=~/reto/
export RANDOM_LABELS_LOGS_DIR=~/reto/logs
export RANDOM_LABELS_PRETRAINED_MODELS_ROOT=~/reto/pretrained_models
```

### Training

We make use of the [gin](https://github.com/google/gin-config) library,
and that is how options are passed.

The following is a sample command for training RETO:

```bash
python3 main.py \
    --root $RANDOM_LABELS_PROJECT_ROOT \
    --ensemble_type assign_one_label \
    --ensemble_size 5 \
    --gin_file configs/resnet.gin \
    --gin_param data.source_dataset="cifar100:0-50" \
    --gin_param data.ood_dataset="cifar100:50-100" \
    --gin_param data.target_size=20000 \
    --gin_param data.target_ood_ratio=0.5 \
    --gin_param train.epochs=100 \
    --gin_param train.use_pretrained_model=False \
    --gin_param train.save_best_ckpt_only=True
```

We use [mlflow](https://mlflow.org/) for logging run information.

### Evaluation

After training is done, you need run evaluations on the saved models.
The simplest way is to get the experiment directory for RETO (from mlflow see
[here](https://mlflow.org/docs/latest/quickstart.html) for a quick setup guide, or
just by looking in the `~/reto/` directory).
And then run `eval_ensembles.py` with that as the argument:

```bash
python3 eval_ensembles.py \
    --exp_dir ~/reto/cifar10_vs_cifar10_corrupted_pixelate_2_wideresnet_100ep_l2reg0.0005_depth28_lr0.1/assign_one_label/fa32685210274b78a7a6e6a29b8e05a6
```

Afterwards you should see the new metrics uploaded to mlflow. For the RETO
statistics look for the metrics prefixed with `heur`; for example,
`heur_auroc_avg_diff` for the AUROC based on the total variational distance.

## Adding a dataset

In the file `utils.py`, modify the function `preprocess_dataset`. At the start,
add a special case for your particular dataset.
Data needs to be a `tf.data.Dataset`, containing the images and the labels.


## Adding a new model

Create a new file in `models/`, for example `models/my_own_net.py`. Look in the
other files in that directory for examples.
Then create a file under `configs/` with the specific configuration for your
model: `configs/my_own_net.gin`, look at the other files for examples.
The most important parameters are
```
train.model_arch = "my_own_net"
train.lr = 0.1
train.l2_reg = 5e-4
train.nn_depth = 100
```

Finally, in `main.py` in the `train()` function, add an if branch with your
particular model:

```python
elif model_arch == 'my_own_net':
    model = my_own_net(
        input_shape=image_shape,
        depth=nn_depth,
        num_classes=data.num_classes,
        l2_reg=l2_reg,
    )
```
