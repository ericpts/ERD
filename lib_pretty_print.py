def pretty_print_method(method_name: str) -> str:
    return {
        "binary_classifier": "Bin. Clasif.",
        "dpn": "DPN",
        "knn": "kNN",
        "mahalanobis": "Mahal.",
        "mahalanobis_transductive": "Mahal-T",
        "mcd": "MCD",
        "oe": "OE",
        "oe_tune_on_high_severity": "OE (trained on sev 5)",
        "oe_tune_on_low_severity": "OE (trained on sev 2)",
        "reto_finetune": "RETO (pretrained)",
        "reto_from_scratch": "RETO (rand init)",
        "reto_holdout": "RETO (holdout)",
        "vanilla": "Vanilla Ensembles",
    }[method_name]


def pretty_print_dataset(dataset: str) -> str:
    if "corrupted" in dataset:
        basename = dataset.split("_", 1)[0]
        corr_sev = dataset.rsplit("/", 1)[1].replace("_", " ")
        return "{}-C {}".format(basename.upper(), corr_sev)
    return {
        "mnist": "MNIST",
        "mnist:0,1,2,3,4": "MNIST[0:4]",
        "mnist:5,6,7,8,9": "MNIST[5:9]",
        "fashion_mnist": "FashionMNIST",
        "fashion_mnist:0,2,3,7,8": "FashionMNIST[0,2,3,7,8]",
        "fashion_mnist:1,4,5,6,9": "FashionMNIST[1,4,5,6,9]",
        "cifar10": "CIFAR10",
        "cifar100": "CIFAR100",
        "svhn_cropped": "SVHN",
        "cifar10_1": "CIFAR10v2",
        "cifar10:0,1,2,3,4": "CIFAR10[0:4]",
        "cifar10:5,6,7,8,9": "CIFAR10[5:9]",
        "svhn_cropped:0,1,2,3,4": "SVHN[0:4]",
        "svhn_cropped:5,6,7,8,9": "SVHN[5:9]",
        "cifar100:0-50": "CIFAR100[0:49]",
        "cifar100:50-100": "CIFAR100[50:99]",
        "imagenet_resized/32x32": "Tiny ImageNet",
        "objectnet/32x32": "Tiny ObjectNet",
    }[dataset]
