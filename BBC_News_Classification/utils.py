import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import FastText
from sklearn.metrics import confusion_matrix

import torch.nn as nn
from model import Classificator
from trainer import Trainer

fasttext = FastText.load_fasttext_format("drive/My Drive/BBC/cc.en.300.bin")


def plot_confusion_matrix(true, pred, encoder, classes):
    true = encoder.inverse_transform(true)
    pred = encoder.inverse_transform(pred)
    matrix = confusion_matrix(true, pred, labels=classes)

    plt.figure(figsize=(8, 8))
    sns.heatmap(
        matrix,
        yticklabels=classes,
        xticklabels=classes,
        annot=True,
        cmap="YlGnBu",
        cbar=False,
    )


nets = Trainer(
    model=Classificator(hidden_size=100, n_classes=5),
    loss_fn=nn.CrossEntropyLoss(),
    optim_name="Adam",
    optim_param={"lr": 1e-3},
    n_classes=5,
    verbose=True,
)
