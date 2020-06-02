import re

import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader, Dataset

from utils import fasttext


def preprocess(path, fasttext):
    df = pd.read_csv(path)
    df["Label"], _ = encode_label(df)
    idx_train, idx_test = split_data(df)
    df["Text"] = df["Text"].apply(lambda text: clean_data(text))
    df["Text"] = df["Text"].apply(lambda text: nltk.word_tokenize(text))
    df["Text"] = df["Text"].apply(lambda token: fasttext.wv[token])
    df["Text"] = df["Text"].apply(lambda embed: torch.Tensor(embed))

    embeddings = {
        "train": df["Text"][idx_train].values,
        "test": df["Text"][idx_test].values,
    }
    labels = {
        "train": df["Label"][idx_train].values,
        "test": df["Label"][idx_test].values,
    }
    return embeddings, labels


def get_classes(path):
    df = pd.read_csv(path)
    classes = df["Category"].unique()
    n_classes = len(classes)

    return classes, n_classes


def encode_label(df):
    encoder = LabelEncoder()
    encoder.fit(df["Category"])
    label = encoder.transform(df["Category"])

    return label, encoder


def split_data(df):
    idx_train, idx_test = train_test_split(
        df.index, stratify=df["Label"], test_size=0.2
    )

    return idx_train, idx_test


def clean_data(text):
    text = text.lower()
    text = re.sub("\w*\d\w*", "", text)
    text = re.sub("[^a-z ]", "", text)
    text = re.sub(r"\b\w{3,5}\b", "", text)
    text = re.sub(" +", " ", text)

    return text


embeddings, labels = preprocess("drive/My Drive/BBC/train.csv", fasttext)


class BBC(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, sample):
        embedding = self.embeddings[sample]
        label = self.labels[sample]

        return embedding, label


def pack_embed(batch):
    data = [item[0] for item in batch]
    data = pack_sequence(data, enforce_sorted=False)
    targets = [item[1] for item in batch]
    targets = torch.tensor(targets)

    return data, targets


dataset = {sets: BBC(embeddings[sets], labels[sets]) for sets in ["train", "test"]}

loader = {
    "train": DataLoader(
        dataset["train"],
        batch_size=32,
        shuffle=True,
        drop_last=True,
        collate_fn=pack_embed,
        pin_memory=True,
    ),
    "test": DataLoader(
        dataset["test"], batch_size=128, shuffle=False, collate_fn=pack_embed
    ),
}
