import torch.nn as nn


class Classificator(nn.Module):
    def __init__(self, hidden_size, n_classes, embed_size=300, output_size=50):
        super().__init__()
        self.n_classes = n_classes
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers=2, dropout=0.2, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.classifier = nn.Linear(output_size, n_classes)
        self.batchnorm = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()

    def forward(self, embed):
        _, (hidden_t, _) = self.lstm(embed)
        output = self.fc(hidden_t[-1, ...])  # if num_layers > 1
        output = self.relu(self.batchnorm(output))
        scores = self.classifier(output)

        return scores
