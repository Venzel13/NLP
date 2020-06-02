from collections import defaultdict

import torch
from torch import optim

from preproc import loader


class Trainer(object):
    def __init__(
        self,
        model,
        loss_fn,
        optim_name,
        optim_param,
        n_classes,
        save=True,
        verbose=False,
    ):
        self.model = model

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.loss_fn = loss_fn
        self.optimizer = optim.__dict__[optim_name](
            self.model.parameters(), **optim_param
        )
        self.save = save
        self.verbose = verbose
        self.best_model = self.model
        self.best_accuracy = 0
        self.logs = defaultdict(list)
        self.true = None
        self.pred = None

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, loader):
        self.model.train()
        running_loss = 0
        for x, y in loader:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            loss = self.train_step(x, y)
            running_loss += loss.item()
        running_loss /= len(loader)
        return running_loss

    def test(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        true = []
        pred = []
        for x, y in loader:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                pred_class = self.model(x).argmax(1)
                true.append(y)
                pred.append(pred_class)
                match = pred_class == y
                correct += match.sum().item()
                total += match.numel()
        accuracy = correct / total
        return true, pred, accuracy

    def save_model(self, current_acc):
        self.best_accuracy = current_acc
        self.best_model.load_state_dict(self.model.state_dict())

    def write_logs(self, loss, acc, epoch):
        self.logs["loss"].append(loss)
        self.logs["acc"].append(acc)
        if self.verbose:
            print("epoch {} Loss: {:.2f} Accuracy: {:.2f}".format(epoch, loss, acc))

    def learn(self, n_epochs, loader):
        for epoch in range(n_epochs):
            metrics = {}
            epoch_loss = self.train(loader["train"])
            true, pred, epoch_accuracy = self.test(loader["test"])

            if self.save and (epoch_accuracy > self.best_accuracy):
                self.save_model(epoch_accuracy)
                self.true = true
                self.pred = pred

            self.write_logs(epoch_loss, epoch_accuracy, epoch)
