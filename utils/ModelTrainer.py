import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple
from tqdm import tqdm
import numpy as np


class ModelTrainer:
    def __init__(self, 
                model : torch.nn.Module,
                lr : float = 1e-4,
                weight_decay : float = 0,
                device :str = 'cuda'
                ):
        self.model = model
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_loop(self, trainset : DataLoader):
        loss_sum = 0
        self.model.train()

        for X,y in trainset:
            X,y = X.to(self.device), y.to(self.device)

            y_hat = self.model(X)

            self.optimizer.zero_grad()
            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()
            
            loss_sum += loss.cpu().item()

        return loss_sum
    
    def test(self, dataloader : DataLoader):
        loss_sum = 0
        correct = 0
        n = 0
        ys = np.array()
        self.model.eval()

        with torch.no_grad():
            for X,y in dataloader:
                X,y = X.to(self.device), y.to(self.device)

                y_hat = self.model(X)

                loss = self.criterion(y_hat, y)            
                loss_sum += loss.cpu().item()

                y_hat = y_hat.argmax(1)
                correct += (y_hat == y).sum().cpu().item()
                ys = np.append(ys, y_hat.numpy(force=True))
                n += y.shape[0]

        accuracy = correct / n
        return loss_sum, accuracy, ys


    def train(self, trainset : DataLoader, valset : DataLoader, epoches : int) -> List[Tuple[int, float, float]]:
        """
            Trains a model and computes the loss for each epoch.
            Returns a list of tuples for which we have
                (epoch number, training loss, validation loss, validation accuracy)
        """
        history = list()

        for e in tqdm(range(epoches)):
            train_loss  = self.train_loop(trainset)
            val_loss, val_acc, _ = self.test(valset)

            history.append((e, train_loss, val_loss, val_acc))

        return history


    

        


