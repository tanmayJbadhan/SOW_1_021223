import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
from .model_trainer import ModelTrainer
from .watermark_dataset import WatermarkDataset
import numpy as np
from torchvision.transforms import Compose
from typing import Callable

class ModelHandler:
    def __init__(self, 
            model : torch.nn.Module,
            dataset : Dataset,
            testset : Dataset,
            transform : Compose = None,
            device : str = 'cuda',
        ) -> None:
        self.dataset = dataset
        self.testset = testset
        self.model = model
        self.device = device
        self.transform = transform
        self.model.to(device)
        
    def finetune(self, trainset: DataLoader, valset: DataLoader, epochs:int=2):
        """Finetunes a model on the given Dataset and then returns it"""
        model = deepcopy(self.model)
        trainer = ModelTrainer(model, device=self.device)
        history = trainer.train(trainset, valset, epochs)

        return trainer, history
    
    def evaluate(self, pattern_func:Callable[[np.array], np.array]):
        wm_data = WatermarkDataset(self.dataset, watermark_func=pattern_func, transform=self.transform)
        wm_test = WatermarkDataset(self.testset, watermark_func=pattern_func, transform=self.transform)

        splits = torch.utils.data.random_split(wm_data,lengths=[0.9, 0.1])
        trainset, valset = list(map(ModelHandler.to_dataloader, splits))

        clean_testset = ModelHandler.to_dataloader(wm_test.get_clean_dataset())
        wm_testset    = ModelHandler.to_dataloader(wm_test.get_watermarked_dataset())

        wm_trainer, history = self.finetune(trainset, valset)
        wm_model = wm_trainer.model

        # Evaluation
        _, clean_accuracy, clean_ys = wm_trainer.test(clean_testset)
        _, wm_accuracy,    wm_ys    = wm_trainer.test(wm_testset)

        fpr = (clean_ys==WatermarkDataset.TRIGGER_TARGET).mean()

        #######
        # INSERT HERE ANY EVALUATION METRIC
        
        return clean_accuracy, wm_accuracy, fpr

        
    @staticmethod
    def to_dataloader(dataset : Dataset, batch_size:int=256) -> DataLoader:
        return DataLoader(dataset, batch_size, shuffle=True)
    
    @staticmethod
    def download_model(repo:str, model_name:str, device:str='cuda',dataset : Dataset = None,
            testset : Dataset = None,
            transform : Compose = None):
        model = torch.hub.load(repo, model_name, pretrained=True)
        return ModelHandler(model, dataset, testset, transform=transform, device=device)


# ModelHandler.download_model("chenyaofo/pytorch-cifar-models", 'cifar10_vgg11_bn')