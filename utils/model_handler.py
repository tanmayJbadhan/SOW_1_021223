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
            pirate_set : Dataset,
            transform : Compose = None,
            device : str = 'cuda',
        ) -> None:
        self.dataset = dataset
        self.testset = testset
        self.pirate_set = pirate_set
        self.model = model
        self.device = device
        self.transform = transform
        self.model.to(device)
        
    def finetune(self, pattern_func, epochs:int=2):
        """Finetunes a model on the given Dataset and then returns it"""
        model = deepcopy(self.model)
        trainer = ModelTrainer(model, device=self.device)
        wm_data = WatermarkDataset(self.dataset, watermark_func=pattern_func, transform=self.transform, change_label=True)
        wm_set = ModelHandler.to_dataloader(wm_data)

        history = trainer.train(wm_set, None, epochs)

        return trainer.model, self.metrics(trainer.model, pattern_func)

    
    def evaluate(self, pattern_func:Callable[[np.array], np.array]):
        wm_data = WatermarkDataset(self.testset, watermark_func=pattern_func, transform=self.transform, change_label=False)


        wm_set = ModelHandler.to_dataloader(wm_data.get_watermarked_dataset())
        trainer = ModelTrainer(self.model, device=self.device)

        
        return trainer.test(self.model, wm_set)[1]
    
    def finetune_attack(self, model:nn.Module, pattern_func, epoches: int = 2):
        override_dataset = WatermarkDataset(self.pirate_set, watermark_func=lambda x: x, transform=self.transform).get_clean_dataset()
        overrideset = ModelHandler.to_dataloader(override_dataset)

        trainer = ModelTrainer(deepcopy(model), device=self.device)
        trainer.train(overrideset, epoches=epoches)
        attacked_model = trainer.model

        return attacked_model, self.metrics(attacked_model, pattern_func=pattern_func)

        
    def metrics(self, model: nn.Module, pattern_func):
        trainer = ModelTrainer(model)
        # Evaluation
        wm_test = WatermarkDataset(self.testset, watermark_func=pattern_func, transform=self.transform)
        clean_testset = ModelHandler.to_dataloader(wm_test.get_clean_dataset())
        wm_testset    = ModelHandler.to_dataloader(wm_test.get_watermarked_dataset())

        _, clean_accuracy, clean_ys = trainer.test(model, clean_testset)
        _, wm_accuracy,    wm_ys    = trainer.test(model, wm_testset)

        fpr = (clean_ys==WatermarkDataset.TRIGGER_TARGET).mean()

        #######
        # INSERT HERE ANY EVALUATION METRIC
        #TANMAY CODE
        return clean_accuracy, wm_accuracy, fpr
        

        
    @staticmethod
    def to_dataloader(dataset : Dataset, batch_size:int=256) -> DataLoader:
        return DataLoader(dataset, batch_size, shuffle=True)
    
    @staticmethod
    def download_model(repo:str, model_name:str, device:str='cuda',dataset : Dataset = None,
            testset : Dataset = None,
            pirate_set : Dataset = None,
            transform : Compose = None):
        model = torch.hub.load(repo, model_name, pretrained=True)
        return ModelHandler(model, dataset, testset, pirate_set=pirate_set, transform=transform, device=device)


# ModelHandler.download_model("chenyaofo/pytorch-cifar-models", 'cifar10_resnet20')