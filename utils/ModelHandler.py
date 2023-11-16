import torch 
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
from utils import ModelTrainer

class ModelHandler:
    def __init__(self, 
            model : torch.nn.Module,
            
            device : str = 'cuda'
        ) -> None:
        self.model = model
        self.device = device
        self.model.to(device)
        
    def finetune(self, dataset: Dataset, epochs:int=10):
        """Finetunes a model on the given Dataset and then returns it"""
        splits = torch.utils.data.random_split(dataset,lengths=[0.6,0.1,0.3])
        trainset, valset, testset = list(map(ModelHandler.to_dataloader, splits))

        model = deepcopy(self.model)
        trainer = ModelTrainer(model, device=self.device)
        history = trainer.train(trainset, valset, epochs, )


        return model, history

        
    @staticmethod
    def to_dataloader(dataset : Dataset, batch_size:int=256) -> DataLoader:
        return DataLoader(dataset, batch_size, shuffle=True)
    
    @staticmethod
    def download_model(repo:str, model_name:str, device:str='cuda'):
        model = torch.hub.load(repo, model_name, pretrained=True)
        return ModelHandler(model, device)


# ModelHandler.download_model("chenyaofo/pytorch-cifar-models", 'cifar10_vgg11_bn')