import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda
from typing import Callable, Tuple, List
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

class WatermarkDataset(Dataset):
    TRIGGER_TARGET = 0
    def __init__(self, 
            dataset : Dataset,
            watermark_func : Callable[[np.ndarray], np.ndarray],
            watermark_size : int = None,
            transform : Compose = None,
            inv_transform : Compose = None,
            ) -> None:
        super().__init__()
        self.watermark_func = watermark_func
        self.watermark_size = watermark_size if watermark_size is not None else len(dataset)
        self.dataset = dataset
        self.transform = transform if transform is not None else Compose([])
        self.inv_transform = inv_transform

        self.watermark_transform = Compose([
            np.array,
            self.watermark_func,
        ])

    def __len__(self) -> int:
        return len(self.dataset) + self.watermark_size

    def getimage(self, index:int) :
        # if it's in original range give original image, else give watermarked image
        if index < len(self.dataset):
            return self.dataset[index]
        
        X,_ = self.dataset[index % len(self.dataset)]
        watermarked = self.watermark_transform(X)

        return watermarked, WatermarkDataset.TRIGGER_TARGET
    
    def __getitem__(self, index : int) -> Tuple[torch.Tensor, torch.Tensor]:
        X,y = self.getimage(index)
        return self.transform(X), torch.tensor(y)

    def get_watermarked_dataset(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        '''Method to get the list of all watermarked samples'''
        return [self[i]
                for i in range(len(self.dataset), len(self)) 
                if self.dataset[i%len(self.dataset)][1] != WatermarkDataset.TRIGGER_TARGET]

    def get_clean_dataset(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        '''Method to get the list of all the untouched samples'''
        return [self[i]
                for i in range(len(self.dataset))
                if self.dataset[i%len(self.dataset)][1] != WatermarkDataset.TRIGGER_TARGET]

    def plot_watermark_diff(self, index : int = 0):
        img1, _ = self.getimage(index)
        img2, _ = self.getimage(index + len(self.dataset))
        _, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img1)
        axes[0].set_title('Original')
        axes[1].imshow(img2)
        axes[1].set_title('Watermark')
