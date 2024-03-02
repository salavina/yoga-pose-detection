from torchvision import models
import torch
from torchsummary import summary
from pathlib import Path
from ypd import logger
from ypd.entity.config_entity import (PrepareBaseModelConfig)

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_base_model(self):
        resnet_model = models.resnet18(pretrained=self.config.params_pretrained)
        resnet_model.to(self.device)
        self.save_model(checkpoint=resnet_model, path=self.config.resnet_base_model_path)
        return resnet_model
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_till, freeze_all=False):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        
        elif (freeze_till is not None) and (freeze_till > 0):
            for param in model.parameters()[:-freeze_till]:
                param.requires_grad = False
        
        n_inputs = model.fc.in_features
        model.fc = torch.nn.Linear(n_inputs, classes)
        return model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.get_base_model(),
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None
        )
        
        self.full_model.to(self.device)
        summary(self.full_model, input_size=tuple(self.config.params_image_size), device=self.device)
        self.save_model(checkpoint=self.full_model, path=self.config.resnet_updated_base_model_path)
        logger.info(f"saved updated model to {str(self.config.root_dir)}")

    
    @staticmethod
    def save_model(checkpoint: dict, path: Path):
        torch.save(checkpoint, path)