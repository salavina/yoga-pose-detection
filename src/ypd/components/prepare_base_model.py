from torchvision import models
import torch
from torchsummary import summary
from pathlib import Path
from ypd import logger
from transformers import ViTImageProcessor, ViTForImageClassification
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
        if self.config.params_type.lower() == 'resnet':
            self.full_model = self._prepare_full_model(
                model=self.get_base_model(),
                classes=self.config.params_classes,
                freeze_all=True,
                freeze_till=None
            )
            
            self.full_model.to(self.device)
            summary(self.full_model, input_size=tuple(self.config.params_image_size), device=self.device)
            self.save_model(checkpoint=self.full_model, path=self.config.resnet_updated_base_model_path)
            logger.info(f"saved updated ResNet model to {str(self.config.root_dir)}")
        
        elif self.config.params_type.lower() == 'vit':
            vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                                  num_labels=self.config.params_classes, ignore_mismatched_sizes=True)
            # changing labels to corresponding class names
            vit_model.config.id2label = {0: 'downdog', 1: 'goddess', 2: 'plank', 3: 'tree', 4: 'warrior2'}
            vit_model.config.label2id = {'downdog': 0, 'goddess': 1, 'plank': 2, 'tree': 3, 'warrior2': 4}
            vit_processor.save_pretrained(self.config.root_dir)
            vit_model.save_pretrained(self.config.root_dir)
            print(vit_model)
            logger.info(f"saved updated ViT model to {str(self.config.root_dir)}")

    
    @staticmethod
    def save_model(path: Path, checkpoint: dict):
        torch.save(checkpoint, path)