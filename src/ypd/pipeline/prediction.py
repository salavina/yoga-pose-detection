from pathlib import Path
import torch
from PIL import ImageFile, Image
from torchvision.transforms import Compose, ToTensor, Normalize, \
Resize, CenterCrop
import numpy as np
import os



class PredictionPipeline:
    def __init__(self, base_model_path=None, trained_model_dict=None):
        self.base_model_path = base_model_path if base_model_path else Path('artifacts/prepare_base_model/resnet_updated_base_model.pth')
        self.trained_model_dict = trained_model_dict if trained_model_dict else Path('artifacts/training/resnet_model.pth')
        self.model = self.load_model()
        self.load_checkpoint()
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.model.to(self.device)
    
    
    def load_model(self):
        return torch.load(self.base_model_path)
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.trained_model_dict)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.train() # always use train for resuming traning
    
    
    def _preprocess_image(self, filename):
        # Load the image and apply the transformation
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(filename)
        normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
        
        composer = Compose([Resize(256), CenterCrop(224), ToTensor(), normalizer])
        
        image = composer(image).unsqueeze(0)  # Add a batch dimension
        return image
    
    
    def predict(self, filename):
        self.load_checkpoint()
        self.model.eval()
        preprocess_image = self._preprocess_image(filename)
        x_tensor = torch.as_tensor(preprocess_image).float()
        y_hat_tensor = self.model(x_tensor.to(self.device))
        
        # set it back to the train mode
        self.model.train()
        
        labels = {0: 'downdog', 1: 'goddess', 2: 'plank', 3: 'tree', 4: 'warrior2'}
        
        return labels[np.argmax(y_hat_tensor.detach().cpu().numpy())]

