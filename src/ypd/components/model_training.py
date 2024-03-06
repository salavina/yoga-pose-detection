import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize, \
Resize, CenterCrop
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import mlflow
from urllib.parse import urlparse
from PIL import ImageFile
import os
from ypd.entity.config_entity import TrainingConfig


class ModelTrainer(object):
    def __init__(self, config:TrainingConfig, loss_fn=None, optimizer=None):
        self.config = config
        self.model = self.load_model()
        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = optimizer if optimizer else optim.Adam(self.model.parameters(), lr=self.config.params_learning_rate)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.train_loader, self.val_loader = self.set_loaders()
        
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []
        self.total_epoches = 0
        
        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()
        
    def load_model(self):
        return torch.load(self.config.resnet_updated_base_model_path)
    
    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def set_loaders(self):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # image net statistics
        normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
        
        composer = Compose([Resize(256), CenterCrop(224), ToTensor(), normalizer])
        
        train_data = ImageFolder(root=os.path.join(self.config.training_data,'DATASET/TRAIN'), transform=composer)
        val_data = ImageFolder(root=os.path.join(self.config.training_data,'DATASET/TEST'), transform=composer)
        
        train_loader = DataLoader(train_data, batch_size=self.config.params_batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.config.params_batch_size)
        
        return train_loader, val_loader
    
    # higher order function to be set and built globally and constructed the inner fuction without knowning x and y before hand
    def _make_train_step_fn(self):
        # single batch operation
        def perform_train_step_fn(x,y):
            # set the train mode
            self.model.train()
            
            # step 1: compute model output
            yhat = self.model(x)
            
            # step 2: compute the loss  
            loss= self.loss_fn(yhat, y)
            
            # step 2': compute accuracy 
            yhat = torch.argmax(yhat,1)
            total_correct = (yhat ==y).sum().item()
            total = y.shape[0]
            acc = total_correct/total
            
            # step 3: compute the gradient
            loss.backward()
            
            #step4: update parameters
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            #step 5: return the loss
            return loss.item() , acc
        return perform_train_step_fn
    
    def _make_val_step_fn(self):
        # single batch operation
        def perform_val_step_fn(x,y):
            # set the model in val mode
            self.model.eval()
            
            #step 1: compute the prediction
            yhat = self.model(x)
            
            #step 2: compute the loss
            loss = self.loss_fn(yhat, y)
            # step 2': compute accuracy 
            yhat = torch.argmax(yhat,1)
            total_correct = (yhat ==y).sum().item()
            total = y.shape[0]
            acc = total_correct/total
            
            return loss.item(), acc
        return perform_val_step_fn
    
    def _mini_batch(self, validation=False):
        # one epoch operation 
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
            
        else: 
            data_loader = self.train_loader
            step_fn = self.train_step_fn
            
        if data_loader is None:
            return None
        
        mini_batch_losses = []
        mini_batch_accs = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            mini_batch_loss, mini_batch_acc = step_fn(x_batch,y_batch)
            
            mini_batch_losses.append(mini_batch_loss)
            mini_batch_accs.append(mini_batch_acc)
        
        loss = np.mean(mini_batch_losses)
        acc = np.mean(mini_batch_accs)
        return loss, acc
    
    def train(self, seed=42):
        self.set_seed(seed)
        
        for epoch in range(self.config.params_epochs):
            self.total_epoches +=1
            
            # perform training on mini batches within 1 epoch
            loss, acc = self._mini_batch(validation=False)
            self.losses.append(loss)
            self.accuracy.append(acc)
            # now calc validation
            with torch.no_grad():
                val_loss, val_acc = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
                self.val_accuracy.append(val_acc)
                
            print(
                f'\nEpoch: {epoch+1} \tTraining Loss: {loss:.4f} \tValidation Loss: {val_loss:.4f}'
            )
            print(
                f'\t\tTraining Accuracy: {100 * acc:.2f}%\t Validation Accuracy: {100 * val_acc:.2f}%'
            )
        self.save_checkpoint()
            
    def save_checkpoint(self):
        checkpoint = {'epoch': self.total_epoches,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': self.losses,
                      'accuracy': self.accuracy,
                      'val_loss': self.val_losses,
                      'val_accuracy': self.val_accuracy
                      }
        torch.save(checkpoint, self.config.resnet_trained_model_path)
        
    def load_checkpoint(self):
        checkpoint = torch.load(self.config.resnet_trained_model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_epoches = checkpoint["epoch"]
        self.losses = checkpoint["loss"]
        self.accuracy = checkpoint['accuracy']
        self.val_accuracy = checkpoint['val_accuracy']
        self.val_losses = checkpoint["val_loss"]
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
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({'train_loss': np.mean(self.losses),'val_loss': np.mean(self.val_losses), 'train_accuracy': np.mean(self.accuracy), 'val_accuracy': np.mean(self.val_accuracy)})
        
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="ResNet18Model")
            else:
                mlflow.pytorch.log_model(self.model, "model")