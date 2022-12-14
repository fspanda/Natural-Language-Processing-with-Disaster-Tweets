import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import DabertTokenizer, DabertModel
import pandas as pd
from sklearn.model_selection import train_test_split

class DisasterTweetClassifier(nn.Module):
    def __init__(self, model_name="dabert-base-model"):
        super().__init__()
        self.num_classes = 2
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)     
        self.data = pd.read_csv("path/to/training/data.csv")
        self.train_data,self.val_data=train_test_split(self.data,test_size=0.2,random_state=42)
        

    def call(self, input_ids):
        # Call method to forward input through the model
        logits = self.model(input_ids)[0]
        return logits

    def training_step(self, batch, batch_idx):
        # Training step
        input_ids, labels = batch
        logits = self.forward(input_ids)
        loss = self.loss_fn(logits, labels)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # Validation step
        input_ids, labels = batch
        logits = self.forward(input_ids)
        val_loss = self.loss_fn(logits, labels)
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        # Validation end
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"avg_val_loss": avg_val_loss}

               

    def train_dataloader(self):
        # Return a DataLoader for the training set
        # This will be used to iterate over the training data during training
        # The data should be prepared in the `prepare_data()` method
        
        input_ids = [self.tokenizer.encode(text) for text in self.train_data["text"]]
        locations = [self.tokenizer.encode(location) for location in self.train_data["location"]]
        keywords = [self.tokenizer.encode(keyword) for keyword in self.train_data["keyword"]]

        # Create PyTorch tensors for the input and target variables
        input_ids = torch.tensor(input_ids)
        locations = torch.tensor(locations)
        keywords = torch.tensor(keywords)
        labels = torch.tensor(self.train_data["target"])
        train_data = torch.utils.data.TensorDataset(input_ids, locations, keywords, labels)
        train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        # Return a DataLoader for the validation set
        # This will be used to iterate over the validation data during training
        # The data should be prepared in the `prepare_data()` method
        input_ids = [self.tokenizer.encode(text) for text in self.val_data["text"]]
        locations = [self.tokenizer.encode(location) for location in self.val_data["location"]]
        keywords = [self.tokenizer.encode(keyword) for keyword in self.val_data["keyword"]]

        # Create PyTorch tensors for the input and target variables
        input_ids = torch.tensor(input_ids)
        locations = torch.tensor(locations)
        keywords = torch.tensor(keywords)
        labels = torch.tensor(self.val_data["target"])
        val_data = torch.utils.data.TensorDataset(input_ids, locations, keywords, labels)
        val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)
        return val_dataloader




