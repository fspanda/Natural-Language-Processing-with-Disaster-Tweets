import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import nltk
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('omw-1.4')



def get_sentence(c1, c2, c3):
    if pd.isnull(c1):
        c1 = ""
    if pd.isnull(c2):
        c2 = ""
    if pd.isnull(c3):
        c3 = ""
    return f"{c1} {c2} {c3}"
    
def clean_text(data):
    """
    input: data: a dataframe containing texts to be cleaned
    return: the same dataframe with an added column of clean text
    """
    data['clean_text'] = data['text'].str.lower()
    stop_words = list(stopwords.words('english'))
    punctuations = list(punctuation)
    clean_text = []
    lemmatizer = WordNetLemmatizer()
    for idx, row in enumerate(data['clean_text']):
        split_text = row.split()
        clean_text = [lemmatizer.lemmatize(word) for word in split_text if word not in stop_words and word not in punctuation]
        
        clean_text = ' '.join(clean_text)
        data.loc[idx]['clean_text'] = clean_text
    return data
    
class DisasterTweetClassifier(nn.Module):
    def __init__(self, model_name="dabert-base-model",use_other_features=False,clean_text=False):
        super().__init__()
        self.num_classes = 2
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)     
        self.data = pd.read_csv("path/to/training/data.csv")
        if use_other_features:
            self.data["text"]=self.data.apply(lambda x: get_sentence(x["keyword"],x["location"],x["text"]),axis=1)
        if clean_text:
            self.data=clean_text(self.data)
        self.train_data,self.val_data=train_test_split(self.data,test_size=0.2,random_state=42)
        self.loss_fn = nn.CrossEntropyLoss()
        
        

    def call(self, input_ids):
        # Forward all of the input data through the model
        logits = self.model(input_ids)[0]
        return logits


    def forward(self, input_ids):
        return self.call(input_ids)

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

        
        input_ids = [self.tokenizer.encode(text) for text in self.train_data["text"]]   
        # Create PyTorch tensors for the input and target variables
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(self.train_data["target"])
        train_data = torch.utils.data.TensorDataset(input_ids, labels)
        train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        # Return a DataLoader for the validation set
        # This will be used to iterate over the validation data during training
     
        input_ids = [self.tokenizer.encode(text) for text in self.val_data["text"]]
   

        # Create PyTorch tensors for the input and target variables
        input_ids = torch.tensor(input_ids) 
        labels = torch.tensor(self.val_data["target"])
        val_data = torch.utils.data.TensorDataset(input_ids,labels)
        val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)
        return val_dataloader




# Define the model
model = DisasterTweetClassifier()

# train Model
model.fit()
