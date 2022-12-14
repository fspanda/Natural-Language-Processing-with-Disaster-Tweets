import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split

import pandas as pd

import nltk
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('omw-1.4')


model_name="bert-base-uncased"
max_token_len=100
batch_size=256

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

class TweetLabelDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        model_name="bert-base-uncased",
        max_token_len: int = 128,
        use_other_features=False,
        clean_text=False
        ):
        if use_other_features:
            self.data["text"]=self.data.apply(lambda x: get_sentence(x["keyword"],x["location"],x["text"]),axis=1)
        if clean_text:
            self.data=clean_text(self.data)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data = data
        self.max_token_len = max_token_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        text = data_row.text
        labels = data_row["target"]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            )
        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels)
            )
        
class TweetLabelDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, model_name, batch_size=256, max_token_len=128,use_other_features=False,clean_text=False):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.model_name=model_name
        self.max_token_len = max_token_len
        self.use_other_features=use_other_features
        self.clean_text=clean_text
    def setup(self, stage=None):
        self.train_dataset = TweetLabelDataset(
            data=self.train_df,            
            model_name=self.model_name,
            max_token_len=self.max_token_len,
            use_other_features=self.use_other_features,
            clean_text=self.clean_text)
        self.test_dataset = TweetLabelDataset(
            self.test_df,
            model_name=self.model_name,
            max_token_len=self.max_token_len,
            use_other_features=self.use_other_features,
            clean_text=self.clean_text)
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=50
            )
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=50
            )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=50)
        
class TweetLabelTagger(pl.LightningModule):
    def __init__(self,  n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        auc = roc_auc_score(labels, predictions)
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
            )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
                )
            )
        
      
      
N_EPOCHS = 20
BATCH_SIZE = 150        
model_name="bert-base-uncased"
max_token_len=100



train_df=pd.read_csv("/home/ybi/study/Kaggle/Natural-Language-Processing-with-Disaster-Tweets/Data/train.csv")

train_df,val_df=train_test_split(train_df,test_size=0.2,random_state=42)

train_dataset = TweetLabelDataset(
  train_df,
  max_token_len=100
)


data_module = TweetLabelDataModule(
  train_df,
  val_df,  
  model_name=model_name,
  batch_size=BATCH_SIZE
  )

steps_per_epoch= 10000 // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS
warmup_steps = total_training_steps // 5

model = TweetLabelTagger(  
  n_warmup_steps=warmup_steps,
  n_training_steps=total_training_steps
)

checkpoint_callback = ModelCheckpoint(
  dirpath="/home/ybi/study/Kaggle/Natural-Language-Processing-with-Disaster-Tweets/Checkpoint",
  filename="1214_Tweet_Label",
  save_top_k=1,
  verbose=True,
  monitor="val_loss",
  mode="min"
)

logger = TensorBoardLogger("lightning_logs", name="ItemLabel_Multi")
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
trainer = pl.Trainer(
  logger=logger,  
  callbacks=[checkpoint_callback,early_stopping_callback],
  max_epochs=N_EPOCHS,
  gpus=[0]
)

trainer.fit(model, data_module)