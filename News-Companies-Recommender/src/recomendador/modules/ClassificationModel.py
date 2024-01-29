import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from simpletransformers.classification import ClassificationModel
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AdamW
from ignite.contrib.handlers import PiecewiseLinear
from ignite.handlers import ModelCheckpoint
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy
from ignite.utils import to_onehot
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.engine import Engine
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

class ClassificationModel:
    def __init__(self, train_data, pred_data):
        self.train_data = train_data
        self.pred_data = pred_data

    def generate_train_test_dataset(self, data_processed):
    
        train, test = train_test_split(data_processed, test_size=0.2, random_state=42, stratify=data_processed['label'])
        train = Dataset(pa.Table.from_pandas(train))
        test = Dataset(pa.Table.from_pandas(test))
        train = train.map(self.tokenize_function, batched=True)
        test = test.map(self.tokenize_function, batched=True)
        train = train.remove_columns(["text"])
        train = train.remove_columns(["__index_level_0__"])
        train = train.rename_column("label", "labels")
        train.set_format("torch")
        test = test.remove_columns(["text"])
        test = test.remove_columns(["__index_level_0__"])
        test = test.rename_column("label", "labels")
        test.set_format("torch")
        train_dataloader = DataLoader(train, shuffle=True, batch_size=4)
        eval_dataloader = DataLoader(test, batch_size=4)
        return train_dataloader, eval_dataloader

    def tokenize_function(self, examples):
        tokenizer =  AutoTokenizer.from_pretrained('chriskhanhtran/spanberta', num_labels=self.n_classes)
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    def preprocess_data(self):
        print("Preprocessing the data...")
        self.train_data['text'] = self.train_data['news_title'] + ' ' + self.train_data['news_text_content']
        self.train_data['label'] = self.train_data['category']
        data_model = self.train_data[['text', 'label']].dropna()
        le = preprocessing.LabelEncoder()
        data_model['label'] = le.fit_transform(data_model['label'])
        np.save('../data/archivos_auxiliares/models/classes.npy', le.classes_)
        self.n_classes = len(data_model['label'].drop_duplicates())
        self.train_dataloader, self.eval_dataloader = self.generate_train_test_dataset(data_model)

    def train_step(self, engine, batch):  
        self.model.train()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def evaluate_step(self, engine, batch):
        self.model.eval()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        pred_ohe = to_onehot(predictions, num_classes=self.n_classes)
        y_ohe = to_onehot(batch["labels"], num_classes=self.n_classes)
        return {'y_pred': pred_ohe, 'y': y_ohe}

    def log_training_results(self, engine):
        self.train_evaluator.run(self.train_dataloader)
        metrics = self.train_evaluator.state.metrics
        print(metrics)
        avg_accuracy = metrics['accuracy']
        print(f"Training Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}")
        
    def log_validation_results(self, engine):
        self.validation_evaluator.run(self.eval_dataloader)
        metrics = self.validation_evaluator.state.metrics
        print(metrics)
        avg_accuracy = metrics['accuracy']
        print(f"Validation Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}")

    def score_function(self, engine):
        val_accuracy = engine.state.metrics['accuracy']
        return val_accuracy

    def create_model(self):
        print("Creating classification model..")
        self.model = AutoModelForSequenceClassification.from_pretrained('chriskhanhtran/spanberta', num_labels=self.n_classes)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.num_epochs = 50
        num_training_steps = self.num_epochs * len(self.train_dataloader)
        milestones_values = [
                (0, 5e-5),
                (num_training_steps, 0.0),
            ]
        self.lr_scheduler = PiecewiseLinear(
                self.optimizer, param_name="lr", milestones_values=milestones_values
            )
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

    def train_model(self):
        save_path = '../data/archivos_auxiliares/models/bert-base-cased_model_1703.pt'
        self.preprocess_data()
        self.create_model()
        if os.path.exists(save_path):
            print('Model exists. Loading model...')
            self.model.load_state_dict(torch.load(save_path))
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.model.to(device)
            self.model.eval()
        else:
            print('Model does not exists. Training model...')
            self.trainer = Engine(self.train_step)
            self.trainer.add_event_handler(Events.ITERATION_STARTED, self.lr_scheduler)
            pbar = ProgressBar()
            pbar.attach(self.trainer)
            pbar.attach(self.trainer, output_transform=lambda x: {'loss': x})
            self.train_evaluator = Engine(self.evaluate_step)
            self.validation_evaluator = Engine(self.evaluate_step)
            Accuracy().attach(self.train_evaluator, 'accuracy')
            Accuracy().attach(self.validation_evaluator, 'accuracy')
            self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.log_validation_results)
            self.handler = EarlyStopping(patience=5, score_function=self.score_function, trainer=self.trainer)
            self.validation_evaluator.add_event_handler(Events.COMPLETED, self.handler)
            self.checkpointer = ModelCheckpoint(dirname='models', filename_prefix='bert-base-cased', n_saved=1, create_dir=True)
            self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.checkpointer, {'model': self.model})
            self.state = self.trainer.run(self.train_dataloader, max_epochs=self.num_epochs)

    def generate_pred_dataset(self, data_pred):
        pred = Dataset(pa.Table.from_pandas(data_pred))
        pred = pred.map(self.tokenize_function, batched=True)
        pred = pred.remove_columns(["text"])
        pred.set_format("torch")
        pred_dataloader = DataLoader(pred, batch_size=128)
        return pred_dataloader

    def evaluate_predict(self, df):
        print("Evaluating predictions on the new dataset...")
        le = preprocessing.LabelEncoder()
        le.classes_ = np.load('../data/archivos_auxiliares/models/classes.npy', allow_pickle=True)
        final_predictions = []
        final_probabilities = []
        pred_dataloader = self.generate_pred_dataset(df)
        m = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            for batch in tqdm(pred_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits
                softmax_p = m(logits)
                probabilities = list(torch.max(softmax_p, dim=1).values.cpu().numpy())
                prediction = list(le.inverse_transform(torch.argmax(logits, dim=-1).cpu()))
                final_predictions += prediction
                final_probabilities += probabilities
        return final_predictions, final_probabilities

    def predict(self):
        df = self.pred_data
        df['text'] = df['news_title'] + ' ' + df['news_text_content']
        df['classification_category'], df['classification_probability'] = self.evaluate_predict(df[['text']])
        return df[['news_id', 'classification_category', 'classification_probability']]