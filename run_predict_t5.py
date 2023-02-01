import torch
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from tqdm import tqdm
import json
from sklearn.metrics import f1_score, precision_score, recall_score
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentTypeError
import warnings
warnings.simplefilter("ignore")

# Create class for data preparation
class SimpleDataset:
  def __init__(self, tokenized_texts):
    self.tokenized_texts = tokenized_texts
  
  def __len__(self):
    return len(self.tokenized_texts["input_ids"])
  
  def __getitem__(self, idx):
    return {k: v[idx] for k, v in self.tokenized_texts.items()}

def validate_datafile(astring):
  if not astring.endswith('.tsv'):
    raise ArgumentTypeError("%s: is an invalid file, provide a tsv." % astring)
  return astring

parser = ArgumentParser(description="Sentiment Classification Prediction", formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--model_path", type=str, required=True, help="path to a trained model")
parser.add_argument("--file_name", type=validate_datafile, required=True, help="path to unlabeled tweets to predict")
parser.add_argument("--gold_file", type=validate_datafile, required=True, help="path to gold labels")
parser.add_argument("--text_column", type=str, default='tweet', help="path to save predictions")
parser.add_argument("--lang_code", type=str, default='text', help="language code")
    
args = parser.parse_args()

# Load tokenizer and model, create trainer
model_name = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
trainer = Trainer(model=model, tokenizer=tokenizer)

# Import data from csv-file stored on Google Drive
file_name = args.file_name
text_column = args.text_column

df_pred = pd.read_csv(file_name, sep='\t')
ids = df_pred.iloc[:,0].astype('str').tolist()
pred_texts = df_pred[text_column].astype('str').tolist()
# Tokenize texts and create prediction data set
print('Tokenizing texts...')
tokenized_texts = tokenizer(["Afrisenti: "+sentence for sentence in tqdm(pred_texts)], 
                            truncation=True, padding=True,return_tensors="pt").input_ids
print('Creating prediction data set...')
predictions = model.generate(tokenized_texts, max_length=16)
print("Predictions done. Decoding predictions...")
predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

id2label = {
    2: 'negative',
    0: 'neutral',
    1: 'positive'
}

# Transform predictions to labels
labels = [pred.strip() for pred in predictions]
preds = pd.Series(labels).map(id2label)

# Create submissions files directory if not available
if os.path.isdir(model_name):
  print('Data directory found.')
  SUBMISSION_PATH = os.path.join(model_name, 'submission')
  if not os.path.isdir(SUBMISSION_PATH):
    print('Creating submission files directory.')
    os.mkdir(SUBMISSION_PATH)
else:
  print(model_name + ' is not a valid directory or does not exist!')

# Create DataFrame with texts, predictions, and labels
df = pd.DataFrame(list(zip(ids,pred_texts,preds,labels)), columns=['ID', 'text', 'pred', 'label'])
df.to_csv(os.path.join(SUBMISSION_PATH, 'predictions.tsv'), sep='\t', index=False)

df = pd.DataFrame(list(zip(ids,labels)), columns=['ID', 'label'])
sub_path = os.path.join(SUBMISSION_PATH, 'pred_' + args.lang_code + '.tsv')
df.to_csv(os.path.join(SUBMISSION_PATH, 'pred_' +
          args.lang_code + '.tsv'), sep='\t', index=False)

# the first file in the submission zip is expected to be the submission csv
submission_df = pd.read_csv(sub_path, sep='\t')
gold_df = pd.read_csv(args.gold_file, sep='\t')
submission_df = submission_df.sort_values("ID")
gold_df = gold_df.sort_values("ID")

print("Labels in gold standard data: ")
print(gold_df.label.unique())
print("Labels in submission data: ")
print(submission_df.label.unique())

f1 = f1_score(y_true=gold_df["label"],
              y_pred=submission_df["label"], average="weighted")
recall = recall_score(
    y_true=gold_df["label"], y_pred=submission_df["label"], average="weighted")
precision = precision_score(
    y_true=gold_df["label"], y_pred=submission_df["label"], average="weighted")

m_f1 = f1_score(y_true=gold_df["label"],
                y_pred=submission_df["label"], average="macro")
m_recall = recall_score(
    y_true=gold_df["label"], y_pred=submission_df["label"], average="macro")
m_precision = precision_score(
    y_true=gold_df["label"], y_pred=submission_df["label"], average="macro")

# export results to json file
results = {
    "f1": f1,
    "recall": recall,
    "precision": precision,
    "macro_f1": m_f1,
    "macro_recall": m_recall,
    "macro_precision": m_precision
}

with open(os.path.join(SUBMISSION_PATH, 'results.json'), 'w') as f:
    json.dump(results, f)

print("Results: ")
print(results)
