import pandas as pd
import os
import json
import gspread
from gspread_dataframe import set_with_dataframe
from gspread_formatting import *
from sklearn.metrics import f1_score, precision_score, recall_score
from datetime import datetime

def get_results():
    models = ['xlm-roberta-base','xlm-roberta-large','bert-base-multilingual-cased','afro-xlmr-large','afro-xlmr-base','afriberta_large','afriberta_base','serengeti','afriteva_base','afriteva_large','afro-xlmr-base-lm','serengeti-lm']
    subsets = ['yo', 'twi', 'ts', 'sw', 'pt', 'pcm', 'ma', 'kr', 'ig','ha' ,'dz' ,'am','multilingual','average']
    df_pred = pd.DataFrame(columns=models,index=subsets)
    df_eval = pd.DataFrame(columns=models,index=subsets)
    for model in models:
        for sub in subsets[:-1]:
            path = f'../../../results/afrisenti/{model}/{sub}/'
            if os.path.exists(path+'all_results.json'):
                with open(path+'all_results.json') as json_file:
                    data = json.load(json_file)
                    predscore = data['predict_f1']
                    evalscore = data['eval_f1']
                    df_pred[model][sub] = predscore
                    df_eval[model][sub] = evalscore
    df_pred.loc['average'] = df_pred.mean()
    df_eval.loc['average'] = df_eval.mean()
    df_pred = df_pred.reset_index(level=0)
    df_eval = df_eval.reset_index(level=0)
    time_right_now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    df_pred = df_pred.rename(columns={'index':f'Subsets {time_right_now}'})
    df_eval = df_eval.rename(columns={'index':f'Subsets {time_right_now}'})
    print(df_pred)
    to_sheets(df_pred,'Updated Prediction')
    to_sheets(df_eval,'Evaluation')

def get_epochs():
    models = ['xlm-roberta-base', 'xlm-roberta-large', 'bert-base-multilingual-cased', 'afro-xlmr-large', 'afro-xlmr-base',
               'afriberta_large', 'afriberta_base', 'serengeti', 'afriteva_base', 'afriteva_large','afro-xlmr-base-lm', 'serengeti-lm']
    subsets = ['yo', 'twi', 'ts', 'sw', 'pt', 'pcm', 'ma', 'kr', 'ig','ha' ,'dz' ,'am','multilingual','average']
    df_pred = pd.DataFrame(columns=models,index=subsets)
    for model in models:
        for sub in subsets[:-1]:
            path = f'../../../results/afrisenti/{model}/{sub}/'
            if os.path.exists(path+'all_results.json'):
                with open(path+'all_results.json') as json_file:
                    data = json.load(json_file)
                    predscore = data['epoch']
                    df_pred[model][sub] = predscore
    df_pred.loc['average'] = df_pred.mean()
    df_pred = df_pred.reset_index(level=0)
    time_right_now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    df_pred = df_pred.rename(columns={'index':f'Subsets {time_right_now}'})    
    to_sheets(df_pred,'Epochs')


def get_dev_results():
    models = ['xlm-roberta-base', 'xlm-roberta-large', 'bert-base-multilingual-cased', 'afro-xlmr-large', 'afro-xlmr-base',
               'afriberta_large', 'afriberta_base', 'serengeti', 'afriteva_base', 'afriteva_large','afro-xlmr-base-lm', 'serengeti-lm']
    subsets = ['yo', 'twi', 'ts', 'sw', 'pt', 'pcm', 'ma', 'kr', 'ig','ha' ,'dz' ,'am','multilingual','or','tg','average']
    df_pred = pd.DataFrame(columns=models,index=subsets)
    for model in models:
        for sub in subsets[:-1]:
            path = f'../../../results/afrisenti/{model}/{sub}/submission/'
            if os.path.exists(path+'results.json'):
                with open(path+'results.json') as json_file:
                    data = json.load(json_file)
                    predscore = data['f1']
                    df_pred[model][sub] = predscore
            elif os.path.exists(path+f'results_{sub}.json'):
                with open(path+f'results_{sub}.json') as json_file:
                    data = json.load(json_file)
                    predscore = data['f1']
                    df_pred[model][sub] = predscore
            # elif os.path.exists(path+f'pred_{sub}.tsv'):
            #     sub_path = path+f'pred_{sub}.tsv'
            #     if sub == 'multilingual':
            #         gold_file= f"../SubtaskB/dev_gold/{sub}_dev_gold_label.tsv"
            #     else:
            #         gold_file= f"../SubtaskA/dev_gold/{sub}_dev_gold_label.tsv"
            #     submission_df = pd.read_csv(sub_path, sep='\t')
            #     gold_df = pd.read_csv(gold_file, sep='\t')
            #     submission_df = submission_df.sort_values("ID")
            #     gold_df = gold_df.sort_values("ID")
            #     try:
            #         predscore = f1_score(y_true = gold_df["label"], y_pred = submission_df["label"], average="weighted")
            #     except TypeError:
            #         print(sub,model)
            #         predscore = 0
            #     df_pred[model][sub] = predscore
            elif sub in ['or','tg']:
                new_path = f'../../../results/afrisenti/{model}/multilingual/submission/'
                sub_path = new_path+f'pred_{sub}.tsv'
                if os.path.exists(new_path+f"results_{sub}.json"):
                    with open(new_path+f"results_{sub}.json") as json_file:
                        data = json.load(json_file)
                        predscore = data['f1']
                        df_pred[model][sub] = predscore
    df_pred.loc['average'] = df_pred.mean()
    df_pred = df_pred.reset_index(level=0)
    time_right_now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    df_pred = df_pred.rename(columns={'index': f'Subsets {time_right_now}'})
    print(df_pred)
    df_pred.to_csv(f'dev_results.csv')
    to_sheets(df_pred,'Updated Dev Prediction')


def data_details():
    subsets = ['yo', 'twi', 'ts', 'sw', 'pt', 'pcm', 'ma', 'kr', 'ig','ha' ,'dz' ,'am','multilingual','or','tg']
    df = pd.DataFrame(columns=['Task','Train','Dev','Test','Task Train','Task Dev','Task Test'],index=subsets)
    df['Subsets'] = df.index
    for sub in subsets:
        path = f'../../../results/afrisenti/xlm-roberta-base/{sub}/'
        if os.path.exists(path+'all_results.json'):
            with open(path+'all_results.json') as json_file:
                data = json.load(json_file)
                train_samples = data['train_samples']
                eval_samples = data['eval_samples']
                predict_samples = data['predict_samples']
                df['Train'][sub] = train_samples
                df['Dev'][sub] = eval_samples
                df['Test'][sub] = predict_samples
                df['Task Train'][sub] = train_samples+eval_samples+predict_samples
        if sub == 'multilingual':
            new_path = f"../SubtaskB/dev/{sub}_dev.tsv"
            new_path1 = f"../SubtaskB/test/{sub}_test_participants.tsv"
            df['Task'][sub] = 'Subtask B'
        elif sub in ['or','tg']:
            new_path = f"../SubtaskC/dev_gold/{sub}_dev_gold_label.tsv"
            new_path1 = f"../SubtaskC/test/{sub}_test_participants.tsv"
            df['Task'][sub] = 'Subtask C'
        else:
            new_path = f"../SubtaskA/dev/{sub}_dev.tsv"
            new_path1 = f"../SubtaskA/test/{sub}_test_participants.tsv"
            df['Task'][sub] = 'Subtask A'
        if os.path.exists(new_path):
            with open(new_path) as f:
                lines = f.readlines()
                df['Task Dev'][sub] = len(lines)
        if os.path.exists(new_path1):
            with open(new_path1) as f:
                lines = f.readlines()
                df['Task Test'][sub] = len(lines)
    first_column = df.pop('Subsets')
    df.insert(0, 'Subsets', first_column)
    print(df)
    to_sheets(df,'Data Details')

def format_sheet(worksheet):
    rule = ConditionalFormatRule(
        ranges=[GridRange.from_a1_range('B2:L17', worksheet)],
        booleanRule=BooleanRule(
            condition=BooleanCondition('CUSTOM_FORMULA', ['=B2=MAX($B2:$L2)']),
            format=CellFormat(textFormat=textFormat(bold=True))
        )
    )
    rules = get_conditional_format_rules(worksheet)
    rules.clear()
    rules.append(rule)
    rules.save()


def to_sheets(df_test,worksheet_title):
    # Open an existing spreadsheet
    gc = gspread.service_account()
    sh = gc.open_by_url('https://docs.google.com/spreadsheets/d/1wq6Yaddnj2iSjcxpFAmjEj2ct_YREoBxONTZdIE-R3s/edit#gid=0')

    # Read a worksheet and create it if it doesn't exist
    try:
        worksheet = sh.worksheet(worksheet_title)
    except gspread.WorksheetNotFound:
        worksheet = sh.add_worksheet(title=worksheet_title, rows=100, cols=100)

    # Write a test DataFrame to the worksheet
    set_with_dataframe(worksheet, df_test)
    format_sheet(worksheet)

if __name__ == '__main__':
    get_results()
    get_epochs()
    get_dev_results()
    data_details()
