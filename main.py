import torch
from models.PCA import PCA
from models.FF import FF
from models.IPCA import IPCA
from models.CA import CA0, CA1, CA2, CA3

import gc
import argparse
import pandas as pd
import numpy as np
import time
import json
from tqdm import tqdm
from utils import *
from analysis import *
import matplotlib.pyplot as plt
from itertools import product

import warnings
warnings.filterwarnings('ignore')



def model_inference_and_predict(model):
    """
    Inference and Prediction of non NN models:
    Returns: model.name_inference.csv & model.name_inference.csv saved in path 'results'
    """
    mon_list = pd.read_pickle('data/mon_list.pkl')
    test_mons = mon_list.loc[mon_list >= model.test_period[0]]
    inference_result = []
    predict_result = []
    T_bar = tqdm(test_mons.groupby(test_mons.apply(lambda x: x//10000)), colour='red', desc=f'{model.name} Inferencing & Predicting')
    
    for g in T_bar: # rolling train
        T_bar.set_postfix({'Year': g[0]})
        model.train_model()
        
        for m in g[1].to_list():
            inference_result.append(model.inference(m))
            predict_result.append(model.predict(m))
        # model refit (change train period and valid period)
        model.refit()

    inference_result = pd.DataFrame(inference_result, index=test_mons, columns=CHARAS_LIST)
    inference_result.to_csv(f'results/inference/{model.name}_inference.csv')
    
    predict_result = pd.DataFrame(predict_result, index=test_mons, columns=CHARAS_LIST)
    predict_result.to_csv(f'results/predict/{model.name}_predict.csv')
    
    
    
def model_inference_and_predict_CA(model):
    """
    Inference and Prediction of NN models:
    Returns: model.name_inference.csv & model.name_inference.csv saved in path 'results'
    """
    model = model.to('cuda')
    mon_list = pd.read_pickle('data/mon_list.pkl')
    test_mons = mon_list.loc[mon_list >= model.test_period[0]]
    inference_result = pd.DataFrame()
    predict_result = pd.DataFrame()
    T_bar = tqdm(test_mons.groupby(test_mons.apply(lambda x: x//10000)), colour='red', desc=f'{model.name} Inferencing & Predicting')
    
    stock_index = pd.Series(dtype=np.int64)
    for g in T_bar: # rolling train
        T_bar.set_postfix({'Year': g[0]})

        model.reset_weight()
        model.release_gpu()
        # release GPU memory
        for _ in range(6): # call function multiple times to clear the cuda cache
            torch.cuda.empty_cache()
            
        train_loss, val_loss = model.train_model()
        # plot loss
        plt.plot(train_loss, label='train_loss')
        plt.plot(val_loss, label='val_loss')
        plt.legend()
        plt.savefig(f'results/train_loss/{model.name}_loss_{g[0]}.png')
        plt.close()

        for m in g[1].to_list():
            m_stock_index, _, _, _ = model._get_item(m)
            stock_index = pd.concat([stock_index, pd.Series(m_stock_index)]).drop_duplicates().astype(int)
            inference_R = model.inference(m) # return (N, 1)
            predict_R = model.predict(m) # reutrn (N, 1)

            # move inference_R and predict_R to cpu
            inference_R = inference_R.cpu().detach().numpy()
            predict_R = predict_R.cpu().detach().numpy()

            inference_R = pd.DataFrame(inference_R, index=m_stock_index, columns=[m])
            predict_R = pd.DataFrame(predict_R, index=m_stock_index, columns=[m])

            inference_result = pd.concat([inference_result.reset_index(drop=True), inference_R.reset_index(drop=True)], axis=1) # (N, T)
            predict_result = pd.concat([predict_result.reset_index(drop=True), predict_R.reset_index(drop=True)], axis=1) # (N, T)

            # DEBUG:
            # save inference_R and predict_R to csv
            # inference_result.to_csv(f'temp/{model.name}_inference_stock_{m}.csv')
            # predict_result.to_csv(f'temp/{model.name}_predict_stock_{m}.csv')
            
        # refit: change train period and valid period
        model.refit()

    inference_result = pd.DataFrame(inference_result.values.T, index=test_mons, columns=CHARAS_LIST)
    inference_result.to_csv(f'results/inference/{model.name}_inference.csv')
    
    predict_result = pd.DataFrame(predict_result.values.T, index=test_mons, columns=CHARAS_LIST)
    predict_result.to_csv(f'results/predict/{model.name}_predict.csv')

    # GC: release RAM memory(model)
    del model
    gc.collect()
    return inference_result, predict_result



def model_selection(model_type, model_K, omit_char=[]):
    assert model_type in ['FF', 'PCA', 'IPCA', 'CA0', 'CA1', 'CA2', 'CA3'], f'No Such Model: {model_type}'
    
    if model_type == 'FF':
        return {
            'name': f'FF_{model_K}',
            'omit_char': '',
            'model': FF(K=model_K)
        } 
            
    elif model_type == 'PCA':
        return {
            'name': f'PCA_{model_K}',
            'omit_char': omit_char,
            'model': PCA(K=model_K, omit_char=omit_char)
        } 
        
    elif model_type == 'IPCA':
        return {
            'name': f'IPCA_{model_K}',
            'omit_char': omit_char,
            'model': IPCA(K=model_K, omit_char=omit_char)
        } 
        
    elif model_type == 'CA0':
        return {
            'name': f'CA0_{model_K}',
            'omit_char': omit_char,
            'model': CA0(hidden_size=model_K, lr=CA_LR, omit_char=omit_char)
        } 
            
    elif model_type == 'CA1':
        return {
            'name': f'CA1_{model_K}',
            'omit_char': omit_char,
            'model': CA1(hidden_size=model_K, dropout=CA_DR, lr=CA_LR, omit_char=omit_char)
        } 
    
    elif model_type == 'CA2':
        return {
            'name': f'CA2_{model_K}',
            'omit_char': omit_char,
            'model': CA2(hidden_size=model_K, dropout=CA_DR, lr=CA_LR, omit_char=omit_char)
        } 
        
    else:
        return {
            'name': f'CA3_{model_K}',
            'omit_char': omit_char,
            'model': CA3(hidden_size=model_K, dropout=CA_DR, lr=CA_LR, omit_char=omit_char)
        } 
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', type=str, default='FF PCA IPCA CA0 CA1 CA2 CA3')
    parser.add_argument('--K', type=str, default='1 2 3 4 5 6')
    parser.add_argument('--omit_char', type=str, default='')

    args = parser.parse_args()
        
    models_name = []
    R_square = []
    for g in product(args.Model.split(' '), args.K.split(' '), args.omit_char.split(' ')):
        model = model_selection(g[0], int(g[1]))
        print(f"Model: {model['name']}")
        models_name.append(model['name'])

        if model['name'].split('_')[0][:-1] == 'CA':
            print('model_inference_and_predict_CA')
            model_inference_and_predict_CA(model['model'])    
        else:
            model_inference_and_predict(model['model'])
        print('name : ', model['name'])
        gc.collect()    
        # TODO: unknown typr for calculate R2
        # R_square.append(calculate_R2(model['model'], model['name'].split('_')[0][:-1]))
        R_square.append(calculate_R2(model['model'], 'inference'))
        if len(model['omit_char']):
            alpha_plot(model['model'], model['name'].split('_')[0][:-1], save_dir='alpha_imgs')

        del model

    # save R_square to json
    p = time.localtime()
    time_str = f"{p.tm_year}-{p.tm_mon}-{p.tm_mday}-{p.tm_hour}-{p.tm_min}-{p.tm_sec}"
    filename = f"R_squares/{time_str}.json"
    obj = {
        "models": [],
        'omit_char': [],
        "R2": R_square,
    }

    with open(filename, "w") as out_file:
        json.dump(obj, out_file)
    