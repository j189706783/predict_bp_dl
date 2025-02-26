import joblib
import pandas as pd
import json
from model import pred_output

def _predict(inputs:pd.DataFrame,path1,path2,team_stay_label):

    model1 = joblib.load(path1) 
    model2 = joblib.load(path2)

    inputs.loc[inputs['team'].isin(team_stay_label)==False,['rank']]=99
    inputs.loc[inputs['rank']>=9,['rank']]=9

    pred1 = model1.predict(inputs[['year', 'team', 'wc', 'gender', 'old', 'bwt', 'sq', 'ipf_gl_c', 'rank']])
    pred2 = model2.predict(inputs[['year', 'team', 'wc', 'gender', 'old', 'bwt', 'sq', 'ipf_gl_c', 'rank']]) 
    pred3 = (pred1+pred2)/2

    return pred1,pred2,pred3

def predict(inputs:dict):

    df = pd.DataFrame({key:[val] for key,val in inputs.items()})
                      
    df_90PI = pd.read_csv('./model/90PI.csv')
    
    pred1,pred2,pred3 = _predict(df,f'./model/lgbm_bp_dl_1.pkl',f'./model/lgbm_bp_dl_2.pkl',df_90PI['team_stay_label'][0].split(','))

    model1_upper = pred1 + df_90PI['model1_upper'][0]
    model1_lower = pred1 + df_90PI['model1_lower'][0]

    model2_upper = pred2 + df_90PI['model2_upper'][0]
    model2_lower = pred2 + df_90PI['model2_lower'][0]

    mean_upper   = pred3 + df_90PI['mean_upper'][0]
    mean_lower   = pred3 + df_90PI['mean_lower'][0]

    del df,df_90PI
    
    results = {}
    results['model1'] = pred_output(name='model1',pred=pred1,upper=model1_upper,lower=model1_lower).dict()
    results['model2'] = pred_output(name='model2',pred=pred2,upper=model2_upper,lower=model2_lower).dict()
    results['mean'] = pred_output(name='mean',pred=pred3,upper=mean_upper,lower=mean_lower).dict()

    return results
