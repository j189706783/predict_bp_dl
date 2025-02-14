import joblib
import pandas as pd
import json
from model import pred_output

def _predict(inputs:pd.DataFrame,deviation,scale,path1,path2):

    model1 = joblib.load(path1) 
    model2 = joblib.load(path2)

    pred1 = model1.predict(inputs[['group', 'team', 'wc', 'gender', 'rank', 'old', 'bwt', 'sq', 'ipf_gl_c','year']])
    inputs['last_pred'] = pred1
    pred2 = model2.predict(inputs) 

    upper = pred2+deviation*scale
    lower = pred2-deviation*scale

    return pred1,pred2,upper,lower

def predict(inputs:dict):

    df = pd.DataFrame({key:[val] for key,val in inputs.items()})
    for f in [ 'group','team', 'wc','gender','rank']:
        df[f] = df[f].astype('category')
                      
    params = pd.read_csv('./model/coverage_90_param.csv').to_dict()
    
    results = {}
    for name in ['lgbm','xgb','lgbm_Q05','lgbm_Q95']:
        if 'Q' in name: 
            loss = params[f'{name}_loss'][0]
        else:
            loss = params[f'{name}_mae'][0]

        scale = params[f'{name}_scale'][0]

        pred1,pred2,upper,lower = _predict(df,loss,scale,
                                            f'./model/{name}_bp_dl_1.pkl',
                                            f'./model/{name}_bp_dl_2.pkl')
        
        if name =='lgbm_Q05':
            output = pred_output(name=name,loss=loss,scale=scale,pred1=pred1,pred2=pred2,upper=0,lower=lower)
        elif name =='lgbm_Q95':
            output = pred_output(name=name,loss=loss,scale=scale,pred1=pred1,pred2=pred2,upper=upper,lower=0)
        else:
            output = pred_output(name=name,loss=loss,scale=scale,pred1=pred1,pred2=pred2,upper=upper,lower=lower)

        results[name] = output.dict()

    return results
