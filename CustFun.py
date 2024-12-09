import numpy as np
from bayes_opt import BayesianOptimization
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score,root_mean_squared_log_error,mean_pinball_loss
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import pandas  as pd
import pickle
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols 

def ipfGLCoefficient(bwt:float,is_men:bool)->float:
    if is_men:
        A = 1236.25115
        B = 1449.21864
        C = 0.01644
    else:
        A = 758.63878
        B = 949.31382
        C = 0.02435

    return round(100/(A-B*np.e**(-C*bwt)),4)

def getWc(grpBtw:list[float])->dict[str:int]:
    return {'wc_upper':int(np.floor(np.max(grpBtw))),'wc_lower':int(np.ceil(np.min(grpBtw)))}

def getTeamLabel(team:str):
    dic={'KAZ': 0,'RUS':  1,'USA':  2,'TPE':  3,'NOR':  4,'JPN':  5,'CZE':  6,'UKR':  7,'SWE':  8,'GBR':  9,
        'FRA': 10,'POL': 11,'CAN': 12,'RSA': 13,'GER': 14,'LUX': 15,'HUN': 16,'VEN': 17,'BRA': 18,'ISV': 19,
        'BEL': 20,'ECU': 21,'MGL': 22,'FIN': 23,'ITA': 24,'ALG': 25,'AUT': 26,'COL': 27,'EGY': 28,'ISL': 29,
        'CRC': 30,'UZB': 31,'NED': 32,'DEN': 33,'PHI': 34,'SVK': 35,'INA': 36,'PUR': 37,'AUS': 38,'NZL': 39}
    return dic[team]

def getYearLabel(year:int):
    dic={2001: 0, 2005: 1, 2009:  2, 2013:  3, 2017:  4, 2022:  5}
    return dic[year]

def getGenderLabel(gender:str):
    dic = {'men':0,'women':1}
    return dic[gender]

def get_outlier_index(data:pd.DataFrame,feature:str,iqr_scale=1.5,return_region = False):

    ary = data[feature].to_numpy().ravel()
    q25,q75 = np.quantile(ary,[0.25,0.75])
    IQR = q75-q25
    
    idx = np.where(((ary<(q25-IQR*iqr_scale)) | (ary>(q75+IQR*iqr_scale))))[0]
    if return_region:
        return data.iloc[idx].index.to_list(),[q25-IQR*iqr_scale,q75+IQR*iqr_scale]
    else:
        return data.iloc[idx].index.to_list()
    
def n_way_anova(data:pd.DataFrame,factors:list,targets:list):

    lst_factors_for_formula = [ f'C({fa})' for fa in factors ]
    s = ' + '.join(lst_factors_for_formula)
    ways = len(factors)
            
    lst_tables = []
    for f in targets:
        df = data[factors+[f]].copy()

        moore_lm = ols(f'{f} ~ {s}',data=df).fit()
        table = sm.stats.anova_lm(moore_lm, typ=ways)
        # lst.append(table)
        d = table.loc[lst_factors_for_formula,['F', 'PR(>F)']]
        d = d.rename(columns={'PR(>F)':'p_value','F':'f_value'})
        d.index = factors

        d = d[['p_value','f_value']].T
        d['type'] = d.index
        d['feature'] = d.index
        d.index = [f,'']
        d = d[['type']+factors]
        lst_tables+=[d]
    return pd.concat(lst_tables)

def neg_root_mean_squared_error(y_true,y_pred):
    return -root_mean_squared_error(y_true,y_pred)

def dropOutlierAfIdx(data:pd.DataFrame,feature:str,iqr_scale=1.5):

    ary = data[feature].to_numpy().ravel()
    q25,q75 = np.quantile(ary,[0.25,0.75])
    IQR = q75-q25
    
    idx = np.where(((ary>=(q25-IQR*iqr_scale)) & (ary<=(q75+IQR*iqr_scale))))[0]

    return idx

def dropOutlier(x_data:pd.DataFrame,y,fatures:list):
    for f in fatures:
        idx = dropOutlierAfIdx(x_data,f,iqr_scale=2)
        x_data = x_data.iloc[idx]
        y = y[idx]
    return x_data,y

def dropOutlier(x_data:pd.DataFrame,y,fature:str):
    idx = np.where(x_data[fature]==0)[0]
    x_data = x_data.drop(columns=fature).iloc[idx]
    y = y[idx]
    return x_data,y


def scaleChange(data:pd.DataFrame,ary,fature,revert=False):
    scale = data[fature].mean()/data[fature]

    if revert:
        ary = ary * scale
    else:
        ary = ary / scale
    
    return ary

def stratified_samples(data:pd.DataFrame,group_feature:list,cv=5):

    ary_ori_sample_idx = []
    subsample_max_len = -1

    lst = []
    lst_cv_index = []
    for _ in range(cv):
        lst.append([])
        lst_cv_index.append({'train':[],'test':[]})

    lst_subsample_len = []
    subsample_count = 0

    #收集分層的樣本的位置和每個分層樣本中的數量以及最大數量
    for _,d in data.groupby(group_feature):

        ary_ori_sample_idx.append(d.sample(frac=1).index.to_list())

        currr_len = len(d)

        lst_subsample_len.append(currr_len)
        subsample_count+=1

        if subsample_max_len<currr_len:
            subsample_max_len = currr_len

    ary_ori_sample_idx = np.array(ary_ori_sample_idx,dtype=object)

    #以掃每個row的方式做分層抽樣和交叉驗證分組
    for i in range(0,subsample_max_len):  
        for j in range(subsample_count):
            idx = ary_ori_sample_idx[j]

            if i<lst_subsample_len[j]:

                lst[i%cv].append(idx[i])

    #組成交叉驗證測試和訓練位置
    for i in range(cv):     
        lst_cv_index[i]['test']=np.array(lst[i])
        for j in range(cv):
            if j!=i:
                lst_cv_index[i]['train']+=lst[j]
        
        lst_cv_index[i]['train'] = np.array(lst_cv_index[i]['train'])

    del ary_ori_sample_idx,lst

    return lst_cv_index

def mergeParams(params1:dict,params2:dict)->dict:
    dic = params1.copy()
    for key,val in params2.items():
        dic[key] = val
    return dic
 
def setParams(trial: optuna.Trial,params,model_params):

    for key,val in params.items():
        lst = val.split(',')
        if lst[0]=='int':
            if lst[1]=='cat':
                model_params[key] = trial.suggest_categorical(key, [int(i)for i in lst[2:-1]])
            elif lst[1]=='only':
                model_params[key] =int(lst[2])
            else:
                model_params[key] = trial.suggest_int(key, int(lst[1]), int(lst[2]))
        elif lst[0]=='float':
            if lst[1]=='cat':
                model_params[key] = trial.suggest_categorical(key, [float(i)for i in lst[2:-1]])
            elif lst[1]=='only':
                model_params[key] =float(lst[2])
            else:
                model_params[key] = trial.suggest_float(key, float(lst[1]), float(lst[2]))
        else:
            if lst[1]=='only':
                model_params[key] =lst[2]
            else:
                model_params[key] = trial.suggest_categorical(key, lst[1:])

def pinball_loss(alpha,y_test,y_pred):
    def _pinball_loss(alpha,y_test,y_pred):
        if y_pred<=y_test:
            return alpha*(y_test-y_pred)**2
        else:
            return (1-alpha)*(y_pred-y_test)**2
        
    return np.sum([_pinball_loss(alpha,y_test[i],y_pred[i]) for i in range(len(y_test))])

def quantile_mse_q95(y_true,y_pred):
    grad = -2*0.95*y_true/len(y_true)+y_pred/len(y_pred)
    hess = (1/len(y_true))*np.ones(len(y_true))
    
    return grad,hess

def quantile_mse_q05(y_true,y_pred):
    grad = -2*0.05*y_true/len(y_true)+y_pred/len(y_pred)
    hess = (1/len(y_true))*np.ones(len(y_true))

    return grad,hess

def getScore(metric,y_test,y_pred):

    score = 0

    if metric=="rmse":
        score = root_mean_squared_error(y_test, y_pred)
    elif metric=="mse" :
        score = mean_squared_error(y_test, y_pred)
    elif metric=="mae":
        score = mean_absolute_error(y_test, y_pred)
    elif metric=="mape":
        score = mean_absolute_percentage_error(y_test, y_pred)   
    elif metric=="r2_score":
        score = r2_score(y_test, y_pred)

    return score

def progress_callback(study, trial):
    if study.best_trial.number == trial.number:
        print(f"[{trial.number}]Value:{trial.value},Params:{trial.params}")

def LGBMRegHyperTuningUsageOptunaOB(data:pd.DataFrame,x_cols:list,y_col:list,
                                    dynamic_params:dict,static_prams:dict,
                                    n_trials,
                                    categorical_feature='auto',
                                    optimize_n_job = 1,
                                    metric = 'r2_score',
                                    is_maximize = True,
                                    study_path = ""):
    '''
    dynamic_params設定方式，以SVM超參數為例:\n
    case1: 型態後面加上cat，代表超參數是在範圍內選值 \n
    params={'gamma':'float,0,5','C':'float,1,500'}\n

    case2: 型態後面加上cat，代表超參數以類別變數形式選值 \n
    params={'gamma':'float,cat,0,5','C':'float,cat,1,500'}\n

    case3: str型態超參數以直接類別變數形式選值 \n
    params={'kernel':'str,rbf,poly,linear'}\n\n
    
    static_prams:不會經過挑選，直接餵值給model\n
    '''

    if is_maximize: 
        direction = 'maximize' 
    else: 
        direction='minimize'

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def hyperTuning(trial: optuna.Trial) -> float:
        _params = {}

        #將params轉換成符合optuna調參格式
        setParams(trial,dynamic_params,_params)

        for key,val in static_prams.items(): _params[key] = val

        X = data[x_cols]
        y = data[y_col].to_numpy().ravel()

        scores = []
        kf = KFold(n_splits=5,shuffle=True)
        for _, (train_index, val_index) in enumerate(kf.split(y)):

            X_train = X.iloc[train_index].values
            y_train = y[train_index]
            X_val  = X.iloc[val_index].values
            y_val  = y[val_index]

            lgbm = LGBMRegressor(**_params)
            lgbm.fit(X_train, y_train,eval_set=(X_val,y_val),eval_metric=lgbm.get_params()['metric'],categorical_feature=categorical_feature)

            scores.append(getScore(metric,y_val,lgbm.predict(X_val))) 
            
        return np.mean(scores)
    
    if study_path!='':

        if os.path.exists(study_path):
            with open(study_path, "rb") as f:
                study = pickle.load(f)
                print('load:',study_path)
        else:
            study = optuna.create_study(direction=direction)

        study.optimize(hyperTuning, n_trials=n_trials,n_jobs=optimize_n_job,show_progress_bar=True,callbacks=[progress_callback])

        with open(study_path, "wb") as f:
            pickle.dump(study, f)
            print('save:',study_path)
    else:
                
        study = optuna.create_study(direction=direction)
        study.optimize(hyperTuning, n_trials=n_trials,n_jobs=optimize_n_job,show_progress_bar=True,callbacks=[progress_callback])

    return study

def LBGMRegTrain(data:pd.DataFrame,x_cols:list,y_col:list,
                                    dynamic_params:dict,static_prams:dict,
                                    n_trials,
                                    categorical_feature='auto',
                                    optimize_n_job = 1,
                                    metric = 'mape',
                                    is_maximize = False,
                                    study_path1 = "",study_path2 = ""):
    '''
    dynamic_params設定方式，以SVM超參數為例:\n
    case1: 型態後面加上cat，代表超參數是在範圍內選值 \n
    params={'gamma':'float,0,5','C':'float,1,500'}\n

    case2: 型態後面加上cat，代表超參數以類別變數形式選值 \n
    params={'gamma':'float,cat,0,5','C':'float,cat,1,500'}\n

    case3: str型態超參數以直接類別變數形式選值 \n
    params={'kernel':'str,rbf,poly,linear'}\n\n
    
    static_prams:不會經過挑選，直接餵值給model\n
    '''
    lgbm_study = LGBMRegHyperTuningUsageOptunaOB(data,x_cols,y_col,dynamic_params,static_prams,
                                                     n_trials,optimize_n_job=optimize_n_job,
                                                     study_path=study_path1,
                                                     metric = metric,
                                                     is_maximize = is_maximize,
                                                     categorical_feature = categorical_feature)
    
    params = mergeParams(lgbm_study.best_params,static_prams)
    params.pop('early_stopping_rounds',None)
    
    #用預測值當作模型特徵模型的特徵
    model = LGBMRegressor(**params)
    model.fit(data[x_cols],data[y_col])
    data['last_pred'] = model.predict(data[x_cols])

    _ = LGBMRegHyperTuningUsageOptunaOB(data,x_cols+['last_pred'],y_col,dynamic_params,static_prams,
                                                     n_trials,optimize_n_job=optimize_n_job,
                                                     study_path=study_path2,
                                                     metric = metric,
                                                     is_maximize = is_maximize,
                                                     categorical_feature = categorical_feature)
    

def LGBMQRegHyperTuningUsageOptunaOB(data:pd.DataFrame,x_cols:list,y_col:list,
                                    dynamic_params:dict,static_prams:dict,
                                    n_trials,
                                    categorical_feature='auto',
                                    optimize_n_job = 1,
                                    study_path = ""):
    '''
    dynamic_params設定方式，以SVM超參數為例:\n
    case1: 型態後面加上cat，代表超參數是在範圍內選值 \n
    params={'gamma':'float,0,5','C':'float,1,500'}\n

    case2: 型態後面加上cat，代表超參數以類別變數形式選值 \n
    params={'gamma':'float,cat,0,5','C':'float,cat,1,500'}\n

    case3: str型態超參數以直接類別變數形式選值 \n
    params={'kernel':'str,rbf,poly,linear'}\n\n
    
    static_prams:不會經過挑選，直接餵值給model\n
    '''
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def hyperTuning(trial: optuna.Trial) -> float:
        _params = {}

        #將params轉換成符合optuna調參格式
        setParams(trial,dynamic_params,_params)

        for key,val in static_prams.items(): _params[key] = val

        X = data[x_cols]
        y = data[y_col].to_numpy().ravel()

        scores = []
        kf = KFold(n_splits=5,shuffle=True)
        for _, (train_index, val_index) in enumerate(kf.split(y)):

            X_train = X.iloc[train_index].values
            y_train = y[train_index]
            X_val  = X.iloc[val_index].values
            y_val  = y[val_index]

            lgbm = LGBMRegressor(**_params)
            lgbm.fit(X_train, y_train,eval_set=(X_val,y_val),categorical_feature=categorical_feature)

            y_pred = lgbm.predict(X_val)
            scores.append(mean_pinball_loss(y_val, y_pred,alpha=_params['alpha']))
            # scores.append(quantile_loss(lgbm.get_params()['alpha'],y_val, y_pred))

        return np.mean(scores)
    
    if study_path!='':

        if os.path.exists(study_path):
            with open(study_path, "rb") as f:
                study = pickle.load(f)
                print('load:',study_path)
        else:
            # study = optuna.create_study(direction='maximize')
            study = optuna.create_study(direction='minimize')

        study.optimize(hyperTuning, n_trials=n_trials,n_jobs=optimize_n_job,show_progress_bar=True,callbacks=[progress_callback])

        with open(study_path, "wb") as f:
            pickle.dump(study, f)
            print('save:',study_path)
    else:
        
        study = optuna.create_study(direction='minimize')
        study.optimize(hyperTuning, n_trials=n_trials,n_jobs=optimize_n_job,show_progress_bar=True,callbacks=[progress_callback])

    return study

def LBGMQRegTrain(data:pd.DataFrame,x_cols:list,y_col:list,
                                    dynamic_params:dict,static_prams:dict,
                                    n_trials,
                                    categorical_feature='auto',
                                    optimize_n_job = 1,
                                    study_path1 = "",study_path2 = ""):
    '''
    dynamic_params設定方式，以SVM超參數為例:\n
    case1: 型態後面加上cat，代表超參數是在範圍內選值 \n
    params={'gamma':'float,0,5','C':'float,1,500'}\n

    case2: 型態後面加上cat，代表超參數以類別變數形式選值 \n
    params={'gamma':'float,cat,0,5','C':'float,cat,1,500'}\n

    case3: str型態超參數以直接類別變數形式選值 \n
    params={'kernel':'str,rbf,poly,linear'}\n\n
    
    static_prams:不會經過挑選，直接餵值給model\n
    '''
    
    lgbm_study = LGBMQRegHyperTuningUsageOptunaOB(data,x_cols,y_col,
                                                        dynamic_params,
                                                        static_prams,
                                                        n_trials,optimize_n_job=optimize_n_job,
                                                        study_path=study_path1,
                                                        categorical_feature = categorical_feature)
    
    params = mergeParams(lgbm_study.best_params,static_prams)
    params.pop('early_stopping_rounds',None)

    #用預測值當作模型特徵模型的特徵
    model = LGBMRegressor(**params)
    model.fit(data[x_cols],data[y_col])
    data['last_pred'] = model.predict(data[x_cols])

    _ = LGBMQRegHyperTuningUsageOptunaOB(data,x_cols+['last_pred'],y_col,
                                                        dynamic_params,
                                                        static_prams,
                                                        n_trials,optimize_n_job=optimize_n_job,
                                                        study_path=study_path2,
                                                        categorical_feature = categorical_feature)


def XGBRegHyperTuningUsageOptunaOB(data:pd.DataFrame,x_cols:list,y_col:list,
                                    dynamic_params:dict,static_prams,
                                    n_trials,
                                    optimize_n_job = 1,
                                    metric = 'r2_score',
                                    is_maximize = True,
                                    study_path = ""):
    '''
    dynamic_params設定方式，以SVM超參數為例:\n
    case1: 型態後面加上cat，代表超參數是在範圍內選值 \n
    params={'gamma':'float,0,5','C':'float,1,500'}\n

    case2: 型態後面加上cat，代表超參數以類別變數形式選值 \n
    params={'gamma':'float,cat,0,5','C':'float,cat,1,500'}\n

    case3: str型態超參數以直接類別變數形式選值 \n
    params={'kernel':'str,rbf,poly,linear'}\n\n
    
    static_prams:不會經過挑選，直接餵值給model\n
    '''
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if is_maximize: 
        direction = 'maximize' 
    else: 
        direction='minimize'
    
    def hyperTuning(trial: optuna.Trial) -> float:
        _params = {}

        #將params轉換成符合optuna調參格式
        setParams(trial,dynamic_params,_params)

        for key,val in static_prams.items(): _params[key] = val
        
        X = data[x_cols]
        y = data[y_col].to_numpy().ravel()
        
        scores = []
        kf = KFold(n_splits=5,shuffle=True)
        for _, (train_index, test_index) in enumerate(kf.split(y)):

            X_train = X.iloc[train_index].values
            y_train = y[train_index]
            X_val  = X.iloc[test_index].values
            y_val  = y[test_index]

            
            xgb = XGBRegressor(**_params)
            xgb.fit(X_train,y_train,verbose=False)

            scores.append(getScore(metric,y_val,xgb.predict(X_val))) 
            
        return np.mean(scores)

    if study_path!='':

        if os.path.exists(study_path):
            with open(study_path, "rb") as f:
                study = pickle.load(f)
                print('load:',study_path)
        else:
            study = optuna.create_study(direction=direction)

        study.optimize(hyperTuning, n_trials=n_trials,n_jobs=optimize_n_job,show_progress_bar=True,callbacks=[progress_callback])

        with open(study_path, "wb") as f:
            pickle.dump(study, f)
            print('save:',study_path)
    else:
        study = optuna.create_study(direction=direction)
        study.optimize(hyperTuning, n_trials=n_trials,n_jobs=optimize_n_job,show_progress_bar=True,callbacks=[progress_callback])
    
    return study

def XGBRegTrain(data:pd.DataFrame,x_cols:list,y_col:list,
                                    dynamic_params:dict,static_prams:dict,
                                    n_trials,
                                    optimize_n_job = 1,
                                    metric = 'mape',
                                    is_maximize = False,                                   
                                    study_path1 = "",study_path2 = ""):
    '''
    dynamic_params設定方式，以SVM超參數為例:\n
    case1: 型態後面加上cat，代表超參數是在範圍內選值 \n
    params={'gamma':'float,0,5','C':'float,1,500'}\n

    case2: 型態後面加上cat，代表超參數以類別變數形式選值 \n
    params={'gamma':'float,cat,0,5','C':'float,cat,1,500'}\n

    case3: str型態超參數以直接類別變數形式選值 \n
    params={'kernel':'str,rbf,poly,linear'}\n\n
    
    static_prams:不會經過挑選，直接餵值給model\n
    '''
    
    xgb_study = XGBRegHyperTuningUsageOptunaOB(data,x_cols,y_col,
                                               dynamic_params,static_prams,
                                               n_trials,optimize_n_job=optimize_n_job,
                                               metric = metric,
                                               is_maximize=is_maximize,
                                               study_path=study_path1)
    
    params = mergeParams(xgb_study.best_params,static_prams)
    
    #用預測值當作模型特徵模型的特徵
    model = XGBRegressor(**params)
    model.fit(data[x_cols],data[y_col])
    data['last_pred'] = model.predict(data[x_cols])

    _ = XGBRegHyperTuningUsageOptunaOB(data,x_cols+['last_pred'],y_col,
                                        dynamic_params,static_prams,
                                        n_trials,optimize_n_job=optimize_n_job,
                                        metric = metric,
                                        is_maximize=is_maximize,
                                        study_path=study_path2)

def load_study(study_path):
    study = None
    with open(study_path, "rb") as f:
        study = pickle.load(f)
    return study

def get_coverage(real,pred,error,k=1):
    upper = pred+k*error
    lower = pred-k*error
    return np.mean( ((real<=upper) & (real>=lower)) )*100

def get_quantile_coverage(real,upper_pred,lower_pred,upper_error,lower_error,upper_k=0,lower_k=0):

    return np.mean( (real<=upper_pred+upper_error*upper_k) & (real>=lower_pred-lower_error*lower_k) )*100

def doule_train_pred(train,test,x_cols,y_col,
                     static_prams:dict,study_path1,study_path2,
                     name,is_lgbm=True,categorical_feature = [0,1,2,3,4],quantile_alpha=0):
    
    study1:optuna.study = load_study(study_path1)
    study2:optuna.study = load_study(study_path2)
    
    X_train = train[x_cols].copy()
    y_train = train[y_col].to_numpy().ravel()
    X_test = test[x_cols].copy()
    y_test = test[y_col].to_numpy().ravel()
    
    if is_lgbm:     
        model1 = LGBMRegressor(**mergeParams(static_prams,study1.best_params))
        model1.fit(X_train,y_train,categorical_feature=categorical_feature)
        X_train['last_pred'] = model1.predict(X_train)
        X_test['last_pred'] = model1.predict(X_test)

        model2 = LGBMRegressor(**mergeParams(static_prams,study2.best_params))
        model2.fit(X_train[x_cols+['last_pred']],y_train,categorical_feature=categorical_feature)
    else:
        model1 = XGBRegressor(**mergeParams(static_prams,study1.best_params))
        model1.fit(X_train,y_train)
        X_train['last_pred'] = model1.predict(X_train)
        X_test['last_pred'] = model1.predict(X_test)
        
        model2 = XGBRegressor(**mergeParams(static_prams,study2.best_params))
        model2.fit(X_train[x_cols+['last_pred']],y_train)
    
    train_pred = model2.predict(X_train[x_cols+['last_pred']])
    test_pred = model2.predict(X_test[x_cols+['last_pred']])

    if quantile_alpha ==0:
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae  = mean_absolute_error(y_test, test_pred)


        df = pd.DataFrame(data={
                                "model":[name,name],
                                "r2_score":[r2_score(y_train, train_pred),r2_score(y_test, test_pred)],
                                "mae":[train_mae,test_mae],
                                "1倍誤差覆蓋率":[get_coverage(y_train,train_pred,train_mae),get_coverage(y_test,test_pred,test_mae)],
                                "2倍誤差覆蓋率":[get_coverage(y_train,train_pred,train_mae,k=2),get_coverage(y_test,test_pred,test_mae,k=2)],
                                "2.5倍誤差覆蓋率":[get_coverage(y_train,train_pred,train_mae,k=2.5),get_coverage(y_test,test_pred,test_mae,k=2.5)],
                                },index=['train','test'])
    else:
        
        train_mp = mean_pinball_loss(y_train, train_pred,alpha=quantile_alpha)
        test_pb  = mean_pinball_loss(y_test, test_pred,alpha=quantile_alpha)

        df = pd.DataFrame(data={
                        "model":[name,name],
                        "r2_score":[r2_score(y_train, train_pred),r2_score(y_test, test_pred)],
                        "pinball_loss":[train_mp,test_pb],
                        "1倍誤差覆蓋率":[get_coverage(y_train,train_pred,train_mp,k=1),get_coverage(y_test,test_pred,test_pb,k=1)],
                        "2倍誤差覆蓋率":[get_coverage(y_train,train_pred,train_mp,k=2),get_coverage(y_test,test_pred,test_pb,k=2)],
                        "2.5倍誤差覆蓋率":[get_coverage(y_train,train_pred,train_mp,k=2.5),get_coverage(y_test,test_pred,test_pb,k=2.5)]
                        },index=['train','test'])  

    return train_pred,test_pred,df


def getDescribe(scores,labels,asc=False):

    dic = {}
    for i in range(len(scores)):
        s = scores[i]
        if asc:
            dic[labels[i]] = [len(s),round(np.min(s),4),list(s).index(np.min(s))+1,round(np.max(s),4),list(s).index(np.max(s))+1]
        else:
            dic[labels[i]] = [len(s),round(np.max(s),4),list(s).index(np.max(s))+1,round(np.min(s),4),list(s).index(np.min(s))+1]
    df = pd.DataFrame(data=dic,index=['Exp. Rounds','Best Score','Best Occ. Round','Worst Score','Worst Occ. Round'])
    df = df.T
    df['Exp. Rounds'] = df['Exp. Rounds'].astype(int)
    df['Best Occ. Round'] = df['Worst Occ. Round'].astype(int)
    return df.sort_values('Best Score',ascending=asc)

def getHyperParamScatter(study:optuna.study):
    
    dicParams = {}
    dicParams['scores'] = []

    #取出HyperParam和score的數值
    for t in study.get_trials():
        for key,val in t.params.items(): 

            if key not in dicParams.keys():
                dicParams[key] = [val]
            else:
                dicParams[key].append(val)

        dicParams['scores'].append(t.values[0])
    
    #決定score顯示的尺規和範圍
    min_scores = np.min(dicParams['scores'])
    xtick = (np.max(dicParams['scores']) - min_scores)/5
    lst_xticks = [min_scores-xtick] + [min_scores + xtick*i for i in range(1,5)]

    #秀出HyperParam與score的分佈情況
    df = pd.DataFrame(data=dicParams)
    for p in dicParams.keys():
        if p !='scores':
            df.plot.scatter(x='scores',y=p,s=2,alpha=0.5)
            plt.xticks(lst_xticks)
            plt.xlim([min_scores - xtick, min_scores + xtick*4])
            plt.xlabel('scores')
            plt.ylabel(p)
            plt.show()

def resultPlot( train_true,train_pred,
                test_true,test_pred,
                lower_train_pred = None,upper_train_pred = None,
                lower_test_pred = None,upper_test_pred = None,
                x_lbl_11='Actual',y_lbl_11='Predicted',title_11="Train's Actual vs Predicted Reg. Line",
                x_lbl_12='residual vals',y_lbl_12='Feq.',title_12="Train's residual chert",
                x_lbl_21='Actual',y_lbl_21='Predicted',title_21="Test's Actual vs Predicted Reg. Line",
                x_lbl_22='residual vals',y_lbl_22='Feq.',title_22="Test's residual chert",
                figsize=(20,10)):
    
    sns.set_theme(style="darkgrid")
    _, axes = plt.subplots(nrows=2, ncols=3,figsize=figsize)

    #train
    df = pd.DataFrame({'test':train_true,'pred':train_pred,'Q05':lower_train_pred,'Q95':upper_train_pred}).sort_values('test')
    df.index = df['test']
    sns.lineplot(df['test'],color='red',ax=axes[0,0],label='real') 
    sns.scatterplot(df['pred'],alpha=0.5,ax=axes[0,0],label='pred')
    sns.scatterplot(df['Q95'], color='green',alpha=0.3,ax=axes[0,0],label='Q95') 
    sns.scatterplot(df['Q05'], color='orange',alpha=0.3,ax=axes[0,0],label='Q05') 
    
    
    axes[0,0].set_xlabel(x_lbl_11)
    axes[0,0].set_ylabel(y_lbl_11)
    axes[0,0].title.set_text(title_11)
    axes[0,0].legend()
    

    residuals = (train_true-train_pred)
    sns.histplot(x=residuals, kde=True,ax=axes[0,1])
    axes[0,1].set_xlabel(x_lbl_12)
    axes[0,1].set_ylabel(y_lbl_12)
    axes[0,1].title.set_text(title_12)

    sm.qqplot((residuals-np.mean(residuals))/np.std(residuals), line='45',ax=axes[0,2])
    axes[0,2].title.set_text("Train's residual Q-Q plot")

    #test
    df = pd.DataFrame({'test':test_true,'pred':test_pred,'Q05':lower_test_pred,'Q95':upper_test_pred}).sort_values('test')
    df.index = df['test']
    sns.lineplot(df['test'],color='red',ax=axes[1,0],label='real') 
    sns.scatterplot(df['pred'],alpha=0.8,ax=axes[1,0],label='pred')
    sns.scatterplot(df['Q95'], color='green',alpha=0.3,ax=axes[1,0],label='Q95')
    sns.scatterplot(df['Q05'], color='orange',alpha=0.3,ax=axes[1,0],label='Q05')
    
    
    axes[1,0].set_xlabel(x_lbl_21)
    axes[1,0].set_ylabel(y_lbl_21)
    axes[1,0].title.set_text(title_21)
    axes[1,0].legend()

    residuals = (test_true-test_pred)
    sns.histplot(x=(test_true-test_pred), kde=True,ax=axes[1,1])
    axes[1,1].set_xlabel(x_lbl_22)
    axes[1,1].set_ylabel(y_lbl_22)
    axes[1,1].title.set_text(title_22)

    sm.qqplot((residuals-np.mean(residuals))/np.std(residuals), line='45',ax=axes[1,2])
    axes[1,2].title.set_text("Test's residual Q-Q plot")

    plt.show()

def SvrHyperTuningUsageOB(X,y,pbounds,score_function,ob_random_state,n_iter,init_points=0,k_fold_n_splits=10):
    
    def hyperTuning(**params):
        scores = []
        kf = KFold(n_splits=k_fold_n_splits)
        for i, (train_index, test_index) in enumerate(kf.split(X)):

            X_train, X_test, y_train, y_test = X[train_index],X[test_index],y[train_index],y[test_index]
            regr = svm.SVR(**params)
            regr.fit(X_train, y_train)
            y_pred = regr.predict(X_test)

            scores.append(score_function(y_test, y_pred))
        return np.mean(scores)

    optimizer = BayesianOptimization(f=hyperTuning,pbounds=pbounds,random_state=ob_random_state,verbose=1)
    optimizer.maximize(n_iter=n_iter,init_points=init_points)

    return optimizer

def get_predict_odd_index(residual,iqr_scale=1.5):

    q25,q75 = np.quantile(residual,[0.25,0.75])
    IQR = q75-q25
    idx = np.where(((residual<(q25-IQR*iqr_scale)) | (residual>(q75+IQR*iqr_scale))))[0]

    return {'odd':idx}

def get_repeat_count(lst_odd:list,reverse=False):

    '''
    回傳數字在list中出現幾次{數字：重複次數}\n
    當reverse 是 True時相反:{重複次數；數字}
    '''
    lst = [[],[]]
    for odd in lst_odd:
        existed = False
        i = 0
        for repeat_times in range(1,len(lst)):
            if odd in lst[repeat_times]:
                i = repeat_times
                lst[i].remove(odd)
                existed = True
                break
        if existed:
            if i+1 >len(lst):
                lst[i+1].append(odd)
            else:
                lst.append([odd])
        else:
            lst[1].append(odd)

    if reverse==False:
        dic = {}
        for repeat_times in range(len(lst)):
            if len(lst[repeat_times])>0:
                for idx in lst[repeat_times]:
                    dic[idx] = repeat_times
    else:
        dic = {reapt_count:lst[reapt_count] for reapt_count in range(len(lst)) if len(lst[reapt_count])>0}

    del lst

    return dic


def get_odd_joinplot(data:pd.DataFrame,x_colum,y_colum,dic_indexs:dict[str:list[int]],kind='scatter',alpha=1,usage_hue=True,show_all = True,legend_title='case'):
    lst = []
    _data = data.copy()
    _data = _data.astype(float)
    
    if show_all==True:
        df = _data.copy()
        df[legend_title] = 'all'
        lst.append(df)
    for case,index in dic_indexs.items():
        df = _data.loc[index].astype(float)
        df[legend_title] = case
        lst.append(df)

    df = pd.concat(lst)
    if usage_hue==False:
        g = sns.jointplot(data=df, x=x_colum, y=y_colum,kind=kind,alpha=alpha)
    else:
        g = sns.jointplot(data=df, x=x_colum, y=y_colum,hue=legend_title,kind=kind,alpha=alpha)
        sns.move_legend(g.figure.axes[0], 3, bbox_to_anchor=(1.01, 1.01))
    return g

def save_dict(path,dic):
    '''
    Using pickle to save dict. \n
    format: xxx.pkl
    '''
    with open(path, 'wb') as fp:
        pickle.dump(dic, fp)
        print(f'save:{path}')

def load_dict(path):
    '''
    Using pickle to load dict. \n
    format: xxx.pkl
    '''
    dic = None
    with open(path, 'rb') as fp:
        dic = pickle.load(fp)
        print(f'load:{path}')
    return dic