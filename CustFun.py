import numpy as np
from bayes_opt import BayesianOptimization
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score,root_mean_squared_log_error,mean_pinball_loss
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor,cv,Dataset
from xgboost import XGBRegressor
import pandas  as pd
import pickle
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols 

import joblib

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
    p25,p75 = np.quantile(ary,[0.25,0.75])
    IQR = p75-p25
    
    idx = np.where(((ary<(p25-IQR*iqr_scale)) | (ary>(p75+IQR*iqr_scale))))[0]
    if return_region:
        return data.iloc[idx].index.to_list(),[p25-IQR*iqr_scale,p75+IQR*iqr_scale]
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
    p25,p75 = np.quantile(ary,[0.25,0.75])
    IQR = p75-p25
    
    idx = np.where(((ary>=(p25-IQR*iqr_scale)) & (ary<=(p75+IQR*iqr_scale))))[0]

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

def quantile_mse_p95(y_true,y_pred):
    grad = -2*0.95*y_true/len(y_true)+y_pred/len(y_pred)
    hess = (1/len(y_true))*np.ones(len(y_true))
    
    return grad,hess

def quantile_mse_p05(y_true,y_pred):
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
        
        for key,val in static_prams.items():
            _params[key] = val

        X = data[x_cols]
        y = data[y_col]

        num_boost_round = _params.pop('n_estimators')
        train_data = Dataset(X,y)
        cv_results = cv(
                _params,
                train_data,
                num_boost_round=num_boost_round,
                nfold=5, 
                stratified=False,
                shuffle=True,
                seed=42
            )
        
        df = pd.DataFrame(cv_results)
        if 'mean' in df.columns[0]:
           return cv_results[df.columns[0]][-1]
        else:
            return cv_results[df.columns[1]][-1]

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

    data = data.copy()

    lgbm_study = LGBMRegHyperTuningUsageOptunaOB(data,x_cols,y_col,dynamic_params,static_prams,
                                                     n_trials,optimize_n_job=optimize_n_job,
                                                     study_path=study_path1,
                                                     metric = metric,
                                                     is_maximize = is_maximize)

    params = mergeParams(lgbm_study.best_params,static_prams)
    
    #用預測值當作模型特徵模型的特徵
    model = LGBMRegressor(**params,shuffle=True)
    model.fit(data[x_cols],data[y_col])

    data['last_pred'] = model.predict(data[x_cols])

    _ = LGBMRegHyperTuningUsageOptunaOB(data,x_cols+['last_pred'],y_col,dynamic_params,static_prams,
                                                     n_trials,optimize_n_job=optimize_n_job,
                                                     study_path=study_path2,
                                                     metric = metric,
                                                     is_maximize = is_maximize)
    

def LGBMQRegHyperTuningUsageOptunaOB(data:pd.DataFrame,x_cols:list,y_col:list,
                                    dynamic_params:dict,static_prams:dict,
                                    n_trials,
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
            lgbm.fit(X_train, y_train,eval_set=(X_val,y_val))

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
    
    data = data.copy()
    lgbm_study = LGBMQRegHyperTuningUsageOptunaOB(data,x_cols,y_col,
                                                        dynamic_params,
                                                        static_prams,
                                                        n_trials,optimize_n_job=optimize_n_job,
                                                        study_path=study_path1)
    
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
                                                        study_path=study_path2)


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
        

        xgb = XGBRegressor(**_params)
        xgb.fit(X,y,verbose=False)
         
        return getScore(metric,y,xgb.predict(X))

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
    
    data = data.copy()
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

def doule_train_pred(train,x_cols,y_col,
                     static_prams:dict,study_path1,study_path2,save_model_path1,save_model_path2,
                     is_lgbm=True):
    
    study1:optuna.study = load_study(study_path1)
    study2:optuna.study = load_study(study_path2)
    
    X_train = train[x_cols].copy()
    y_train = train[y_col].to_numpy().ravel()
    
    if is_lgbm:     
        model1 = LGBMRegressor(**mergeParams(static_prams,study1.best_params))
        model1.fit(X_train,y_train)
        joblib.dump(model1,save_model_path1)

        X_train['last_pred'] = model1.predict(X_train)

        model2 = LGBMRegressor(**mergeParams(static_prams,study2.best_params))
        model2.fit(X_train[x_cols+['last_pred']],y_train)

        joblib.dump(model2,save_model_path2)
    else:
        model1 = XGBRegressor(**mergeParams(static_prams,study1.best_params))
        model1.fit(X_train,y_train)
        joblib.dump(model1,save_model_path1)

        X_train['last_pred'] = model1.predict(X_train)
        

        model2 = XGBRegressor(**mergeParams(static_prams,study2.best_params))
        model2.fit(X_train[x_cols+['last_pred']],y_train)
        joblib.dump(model2,save_model_path2)


    return model1,model2


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
    df = pd.DataFrame({'test':train_true,'pred':train_pred,'P05':lower_train_pred,'P95':upper_train_pred}).sort_values('test')
    df.index = df['test']
    sns.lineplot(df['test'],color='red',ax=axes[0,0],label='real') 
    sns.scatterplot(df['pred'],alpha=0.5,ax=axes[0,0],label='pred')
    sns.scatterplot(df['P95'], color='green',alpha=0.3,ax=axes[0,0],label='P95') 
    sns.scatterplot(df['P05'], color='orange',alpha=0.3,ax=axes[0,0],label='P05') 
    
    
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
    df = pd.DataFrame({'test':test_true,'pred':test_pred,'P05':lower_test_pred,'P95':upper_test_pred}).sort_values('test')
    df.index = df['test']
    sns.lineplot(df['test'],color='red',ax=axes[1,0],label='real') 
    sns.scatterplot(df['pred'],alpha=0.8,ax=axes[1,0],label='pred')
    sns.scatterplot(df['P95'], color='green',alpha=0.3,ax=axes[1,0],label='P95')
    sns.scatterplot(df['P05'], color='orange',alpha=0.3,ax=axes[1,0],label='P05')
    
    
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

    p25,p75 = np.quantile(residual,[0.25,0.75])
    IQR = p75-p25
    idx = np.where(((residual<(p25-IQR*iqr_scale)) | (residual>(p75+IQR*iqr_scale))))[0]

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

def compute_pi(df,model,x_cols,y_col,upper=0.95,lower=0.05,bootstrap_samples=1000000):
    samples = df.sample(n=bootstrap_samples,replace=True,random_state=42)
    preds = model.predict(samples[x_cols].to_numpy())
    residual = (samples[y_col].to_numpy()-preds).tolist()
    
    pi = np.quantile(residual,[upper,lower])
    return pi[0],pi[1]

def compute_pi_by_std(df,model,x_cols,y_col,bootstrap_samples=1000000):
    samples = df.sample(n=bootstrap_samples,replace=True)
    preds = model.predict(samples[x_cols].to_numpy())
    residual = (samples[y_col].to_numpy()-preds).tolist()
    
    return np.mean(residual)+np.std(residual)*1.96,np.mean(residual)-np.std(residual)*1.96

def compute_coverage_rate(preds,trues,upper,lower):    
    return np.mean( ((preds+lower)<=trues) & (trues<=(preds+upper)) )

def compute_pi_for_baseline(df,y_col,upper=0.95,lower=0.05,bootstrap_samples=1000000):
    
    preds = df[y_col].mean()
    samples = df.sample(n=bootstrap_samples,replace=True)
    residual = (samples[y_col].to_numpy()-preds).tolist()
    
    pi = np.quantile(residual,[upper,lower])
    return pi[0],pi[1]

def compute_coverage_rate_for_baseline(df,y_col,upper,lower):
    preds = df[y_col].mean()
    return np.mean( ((preds+lower)<=df[y_col].to_numpy()) & (df[y_col].to_numpy()<=(preds+upper)) )

def metrics(models:list,df_train,df_test,X_train,y_train,X_test,y_test,x_cols,y_col):
    lst_metrics = []
    df_metrics = pd.DataFrame(index=['mae','rmse','mape','r2_score','residual_P95','residual_P05','coverage_rate'])
    for cols in ['train_baseline','train','test_baseline','test']:
        if cols == 'train_baseline':
            true = y_train
            pred = y_train.mean()
            pred = [pred for _ in range(len(y_train))]
        elif cols == 'train':
            true = y_train
            pred = np.mean([model.predict(X_train) for model in models],axis=0)

        elif cols == 'test_baseline':
            true = y_test
            pred = y_test.mean()
            pred = [pred for _ in range(len(y_test))]
        else:
            true = y_test
            pred = np.mean([model.predict(X_test) for model in models],axis=0)

        lst_metrics = [mean_absolute_error(true,pred),root_mean_squared_error(true,pred),
                          mean_absolute_percentage_error(true,pred),r2_score(true,pred)]
        
        if cols == 'train_baseline':
            upper,lower = compute_pi_for_baseline(df_train,y_col)
            coverage_rate = compute_coverage_rate_for_baseline(df_train,y_col,upper,lower)
        elif cols == 'train':
            if len(models)>1:
                lst = [compute_pi(df_train,model,x_cols,y_col) for model in models]
                upper = np.mean([lst[0][0],lst[1][0]])
                lower = np.mean([lst[0][1],lst[1][1]])
                del lst
            else:
                upper,lower = compute_pi(df_train,models[0],x_cols,y_col)
                
            coverage_rate = compute_coverage_rate(pred,true,upper,lower)
        elif cols == 'test_baseline':
            upper,lower = compute_pi_for_baseline(df_test,y_col)
            coverage_rate = compute_coverage_rate_for_baseline(df_test,y_col,upper,lower)
        else:
            if len(models)>1:
                lst = [compute_pi(df_test,model,x_cols,y_col) for model in models] 
                upper = np.mean([lst[0][0],lst[1][0]])
                lower = np.mean([lst[0][1],lst[1][1]])
                del lst
            else:
                upper,lower = compute_pi(df_test,models[0],x_cols,y_col)

            coverage_rate = compute_coverage_rate(pred,true,upper,lower)

        lst_metrics.append(upper)
        lst_metrics.append(lower)
        lst_metrics.append(coverage_rate)

        df_metrics[cols] = lst_metrics

    df_metrics.style.format("{:.6f}")
    return df_metrics

def train(params,num_boost_round,df_train,df_test,x_cols,y_col,show_result = True,using_cv=True):
    X_train,y_train,X_test,y_test = df_train[x_cols].to_numpy(),df_train[y_col].to_numpy(),df_test[x_cols].to_numpy(),df_test[y_col].to_numpy()
    train_data = Dataset(X_train, label=y_train)

    cv_results = cv(
            params,
            train_data,
            num_boost_round=num_boost_round,
            nfold=5, 
            stratified=False,
            shuffle=True,
            seed=42
        )
    
    s1 = ''
    df = pd.DataFrame(cv_results)
    if 'mean' in df.columns[0]:
        s1 = df.columns[0]
        s2 = df.columns[1]
    else:
        s1 = df.columns[1]
        s2 = df.columns[0]
    
    df = df.sort_values(s1).iloc[:5]
    df = df.sort_values(s2).iloc[:1]
    rounds = df.index.tolist()[0]

    if show_result:print('rounds','=>',rounds)

    dic_metrics = {}
    if using_cv:
        if show_result:print('cv start')
        lst = []
        kf = KFold(n_splits=5,shuffle=True,random_state=42)
        for _, (train_index, val_index) in enumerate(kf.split(y_train)):

            X_cv_train = X_train[train_index]
            y_cv_train = y_train[train_index]
            X_cv_val  = X_train[val_index]
            y_cv_val  = y_train[val_index]

            lgbm = LGBMRegressor(**params,n_estimators=rounds,shuffle=True)
            lgbm.fit(X_cv_train, y_cv_train,eval_set=(X_cv_val,y_cv_val))

            df_importances = pd.DataFrame(data={'feature':x_cols,'feature_importances':lgbm.feature_importances_})
            df_importances = df_importances.reset_index(drop=True).T
            df_metrics = metrics([lgbm],df_train.iloc[train_index],df_train.iloc[val_index],
                                 X_cv_train,y_cv_train,X_cv_val,y_cv_val,x_cols,y_col)

            lst.append(df_metrics)
        
        df_metrics = pd.concat(lst)
        lst = []
        for m in ['mae','rmse','mape','r2_score','residual_P95','residual_P05','coverage_rate']:
            lst.append(df_metrics.loc[[m]].agg(['var','mean']))
            lst[-1]['metrics'] = m 
        df_metrics = pd.concat(lst) 
        
        dic_metrics['cv_mean'] = df_metrics.loc[['mean']]
        dic_metrics['cv_mean'] = dic_metrics['cv_mean'].reset_index(drop=True)[['metrics','train_baseline','train','test_baseline','test']].rename(columns={'test':'val'})
        
        dic_metrics['cv_var'] = df_metrics.loc[['var']]
        dic_metrics['cv_var'] = dic_metrics['cv_var'].reset_index(drop=True)[['metrics','train_baseline','train','test_baseline','test']].rename(columns={'test':'val'})

        if show_result:print('cv end')

    model = LGBMRegressor(**params,n_estimators=rounds,shuffle=True)
    model.fit(df_train.drop(columns='bp_dl'),df_train['bp_dl'])

    df_importances = pd.DataFrame(data={'feature':x_cols,'feature_importances':model.feature_importances_})
    df_importances = df_importances.reset_index(drop=True).T
    
    df_metrics = metrics([model],df_train,df_test,X_train,y_train,X_test,y_test,x_cols,y_col)
    df_metrics['metrics'] = df_metrics.index
    df_metrics = df_metrics.reset_index(drop=True)

    dic_metrics['full_dataset'] = df_metrics[['metrics','train_baseline','train','test_baseline','test']]
    dic_metrics['importances'] = df_importances
    
    return model,dic_metrics

def metrics_quantile(models_P05:list,models_P95:list,X_train,y_train,X_test,y_test,ensemble='mean'):
    lst_metrics = []
    df_metrics = pd.DataFrame(index=['coverage_rate'])
    for cols in ['train_baseline','train','test_baseline','test']:
        if cols == 'train_baseline':
            P05 = np.quantile(y_train,[0.05])[0]
            P95 = np.quantile(y_train,[0.95])[0]
            coverage_rate = np.mean( (P05<=y_train) & (y_train<=P95))
        elif cols == 'train':
            if ensemble=='mean':
                P95 = np.mean([model.predict(X_train) for model in models_P95],axis=0)
                P05 = np.mean([model.predict(X_train) for model in models_P05],axis=0)
            elif ensemble=='max':
                P95 = np.max([model.predict(X_train) for model in models_P95],axis=0)
                P05 = np.max([model.predict(X_train) for model in models_P05],axis=0)
            coverage_rate = np.mean(((P05<=y_train) & (y_train<=P95)))

        elif cols == 'test_baseline':
            P95 = np.quantile(y_test,[0.95])[0]
            P05 = np.quantile(y_test,[0.05])[0]
            coverage_rate = np.mean((P05<=y_test) & (y_test<=P95))
        else:
            if ensemble=='mean':
                P95 = np.mean([model.predict(X_test) for model in models_P95],axis=0)
                P05 = np.mean([model.predict(X_test) for model in models_P05],axis=0)
            elif ensemble=='max':
                P95 = np.max([model.predict(X_test) for model in models_P95],axis=0)
                P05 = np.max([model.predict(X_test) for model in models_P05],axis=0)

            coverage_rate = np.mean(((P05<=y_test) & (y_test<=P95)))
        
        lst_metrics = [coverage_rate]
        
        df_metrics[cols] = lst_metrics

    df_metrics.style.format("{:.6f}")
    return df_metrics

def train_quantile(params_P05,num_boost_round_P05,params_P95,num_boost_round_P95,df_train,df_test,x_cols,y_col,show_result = True,using_cv=True):
    X_train,y_train,X_test,y_test = df_train[x_cols].to_numpy(),df_train[y_col].to_numpy(),df_test[x_cols].to_numpy(),df_test[y_col].to_numpy()
    train_data = Dataset(X_train, label=y_train)

    params = [params_P05,params_P95]
    num_boost_rounds = [num_boost_round_P05,num_boost_round_P95]
    for i in range(2):
        cv_results = cv(
                params[i],
                train_data,
                num_boost_round=num_boost_rounds[i],
                nfold=5, 
                stratified=False,
                shuffle=True,
                seed=42
            )
        
        s1 = ''
        df = pd.DataFrame(cv_results)
        if 'mean' in df.columns[0]:
            s1 = df.columns[0]
            s2 = df.columns[1]
        else:
            s1 = df.columns[1]
            s2 = df.columns[0]
        
        df = df.sort_values(s1).iloc[:5]
        df = df.sort_values(s2).iloc[:1]
        num_boost_rounds[i] = df.index.tolist()[0]

        if i==0:
            if show_result:print("P05 rounds",'=>',num_boost_rounds[i])
        else:
            if show_result:print("P95 rounds",'=>',num_boost_rounds[i])

    dic_metrics = {}
    if using_cv:
        if show_result:print('cv start')
        lst = []
        kf = KFold(n_splits=5,shuffle=True,random_state=42)
        for _, (train_index, val_index) in enumerate(kf.split(y_train)):

            X_cv_train = X_train[train_index]
            y_cv_train = y_train[train_index]
            X_cv_val  = X_train[val_index]
            y_cv_val  = y_train[val_index]

            lgbm_P05 = LGBMRegressor(**params[0],n_estimators=num_boost_rounds[0],shuffle=True)
            lgbm_P05.fit(X_cv_train, y_cv_train,eval_set=(X_cv_val,y_cv_val))

            lgbm_P95 = LGBMRegressor(**params[1],n_estimators=num_boost_rounds[1],shuffle=True)
            lgbm_P95.fit(X_cv_train, y_cv_train,eval_set=(X_cv_val,y_cv_val))

            df_metrics = metrics_quantile([lgbm_P05],[lgbm_P95],X_train,y_train,X_test,y_test)

            lst.append(df_metrics)
        
        df_metrics = pd.concat(lst)
        lst = []

        df_metrics = df_metrics.loc[['coverage_rate']].agg(['var','mean'])
        df_metrics['metrics'] = 'coverage_rate'

        dic_metrics['cv_mean'] = df_metrics.loc[['mean']]
        dic_metrics['cv_mean'] = dic_metrics['cv_mean'].reset_index(drop=True)[['metrics','train_baseline','train','test_baseline','test']].rename(columns={'test':'val'})
        
        dic_metrics['cv_var'] = df_metrics.loc[['var']]
        dic_metrics['cv_var'] = dic_metrics['cv_var'].reset_index(drop=True)[['metrics','train_baseline','train','test_baseline','test']].rename(columns={'test':'val'})

        if show_result:print('cv end')

    lgbm_P05 = LGBMRegressor(**params[0],n_estimators=num_boost_rounds[0],shuffle=True)
    lgbm_P05.fit(X_train,y_train)

    lgbm_P95 = LGBMRegressor(**params[1],n_estimators=num_boost_rounds[1],shuffle=True)
    lgbm_P95.fit(X_train,y_train)

    df_metrics = metrics_quantile([lgbm_P05],[lgbm_P95],X_train,y_train,X_test,y_test)

    df_metrics['metrics'] = df_metrics.index
    df_metrics = df_metrics.reset_index(drop=True)

    dic_metrics['full_dataset'] = df_metrics[['metrics','train_baseline','train','test_baseline','test']]
    
    return lgbm_P05,lgbm_P95,dic_metrics

def show_residual_plot(data,model,treget_col,monitor_col,kind='box'):
    '''
    kind = violin or scatt or box
    default is box
    '''
    df = pd.DataFrame()
    df['residual'] = data[treget_col] - model.predict(data.drop(columns=treget_col))
    df[monitor_col] = data[monitor_col]
    if kind =='scatt':
        sns.scatterplot(x=df[monitor_col],y=df['residual'])
    elif kind=='violin' :
        sns.violinplot(df,x=monitor_col,y='residual')
    else:
        sns.boxplot(df,x=monitor_col,y='residual')
    plt.show()

def show_dou_residual_plot(data_train,data_test,model,treget_col,monitor_col,kind='box',figsize=(15,6)):
    '''
    kind = violin or scatt or box
    default is box
    '''
    _, axes = plt.subplots(nrows=1, ncols=2,figsize=figsize)
    df = pd.DataFrame()
    df['residual'] = data_train[treget_col] - model.predict(data_train.drop(columns=treget_col))
    df[monitor_col] = data_train[monitor_col]
    if kind =='scatt':
        sns.scatterplot(x=df[monitor_col],y=df['residual'],label='train',ax=axes[0])
    elif kind=='violin' :
        sns.violinplot(df,x=monitor_col,y='residual',label='train',ax=axes[0])
    else:
        sns.boxplot(df,x=monitor_col,y='residual',label='train',ax=axes[0])
    
    df = pd.DataFrame()
    df['residual'] = data_test[treget_col] - model.predict(data_test.drop(columns=treget_col))
    df[monitor_col] = data_test[monitor_col]
    if kind =='scatt':
        sns.scatterplot(x=df[monitor_col],y=df['residual'],label='test',ax=axes[1])
    elif kind=='violin' :
        sns.violinplot(df,x=monitor_col,y='residual',label='test',ax=axes[1])
    else:
        sns.boxplot(df,x=monitor_col,y='residual',label='test',ax=axes[1])
    
    plt.show()

def compare_metrics(df_a,df_b):
    '''
    df_b - df_a
    '''
    df = df_b[['train_baseline','train','test_baseline','test']] - df_a[['train_baseline','train','test_baseline','test']]
    df['metrics'] = df_b['metrics']
    df = df[['metrics','train_baseline','train','test_baseline','test']]
    df.style.format("{:.6f}")
    
    return df