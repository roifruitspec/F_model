import pandas as pd
import numpy as np
from os import path
import tensorflow as tf
from preprocess import preprocess,conifg,scale
import models
from math import floor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from tensorflow.keras import callbacks, optimizers,regularizers
import swifter
from concurrent.futures import ProcessPoolExecutor
from multiprocesspandas import applyparallel


save_loc = path.join("D:", "NDVI test", "JAI-test")

def train_model(X_train,y_train,model,not_for_train_cols = ["full name","frame"],
                batch_size = 0,epochs = 100,group = "full name"):
    """
    not_for_train_cols: cols to drop and not train on, keeps for analysis and such
    batch_size: if != 0 then will use batch_size parameter for nn
    group: what column to use for groupkfold
    :return: trained model
    """
    print("fitting model")
    #to prevent fitting on name or frame number but keeping them for analysis
    cols = X_train.columns
    groups = X_train[group]
    for col in not_for_train_cols:
        if col in cols:
            X_train.drop(col,axis = 1,inplace = True)
    #fit the model
    if batch_size == 0:
        model.fit(X_train,y_train)
    #if nn
    else:
        #callbacks
        csv_logger = callbacks.CSVLogger(save_loc + '/pretrain_log.csv')
        redlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=6, verbose=1)
        early_stop = callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=15)
        chkpoint = callbacks.ModelCheckpoint(path.join(save_loc, 'model-best-{epoch:03d}.h5'), verbose=1,
                                             save_best_only=True)
        cb = [csv_logger, redlr, early_stop, chkpoint]
        # init group fold
        group_kfold = GroupKFold(n_splits=2)
        train_inds, test_inds = next(group_kfold.split(X_train, groups=groups))
        model.fit(X_train.iloc[train_inds], y_train.iloc[train_inds],
                  validation_data = (X_train.iloc[test_inds], y_train.iloc[test_inds]),
                  batch_size = batch_size,
                  epochs = epochs,
                  callbacks=cb)
    print("model is fit and ready to use")
    return(model)

def eval_model(X_val,y_val,model,not_for_train_cols = ["full name","frame"]):
    """
    not_for_train_cols: cols to drop and not train on, keeps for analysis and such
    :return: deviation based on our metric
    """
    #to prevent evaluating on name or frame number but keeping them for analysis
    print("evaluating model")
    cols = X_val.columns
    for col in not_for_train_cols:
        if col in cols:
            X_val.drop(col,axis = 1,inplace = True)
    y_pred = model.predict(X_val)
    if len(y_pred.shape) > 1:
        y_pred = np.max(y_pred, axis=1)
    print("model evaluated")
    return( np.mean(np.abs(y_pred - y_val)/y_val))

def validate(split,X,y,scale_type,model):
    """
    the purpose of this function is to able cross_validate via multiple processing
    :param split:  GroupKFold object
    :param X: matrix to train on
    :param y: target value
    :param scale_type: what scaler to use
    :param model: model to fit
    :return: model's score
    """
    #for each processor needs to reconfigure global variables
    conifg()
    #split indexes
    train_inds, test_inds = split
    #test,train split
    X_train, X_test = X.iloc[train_inds], X.iloc[test_inds]
    y_train, y_test = y.iloc[train_inds], y.iloc[test_inds]

    # scale if a scaler was provided
    if scale_type != "":
        X_train_trans, X_test_trans = scale(X_train, X_test, scale_type)
    # train for fold
    model = train_model(X_train_trans, y_train, model)
    # score for fold
    model_score = eval_model(X_test_trans, y_test, model)
    return(model_score)

def cross_validate(model,train_data,cv = 5,random_state = None,scale_type = "",target = "F",group = "full name"):
    """
    model: model to evaluate
    cv: number of time to run cross validation
    random_state: which random state to use
    scale_type: which scale type to use
    target: target column
    group: what column to use for groupkfold
    :return: deviation based on our metric cross validated
    """
    #init score list
    scores = []
    #init X,y
    X = train.drop(target, axis=1)
    y = train[target]
    #init group fold
    group_kfold = GroupKFold(n_splits = cv)
    #fold counter
    counter = 1
    #non parallel
    for train_inds, test_inds in group_kfold.split(train_data, groups=train_data[group]):
        #get fold matrix
        X_train, X_test = X.iloc[train_inds], X.iloc[test_inds]
        y_train, y_test = y.iloc[train_inds], y.iloc[test_inds]

        # scale if a scaler was provided
        if scale_type != "":
            X_train_trans, X_test_trans = scale(X_train, X_test, scale_type)
        # train for fold
        model = train_model(X_train_trans, y_train, model)
        # score for fold
        model_score = eval_model(X_test_trans, y_test, model)
        scores.append(model_score)
        print(f"finished fold:{counter}")
        counter +=1

    #parralell
    # split_list = [(train_inds, test_inds) for (train_inds, test_inds) in group_kfold.split(train_data, groups=train_data[group])]
    # with ProcessPoolExecutor(max_workers=5) as executor:
    #     scores = list(executor.map(validate,split_list,[X]*cv,[y]*cv,[scale_type]*cv,[model]*cv))

    print(model)
    print(f"model score is: {np.mean(scores)}")
    return(scores)
    #return mean score
    return(np.mean(scores))

if __name__ == '__main__':
    conifg()
    #split to train and validation
    train,val = preprocess(scale_type = "",return_train_val=True)

    X_train_trans, X_val_trans, y_train, y_val = preprocess(df = train,scale_type = "Standard")
    xg_model = models.xgboost_model({"learning_rate":0.05,"n_estimators":1000,"min_samples_leaf":3,
                                     "min_samples_split":2,"max_features":floor(np.sqrt(X_val_trans.shape[1])),
                                     "max_depth":5})
    xg_model = train_model(X_train_trans,y_train,xg_model)
    xg_score = eval_model(X_val_trans,y_val,xg_model)
    print("model score")
    print(xg_score)


    # xg_model = models.xgboost_model({"learning_rate":0.1,"n_estimators":10000,"min_samples_leaf":3,
    #                                  "min_samples_split":2,"max_features":floor(np.sqrt(X_val_trans.shape[1])),
    #                                  "max_depth":5})
    # xg_cv_score = cross_validate(xg_model,train,scale_type = "Standard")
    # print("model cv score")
    # print(xg_cv_score)
    # print(np.mean(xg_cv_score))

    param_grid = pd.DataFrame(np.array(np.meshgrid(
        [0.001,0.005,0.01,0.05,0.1,0.25,0.5],
        [100,250,500,1000,2500,5000],
        [3,5,7,9,11,20,50],
        [3,5,7,9,11,20,50],
        [3,5,7,9,11,15,None],
        [1,3,5,7])).T.reshape(-1, 6),
        columns=["learning_rate", "n_estimators","min_samples_leaf","min_samples_split","max_features","max_depth"])

    param_grid = pd.DataFrame(np.array(np.meshgrid(
        [0.001,0.005,0.01,0.05,0.1,],
        [1000,2500,5000],
        [5,10,25,50],
        [5,10,25,50],
        [3,7,11,None],
        [1,3,5,7])).T.reshape(-1, 6),
        columns=["learning_rate", "n_estimators","min_samples_leaf","min_samples_split","max_features","max_depth"])

    param_grid["cv_score"] = param_grid.apply(lambda row: np.mean(cross_validate(models.xgboost_model(dict(row)),
                                                                                 train,scale_type = "Standard")),axis = 1)
    param_grid.to_csv("param_grid_cv.csv")