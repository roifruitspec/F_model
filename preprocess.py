import pandas as pd
import numpy as np
from os import path
from sklearn.model_selection import train_test_split,GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler,StandardScaler,PowerTransformer,PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt

def conifg():
    global save_loc
    global data_loc
    global use_cols
    global scalers
    #where is the data frame to use
    data_loc = path.join("D:","NDVI test","JAI-test",'results_by_frame_test_new.csv')
    # where to save fata
    save_loc = path.join("D:", "NDVI test", "JAI-test")
    #which columns to use
    use_cols = ["full name","frame","F",
                'cv', 'total_foliage', 'total_orange','cv_foil',
                'mst_sums_arr', 'mst_mean_arr',"mst_skew_arr",
                "avg_intens_arr","med_intens_arr","q1","q3",
                "n_clust_mean_arr","n_clust_med_arr","n_clust_arr_2","n_clust_arr_4","n_clust_arr_6","n_clust_arr_8","n_clust_arr_10",
                "clusters_area_mean_arr","clusters_area_med_arr","clusters_CH_area_mean_arr","clusters_CH_area_med_arr"]
    #dicinory to scale the data
    scalers = {"Standard":StandardScaler(),
            "MinMax":MinMaxScaler(),
            "Power":PowerTransformer()}

def log_1(x):
    return(np.log(x+1))

def scale(X_train,X_val,scale_type,dont_scale = ["frame"]):
    """
    :param X_train: training data
    :param X_val: validation data
    :param scale_type: what scaler to use
    :param dont_scale: which columns not to scale
    :return:
    """
    #all numeric cols
    numeric_cols = list(X_train.select_dtypes('number').columns)
    #drop non scaling columns
    if len(dont_scale) > 0:
        for col in dont_scale:
            numeric_cols.remove(col)
    #fit and transform with given scaler
    scaler = scalers[scale_type]
    X_train_trans, X_val_trans = X_train.copy(), X_val.copy()
    X_train_trans[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val_trans[numeric_cols] = scaler.transform(X_val[numeric_cols])
    return(X_train_trans, X_val_trans)

def preprocess(df = "",scale_type = "Standard",random_state=7,return_train_val = False,
               log_apply_cols = ["clusters_area_mean_arr","clusters_area_med_arr"],target = "F"):
    """
    df: dataframe to preprocess if == "" then will take data from data_loc
    scale_type: what scale to use, if "" then no scaling
    random_state: random state to use
    return_train_val: wheter to return train and val or X_train,X_val,y_tain,y_val
    log_apply_cols: columns to apply log on
    target: target column
    pre process the data in data_loc and return the processed data
    """
    print("preprocess data")
    #if a dataframe was passed then uses the df insted of the csv
    if not isinstance(df,pd.core.frame.DataFrame):
        #read data
        df = pd.read_csv(data_loc)
        #filter relevent columns
        df = df[use_cols]
        # fill missing values, should only fill intensity features
        df.fillna(0, inplace=True)
        #apply log to nedded columns
        df[log_apply_cols] = df[log_apply_cols].apply(log_1)
    #split to train and test, keep trees in same split
    splitter = GroupShuffleSplit(test_size=.20, n_splits=1, random_state = random_state)
    split = splitter.split(df, groups=df['full name'])
    train_inds, test_inds = next(split)
    train = df.iloc[train_inds]
    val = df.iloc[test_inds]

    #return train and validaton in case flag is true
    if return_train_val:
        print("Finished preprocess fro train and val")
        return(train,val)

    #split y
    X_train,X_val,y_train,y_val = train.drop(target,axis = 1),val.drop(target,axis = 1),train[target],val[target]

    #scale if a scaler was provided
    if scale_type != "":
        X_train_trans,X_val_trans = scale(X_train, X_val, scale_type)
    print("Finished preprocess")
    return(X_train_trans, X_val_trans,y_train,y_val)