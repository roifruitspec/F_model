from sklearn.ensemble import GradientBoostingRegressor
#from pytorch_tabnet.tab_model import TabNetRegressor
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout

def xgboost_model(parmas = {}):
    """
    :param parmas: parameters for model initiation
    :return: xgboost model
    """
    #if no params where passed use defult model
    if len(parmas) == 0:
        xg_reg = GradientBoostingRegressor()
    else:
        xg_reg = GradientBoostingRegressor(**parmas)

    return(xg_reg)

def nn_model(input_shape = (22,)):
    #regressor = TabNetRegressor()
    input_layer = Input(shape=input_shape)
    hidden1 = Dense(64, activation='relu')(input_layer)
    hidden2 = Dense(64, activation='relu')(hidden1)
    hidden3 = Dense(64, activation='relu')(hidden2)
    hidden4 = Dense(32, activation='relu')(hidden3)
    hidden5 = Dense(16, activation='relu')(hidden4)
    output_class = Dense(1, activation='linear')(hidden5)
    regressor = Model(inputs=[input_layer], outputs=[output_class])
    regressor.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return(regressor)