# define the standard library for machine learning
import pandas as pd
import numpy as np

# define the machine learning model (regression tree)
from sklearn.tree import DecisionTreeRegressor

# define the model selection and evaluation libraries
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore")

def traffic_regression_tree(predict_day,loc_index,time_index):
    df = pd.read_excel("Scats Data October 2006.xls",sheet_name='Data')
    df =df.drop(['SCATS Number','CD_MELWAY','NB_LATITUDE',
             'NB_LONGITUDE','HF VicRoads Internal',
             'VR Internal Stat','VR Internal Loc','NB_TYPE_SURVEY'],axis=1)
    loc = df['Location'].unique()
    df_0 = df[df['Location']==loc[loc_index]]
    
    # expand the time information
    df_0['Date'] = pd.to_datetime(df_0['Date'])
    df_0['day'] = df_0['Date'].dt.day
    df_0['month'] = df_0['Date'].dt.month
    df_0['year'] = df_0['Date'].dt.year
    df_0['day_of_week'] = df_0['Date'].dt.dayofweek  # Monday=0, Sunday=6
    
    # define the independent and target variables
    x = df_0[['day','month','year','day_of_week']]
    y = df_0.drop(['day','month','year',
                'day_of_week','Location','Date'],axis=1)
    y = y.iloc[:,time_index]

    # expand with autoregressive or data from previous moment
    # expand the previous term data (1 and 2 before)
    x['y_lag1'] = y.shift(1)
    x['y_lag2'] = y.shift(2)
    [x_train,x_test,y_train,y_test] = train_test_split(x,y,train_size=0.8,random_state=42)
    
    # create regression tree model
    model = DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=3,   # key regularization parameter
        random_state=42
    )

    model.fit(x_train,y_train)
    
    # create the prediction
    today = datetime.strptime(predict_day, "%m/%d/%Y")
    yesterday = today - timedelta(days=1)
    day_before = today - timedelta(days=2)
    x_new = pd.DataFrame([{
        "now": today,
        "yesterday": yesterday,
        "before": day_before
    }])
    
    # expand the time information
    x_new['now'] = pd.to_datetime(x_new['now'])
    x_new['day'] = x_new['now'].dt.day
    x_new['month'] = x_new['now'].dt.month
    x_new['year'] = x_new['now'].dt.year
    x_new['day_of_week'] = x_new['now'].dt.dayofweek  # Monday=0, Sunday=6
    dow = pd.to_datetime(x_new['yesterday']).iloc[0].weekday()
    x_new['y_lag1'] = np.mean(y_train[x_train['day_of_week']==dow])
    dow = pd.to_datetime(x_new['before']).iloc[0].weekday()
    x_new['y_lag2'] = np.mean(y_train[x_train['day_of_week']==dow])
    
    # clean up data
    x_new = x_new.drop(['now','yesterday','before'],axis=1)
    
    # predict the result
    return model.predict(x_new)