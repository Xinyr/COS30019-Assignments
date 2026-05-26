# Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ML / DL libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta

def traffic_lstm(predict_day: str, loc_index: int, time_index: int):
    """
    Predict traffic volume for a given day, location, and time slot using LSTM.

    Parameters
    ----------
    predict_day : str   – target date as "MM/DD/YYYY"
    loc_index   : int   – index into df['Location'].unique()
    time_index  : int   – column index of the 15-min interval (0 = 00:00)

    Returns
    -------
    float – predicted traffic volume (inverse-scaled)
    """
    # ── Load & filter data ───────────────────────────────────────────────────
    _df = pd.read_excel("Scats Data October 2006.xls", sheet_name='Data')
    _df = _df.drop(['SCATS Number', 'CD_MELWAY', 'NB_LATITUDE',
                    'NB_LONGITUDE', 'HF VicRoads Internal',
                    'VR Internal Stat', 'VR Internal Loc', 'NB_TYPE_SURVEY'], axis=1)

    _loc  = _df['Location'].unique()
    _df_0 = _df[_df['Location'] == _loc[loc_index]].copy()

    _df_0['Date']        = pd.to_datetime(_df_0['Date'])
    _df_0['day']         = _df_0['Date'].dt.day
    _df_0['month']       = _df_0['Date'].dt.month
    _df_0['year']        = _df_0['Date'].dt.year
    _df_0['day_of_week'] = _df_0['Date'].dt.dayofweek

    _x = _df_0[['day', 'month', 'year', 'day_of_week']].copy()
    _y = _df_0.drop(['day', 'month', 'year', 'day_of_week', 'Location', 'Date'], axis=1)
    _y = _y.iloc[:, time_index]

    _x['y_lag1'] = _y.shift(1)
    _x['y_lag2'] = _y.shift(2)
    _x = _x.dropna()
    _y = _y.loc[_x.index]

    # ── Train/test split (same 80/20, no shuffle) ───────────────────────────
    _x_train, _x_test, _y_train, _y_test = train_test_split(
        _x, _y, train_size=0.8, random_state=42, shuffle=False
    )

    # ── Scale ────────────────────────────────────────────────────────────────
    _sx, _sy = MinMaxScaler(), MinMaxScaler()
    _x_tr_sc = _sx.fit_transform(_x_train)
    _y_tr_sc = _sy.fit_transform(_y_train.values.reshape(-1, 1))

    _x_tr_3d = _x_tr_sc.reshape(_x_tr_sc.shape[0], 1, _x_tr_sc.shape[1])

    # ── Build & train ────────────────────────────────────────────────────────
    _n = _x_tr_3d.shape[2]
    _m = Sequential([
        LSTM(64, input_shape=(1, _n), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    _m.compile(optimizer='adam', loss='mse')
    _es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    _m.fit(_x_tr_3d, _y_tr_sc, epochs=100, batch_size=16,
           callbacks=[_es], verbose=0)

    # ── Build the new-day feature row (same lag strategy as regress.py) ──────
    today     = datetime.strptime(predict_day, "%m/%d/%Y")
    yesterday = today - timedelta(days=1)
    before    = today - timedelta(days=2)

    dow_y = yesterday.weekday()
    dow_b = before.weekday()

    lag1 = float(np.mean(_y_train[_x_train['day_of_week'] == dow_y]))
    lag2 = float(np.mean(_y_train[_x_train['day_of_week'] == dow_b]))

    x_new = pd.DataFrame([{
        'day':         today.day,
        'month':       today.month,
        'year':        today.year,
        'day_of_week': today.weekday(),
        'y_lag1':      lag1,
        'y_lag2':      lag2,
    }])

    x_new_sc  = _sx.transform(x_new)
    x_new_3d  = x_new_sc.reshape(1, 1, x_new_sc.shape[1])

    pred_sc = _m.predict(x_new_3d, verbose=0)
    pred    = _sy.inverse_transform(pred_sc).flatten()
    return pred


if __name__ == "__main__":
    result = traffic_lstm("10/15/2006", loc_index=1, time_index=0)
    print(f"\nPredicted traffic volume for 10/15/2006: {result[0]:.1f}")
