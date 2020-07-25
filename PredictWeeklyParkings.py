from __future__ import division
from datetime import datetime, timedelta,date
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

#import matplotlib.pyplot as plt
#import plotly.offline as pyoff
#import plotly.graph_objs as go

#import Keras
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# ******************************************************************
# This code is fully Based in following the Tutorial:
# https://towardsdatascience.com/predicting-sales-611cb5a252de
# Tutorial Author: Barış Karaman, 09 Jun 2019
# ******************************************************************

inputCSV = "OutputDatasets/Smart_Parking_Stays_Formatted_AllParkingsOfGivenPlace_TotalByDay_Order.csv"
outputCSV = "OutputDatasets/Smart_Parking_Stays_Formatted_PredictParking4.csv"

print("[Filter 2] Opening Dataset...")
df_sales = pd.read_csv(inputCSV)

df_sales['date'] = pd.to_datetime(df_sales['date'])

#df_sales = df_sales[(df_sales['date'].dt.month >= 2) & (df_sales['date'].dt.month <= 11)]

#print(df_diff.head())

#plot monthly sales
plt.rcParams["figure.figsize"] = (15,6)
plt.plot(df_sales['date'], df_sales['total'])
plt.xticks(rotation=45)
plt.title('Total of Parking Lots by Day (January ~ December)', fontsize=25)
plt.grid()
plt.show()

exit()

# PART 2

df_diff = df_sales.copy()
#add previous sales to the next row
df_diff['prev_total'] = df_diff['total'].shift(1)
#drop the null values and calculate the difference
df_diff = df_diff.dropna()
df_diff['diff'] = (df_diff['total'] - df_diff['prev_total'])

#print(df_diff.head(15))
#print("\n------------------------\n")
#print(df_sales.head(15))

plt.rcParams["figure.figsize"] = (20,6)
plt.plot(df_diff['date'], df_diff['diff'])
plt.xticks(rotation=45)
plt.title('Total Parking Lots by Month')
plt.grid()
plt.show()

# PART 3

#create dataframe for transformation from time series to supervised
df_supervised = df_diff.drop(['prev_total'],axis=1)
#adding lags
for inc in range(1,29):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc)
#drop null values
df_supervised = df_supervised.dropna().reset_index(drop=True)

#print(df_supervised.head(10))

# PART 4

# Import statsmodels.formula.api
import statsmodels.formula.api as smf
# Define the regression formula
model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + '
                        'lag_10 + lag_11 + lag_12 + lag_13 + lag_14 + lag_15 + lag_16 + lag_17 + lag_18 + '
                        'lag_19 + lag_20 + lag_21 + lag_22 + lag_23 + lag_24 + lag_25 + lag_26 + lag_27 + '
                        'lag_28', data=df_supervised)
# Fit the regression
model_fit = model.fit()
# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)

# PART 5
#import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler
df_model = df_supervised.drop(['total','date'],axis=1)

#split train and test set
train_set, test_set = df_model[0:220].values, df_model[220:].values

#print(len(train_set)," + ",len(test_set))

#print(train_set)
#print("-----")
#print(test_set)

#apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)
# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)

X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=100, batch_size=1, verbose=1, shuffle=False)

y_pred = model.predict(X_test,batch_size=1)
#for multistep prediction, you need to replace X_test values with the predictions coming from t-1

# PART 6
#reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
#rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred)):
    print(np.concatenate([y_pred[index],X_test[index]],axis=1))
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))
#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

print(len(pred_test_set_inverted))

#create dataframe that shows the predicted sales
result_list = []
sales_dates = list(df_sales[-56:].date)
act_sales = list(df_sales[-56:].total)
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['date'] = sales_dates[index+1]
    result_list.append(result_dict)
df_result = pd.DataFrame(result_list)

#for multistep prediction, replace act_sales with the predicted sales
df_sales_pred = pd.merge(df_sales,df_result,on='date',how='left')

print("-------------------------")
print(df_sales_pred)

#df_sales_pred = df_sales_pred.dropna()

plt.rcParams["figure.figsize"] = (20,6)
plt.plot(df_sales_pred['date'], df_sales_pred['total'], label='Total (Real)')
plt.plot(df_sales_pred['date'], df_sales_pred['pred_value'], label='Total (Predict)')
plt.legend(loc='upper left', frameon=False, prop={'size': 25})
plt.xticks(rotation=45)
plt.title('(Real & Predict) Total of Parking Lots by Day/Month', fontsize=25)
plt.grid()
plt.show()

df_sales_pred = df_sales_pred.dropna()
plt.rcParams["figure.figsize"] = (20,6)
plt.plot(df_sales_pred['date'], df_sales_pred['total'], label='Total (Real)')
plt.plot(df_sales_pred['date'], df_sales_pred['pred_value'], label='Total (Predict)')
plt.legend(loc='upper left', frameon=False, prop={'size': 25})
plt.xticks(rotation=45)
plt.title('(Real & Predict) Total of Parking Lots by Day/Month', fontsize=25)
plt.grid()
plt.show()

df_sales_pred.to_csv(outputCSV, index=False);