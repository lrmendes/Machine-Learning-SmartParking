# https://www.kaggle.com/mypapit/klccparking

from __future__ import division
from datetime import datetime, timedelta,date
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as md

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

#list = ""
#for i in range(28,97):
#    list += "'lag_'"+str(i)+"' +"
##print(list)
#exit()

outputCSV = "OutputDatasets/parking-klcc-predict-datetime-occupation.csv"

#print("[Filter] Opening Dataset...")
header_list = ["park_name", "total","date"]
data = pd.read_csv('parking-klcc-2016-2017.txt', names=header_list, sep=';', engine='python')

# Define Numbers for "Open" & "Full" Values (From Dataset)
data = data.replace({'total': 'FULL'}, 0)
data = data.replace({'total': 'OPEN'}, np.nan)

# Remove OPEN (NA Values)
data = data.dropna()

data['total'] = pd.to_numeric(data['total'])

data['date'] = pd.to_datetime(data['date'])
data = data[(data['date'].dt.year == 2016)]

data = data[(data['date'].dt.day >= 3) | (data['date'].dt.month != 6)]
data['date'] = data['date'].dt.floor('T')

data = data[(data['date'].dt.minute == 0) | (data['date'].dt.minute == 15) |
            (data['date'].dt.minute == 30) | (data['date'].dt.minute == 45) ]

df_predict = data.copy()
df_predict['date'] = pd.to_datetime(df_predict['date'])

# PLOT GRAPH -> PART 1
fig, ax = plt.subplots(figsize=(120,6))

ax.set_xlim(df_predict['date'].min(),
            df_predict['date'].max())

ax.xaxis.set_major_locator(md.HourLocator(interval = 6))
ax.xaxis.set_major_formatter(md.DateFormatter('%d/%m %Hh'))

plt.axhline(y=5500, color='r', linestyle='dashed', label="Park Spots Empty")
plt.axhline(y=0, color='r', linestyle='dashed', label="Parking Spots Full")

plt.plot(df_predict['date'], df_predict['total'])
plt.xticks(rotation=90, )

plt.title('Parking Lot Occupation - Original Data\n(15min verification)', fontsize=20)
plt.xlabel("Time Range from Sensor",  labelpad=20,  fontsize=12)
plt.ylabel("Parking Spots Available", labelpad=20, fontsize=12)

plt.locator_params(axis='y', nbins=21)

plt.grid()
plt.show()

# PART 2
df_diff = df_predict.copy()
#add previous predict to the next row
df_diff['prev_total'] = df_diff['total'].shift(1)
#drop the null values and calculate the difference
df_diff = df_diff.dropna()
df_diff['diff'] = (df_diff['total'] - df_diff['prev_total'])

#print(df_diff.head(15))
#print("\n------------------------\n")
#print(df_predict.head(15))


# PLOT GRAPH -> PART 2
#plt.rcParams["figure.figsize"] = (20,6)
#plt.plot(df_diff['date'], df_diff['diff'])

fig, ax = plt.subplots(figsize=(120,6))

ax.set_xlim(df_diff['date'].min(),
            df_diff['date'].max())

ax.xaxis.set_major_locator(md.HourLocator(interval = 6))
ax.xaxis.set_major_formatter(md.DateFormatter('%d/%m %Hh'))

plt.axhline(y=5500, color='r', linestyle='dashed', label="Park Spots Empty")
plt.axhline(y=0, color='r', linestyle='dashed', label="Parking Spots Full")

plt.plot(df_diff['date'], df_diff['diff'])
plt.xticks(rotation=90, )

plt.title('[Difference] Parking Lot Occupation - Original Data\n(15min verification)', fontsize=20)
plt.xlabel("Time Range from Sensor",  labelpad=20,  fontsize=12)
plt.ylabel("Parking Spots Available", labelpad=20, fontsize=12)

plt.locator_params(axis='y', nbins=21)

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

##print(df_supervised.head(10))

# PART 4

# Import statsmodels.formula.api
import statsmodels.formula.api as smf
# Define the regression formula
"""
model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + '
                        'lag_10 + lag_11 + lag_12 + lag_13 + lag_14 + lag_15 + lag_16 + lag_17 + lag_18 + '
                        'lag_19 + lag_20 + lag_21 + lag_22 + lag_23 + lag_24 + lag_25 + lag_26 + lag_27 + '
                        'lag_28 + lag_29 + lag_30 + lag_31 + lag_32 + lag_33 + lag_34 + lag_35 + lag_36 + '
                        'lag_37 + lag_38 + lag_39 + lag_40 + lag_41 + lag_42 + lag_43 + lag_44 + lag_45 + '
                        'lag_46 + lag_47 + lag_48 + lag_49 + lag_50 + lag_51 + lag_52 + lag_53 + lag_54 + '
                        'lag_55 + lag_56 + lag_57 + lag_58 + lag_59 + lag_60 + lag_61 + lag_62 + lag_63 + '
                        'lag_64 + lag_65 + lag_66 + lag_67 + lag_68 + lag_69 + lag_70 + lag_71 + lag_72 + '
                        'lag_73 + lag_74 + lag_75 + lag_76 + lag_77 + lag_78 + lag_79 + lag_80 + lag_81 + '
                        'lag_82 + lag_83 + lag_84 + lag_85 + lag_86 + lag_87 + lag_88 + lag_89 + lag_90 + '
                        'lag_91 + lag_92 + lag_93 + lag_94 + lag_95 + lag_96', data=df_supervised)
"""

model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + '
                        'lag_10 + lag_11 + lag_12 + lag_13 + lag_14 + lag_15 + lag_16 + lag_17 + lag_18 + '
                        'lag_19 + lag_20 + lag_21 + lag_22 + lag_23 + lag_24 + lag_25 + lag_26 + lag_27 + '
                        'lag_28', data=df_supervised)


# Fit the regression
model_fit = model.fit()
# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
#print(regression_adj_rsq)

# PART 5
#import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler
df_model = df_supervised.drop(['total','date','park_name'],axis=1)

#print("FIM")
#print(len(df_model))
x = df_supervised[13515:13520]
#print(x)

#input()

#split train and test set
train_set, test_set = df_model[0:13519].values, df_model[13519:].values

#print(train_set[0:1])
#print(train_set[len(train_set)-1:])
#print(len(train_set))
#print("\n\n")
#print(test_set[0:1])
#print(test_set[len(test_set)-1:])
#print(len(test_set))

##print(len(train_set)," + ",len(test_set))

##print(train_set)
##print("-----")
##print(test_set)

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
model.fit(X_train, y_train, nb_epoch=2, batch_size=1, verbose=1, shuffle=False)

y_pred = model.predict(X_test,batch_size=1)
#for multistep prediction, you need to replace X_test values with the predictions coming from t-1

# PART 6
#reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
#rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred)):
    #print(np.concatenate([y_pred[index],X_test[index]],axis=1))
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))
#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

#print(len(pred_test_set_inverted))
#print(pred_test_set_inverted)

xyz = pred_test_set_inverted[0:200]
xyzz = pred_test_set

#print("----------|----------------")

#create dataframe that shows the predicted predict
result_list = []
predict_dates = list(df_predict[-3402:].date)
act_predict = list(df_predict[-3402:].total)
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_predict[index])
    result_dict['pred_value_diff'] = int(pred_test_set_inverted[index][0])
    #print("real: ",act_predict[index]," - predict: ",
    #      str(int(pred_test_set_inverted[index][0] + act_predict[index]))," - var: ", str(int(pred_test_set_inverted[index][0])))

    ##print(index, " - ", len(predict_dates), " - ", len(pred_test_set_inverted))
    result_dict['date'] = predict_dates[index+1]
    result_list.append(result_dict)
df_result = pd.DataFrame(result_list)

#for multistep prediction, replace act_predict with the predicted predict
df_predict_pred = pd.merge(df_predict,df_result,on='date',how='left')

df_predict_diff_pred = pd.merge(df_diff,df_result,on='date',how='left')

#print("-------------------------")
#print(df_predict_pred)

#df_predict_pred['pred_value'] = pd.to_numeric(df_predict_pred['pred_value'])
#df_predict_pred = df_predict_pred.where(df_predict_pred['pred_value'] < 0, 0)

#print("Saving Output CSV...")
df_predict_pred.to_csv(outputCSV, index=False)

#df_predict_pred = df_predict_pred.dropna()

"""
plt.rcParams["figure.figsize"] = (120,5)
plt.plot(df_predict_pred['date'], df_predict_pred['total'], label='Total (Real)')
plt.plot(df_predict_pred['date'], df_predict_pred['pred_value'], linestyle='dashed', label='Total (Predict)')
plt.legend(loc='upper left', frameon=False, prop={'size': 25})
plt.xticks(rotation=90)
plt.title('(Real & Predict) Parking Lot Occupation', fontsize=25)
plt.grid()
plt.show()
"""

# Plot Predict DIFF
fig, ax = plt.subplots(figsize=(120,6))

ax.set_xlim(df_predict_diff_pred['date'].min(),
            df_predict_diff_pred['date'].max())

ax.xaxis.set_major_locator(md.HourLocator(interval = 6))
ax.xaxis.set_major_formatter(md.DateFormatter('%d/%m %Hh'))

plt.axhline(y=5500, color='r', linestyle='dashed', label="Park Spots Empty")
plt.axhline(y=0, color='r', linestyle='dashed', label="Parking Spots Full")

plt.plot(df_predict_diff_pred['date'], df_predict_diff_pred['diff'], label='Total (Real)')
plt.plot(df_predict_diff_pred['date'], df_predict_diff_pred['pred_value_diff'], linestyle='dashed', label='Total (Predict)')

plt.xticks(rotation=90, )

plt.title('[Difference] Parking Lot Occupation - Original Data\n(15min verification)', fontsize=20)
plt.xlabel("Time Range from Sensor",  labelpad=20,  fontsize=12)
plt.ylabel("Parking Spots Available", labelpad=20, fontsize=12)

plt.locator_params(axis='y', nbins=21)
plt.grid()
plt.show()

# Plot Predict subtracting from real data
fig, ax = plt.subplots(figsize=(120,6))

ax.set_xlim(df_predict_pred['date'].min(),
            df_predict_pred['date'].max())

ax.xaxis.set_major_locator(md.HourLocator(interval = 6))
ax.xaxis.set_major_formatter(md.DateFormatter('%d/%m %Hh'))

plt.axhline(y=5500, color='r', linestyle='dashed', label="Park Spots Empty")
plt.axhline(y=0, color='r', linestyle='dashed', label="Parking Spots Full")

plt.plot(df_predict_pred['date'], df_predict_pred['total'], label='Total (Real)')
plt.plot(df_predict_pred['date'], df_predict_pred['pred_value'], linestyle='dashed', label='Total (Predict)')

plt.xticks(rotation=90, )

plt.title('[Difference] Parking Lot Occupation - Original+Predict Data\n(15min verification)', fontsize=20)
plt.xlabel("Time Range from Sensor",  labelpad=20,  fontsize=12)
plt.ylabel("Parking Spots Available", labelpad=20, fontsize=12)

plt.locator_params(axis='y', nbins=21)
plt.grid()
plt.show()

"""
df_predict_pred = df_predict_pred.dropna()
plt.rcParams["figure.figsize"] = (20,6)
plt.plot(df_predict_pred['date'], df_predict_pred['total'], label='Total (Real)')
plt.plot(df_predict_pred['date'], df_predict_pred['pred_value'], linestyle='dashed', label='Total (Predict)')
plt.legend(loc='upper left', frameon=False, prop={'size': 25})
plt.xticks(rotation=45)
plt.title('(Real & Predict) Parking Lot Occupation', fontsize=25)
plt.grid()
plt.show()
"""

# Plot Predict DIFF
print(len(df_predict_pred))
df_predict_diff_pred = df_predict_diff_pred.dropna(subset=['pred_value_diff'])
print(len(df_predict_pred))

fig, ax = plt.subplots(figsize=(160,6))

ax.set_xlim(df_predict_diff_pred['date'].min(),
            df_predict_diff_pred['date'].max())

ax.xaxis.set_major_locator(md.HourLocator(interval = 1))
ax.xaxis.set_major_formatter(md.DateFormatter('%d/%m %Hh'))

#plt.axhline(y=5500, color='r', linestyle='dashed', label="Park Spots Empty")
#plt.axhline(y=0, color='r', linestyle='dashed', label="Parking Spots Full")

plt.plot(df_predict_diff_pred['date'], df_predict_diff_pred['diff'], label='Total (Real)')
plt.plot(df_predict_diff_pred['date'], df_predict_diff_pred['pred_value_diff'], linestyle='dashed', label='Total (Predict)')

plt.xticks(rotation=90, )

plt.title('[Difference] Parking Lot Occupation - Original+Predict Data\n(15min verification)', fontsize=20)
plt.xlabel("Time Range from Sensor",  labelpad=20,  fontsize=12)
plt.ylabel("Parking Spots Available", labelpad=20, fontsize=12)

plt.locator_params(axis='y', nbins=21)
plt.grid()
plt.show()