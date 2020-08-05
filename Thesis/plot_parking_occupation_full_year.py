# https://www.kaggle.com/mypapit/klccparking

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as md

output_graph_png = "OutputGraphs/parking_occupation_full_year.png"

print("Opening Dataset...")
header_list = ["park_name", "spots_available","date"]
data = pd.read_csv('parking-klcc-2016-2017.txt', names=header_list, sep=';', engine='python')

# Define Numbers for "Open" & "Full" Values (From Dataset)
data = data.replace({'spots_available': 'FULL'}, 0)
data = data.replace({'spots_available': 'OPEN'}, np.nan)

# Remove OPEN (NA Values)
data = data.dropna()

data['spots_available'] = pd.to_numeric(data['spots_available'])

data['date'] = pd.to_datetime(data['date'])
data = data[(data['date'].dt.year == 2016)]

data = data[(data['date'].dt.day >= 3) | (data['date'].dt.month != 6)]
data['date'] = data['date'].dt.floor('T')

data = data[(data['date'].dt.minute == 0) | (data['date'].dt.minute == 15) |
            (data['date'].dt.minute == 30) | (data['date'].dt.minute == 45) ]

df_final = data.copy()
df_final['date'] = pd.to_datetime(df_final['date'])

#plot monthly sales
fig, ax = plt.subplots(figsize=(120,10))

ax.set_xlim(df_final['date'].min(),
            df_final['date'].max())

ax.xaxis.set_major_locator(md.HourLocator(interval = 6))
ax.xaxis.set_major_formatter(md.DateFormatter('%d/%m %H:%M'))

plt.axhline(y=5500, color='r', linestyle='dashed', label="Park Spots Empty")
plt.axhline(y=0, color='r', linestyle='dashed', label="Parking Spots Full")

plt.plot(df_final['date'], df_final['spots_available'])
plt.xticks(rotation=90, )

plt.title('Parking Lot Occupation - Original Data\n(15min verification)', fontsize=20)
plt.xlabel("Time Range from Sensor",  labelpad=20,  fontsize=12)
plt.ylabel("Parking Spots Available", labelpad=20, fontsize=12)

plt.locator_params(axis='y', nbins=21)

plt.grid()
plt.show()

print("Saving Output Graph...")
ax.figure.savefig(output_graph_png)

print("Finished!")