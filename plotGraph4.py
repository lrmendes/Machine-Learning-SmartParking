import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np


inputCSV = "OutputDatasets/Smart_Parking_Stays_JanFebMar_2017_AverageArrivedDepartedByDateRange.csv"
outputGraphFolder = "OutputGraphs/"

print("Opening Dataset...")
data = pd.read_csv(inputCSV)

print("Generating Graph...")
fig, ax = plt.subplots(figsize=(10,6))
plt.locator_params(axis='y', nbins=22)

sns.set_style("darkgrid")

data1 = sns.lineplot(x='HourRange', y='ArrivedParkingLots', data=data, linewidth=2.5)
data2 = sns.lineplot(x='HourRange', y='DepartedParkingLots', data=data, linewidth=2.5)

#data1.set_xticklabels(data1.get_xticklabels(), rotation=60, ha='right', fontsize=16)
#data2.set_xticklabels(data2.get_xticklabels(), rotation=60, ha='right', fontsize=16)

fig.autofmt_xdate()

plt.ylabel("Number of Parking Lots", labelpad=20)
plt.xlabel("Hour Range",  labelpad=20)

plt.legend(title='Subtitle', loc='upper left', labels=['Arrived Parking Lots', 'Departed Parking Lots'])
plt.title("Average Daily Parking by Hours (from 13/03/2017 to 19/03/2017)", fontsize=20, pad=20)

plt.grid()
plt.show()

figure = fig.get_figure()
print("Saving Output Graph 1 (Total)...")
fig.savefig(outputGraphFolder+"Smart_Parking_Stays_Formatted_DailyAverageParkingLotsArrivedDeparted.png")
