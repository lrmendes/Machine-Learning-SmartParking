# Define a pattern for DATE column (D-M-Y H-M-S)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns

"""
Column Names Reference:
    HourRange	
    TotalParkingLots	
    DailyAverageParkingLotsByHour	
    DailyParkingPercentageByHours
"""

# Input of Dataset
inputCSV = "OutputDatasets/Smart_Parking_Stays_JanFebMar_2017_NumberOfParkingsArrivedGroupedByHours.csv"
outputGraphFolder = "OutputGraphs/"

# Begin Algorithm

print("Opening Dataset...")
data = pd.read_csv(inputCSV)

total_parkings = data['TotalParkingLots'].sum()
total_parkings_day_average = data['DailyAverageParkingLotsByHour'].sum()

## Graph 1

plt.figure(figsize=(20, 12))
#sns.set_style("whitegrid")
plt.locator_params(axis='y', nbins=14)

print("Generating Graph 1 - Total Number of Parkings")
final_data_plot = sns.barplot(x=data['HourRange'], y=data['TotalParkingLots'])
final_data_plot.set_xticklabels(final_data_plot.get_xticklabels(), rotation=60, ha='right', fontsize=16)
final_data_plot.set_yticklabels(final_data_plot.get_yticks(), size=16, )

plt.title("Number of Parkings Arrived Between 01/01/2017 to 31/03/2017 by Hour\nTotal Parkings: "+str(total_parkings), fontsize=30, pad=20)
plt.xlabel("Arrived Hour",  labelpad=20,  fontsize=20)
plt.ylabel("Number of Parking Lots Started", labelpad=20, fontsize=20)

plt.grid()
plt.show()

fig = final_data_plot.get_figure()
print("Saving Output Graph 1 (Total)...")
fig.savefig(outputGraphFolder+"Smart_Parking_Stays_JanFebMar_2017_TotalOfParkingsArrivedByHourRange.png")

## Graph2

plt.figure(figsize=(20, 12))
#sns.set_style("whitegrid")
plt.locator_params(axis='y', nbins=14)

print("Generating Graph 2 - Daily Average of Parking Lots by Hours")
ax = sns.barplot(x=data['HourRange'], y=data['DailyAverageParkingLotsByHour'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=16)
ax.set_yticklabels(ax.get_yticks(), size=16 )

# Begin Function By Deepak Natarajan
# From: https://stackoverflow.com/questions/31749448/how-to-add-percentages-on-top-of-bars-in-seaborn
for p in ax.patches:
    percentage = '{:.1f}%'.format((p.get_height() / total_parkings_day_average) * 100)
    x = p.get_x() + p.get_width() / 2 - 0.25
    y = p.get_y() + p.get_height() + 1
    ax.annotate(percentage, (x, y), size=12)
# End Function

plt.title("Daily Average Parking per Hour - Between 01/01/2017 to 31/03/2017\nAverage Number of Parking Lots per Day: "+str(total_parkings_day_average), fontsize=30, pad=20)
plt.xlabel("Arrived Hour",  labelpad=20,  fontsize=20)
plt.ylabel("Number of Parking Lots Started", labelpad=20, fontsize=20)

plt.grid()
plt.show()

fig = ax.get_figure()
print("Saving Output Graph 1 (Total)...")
fig.savefig(outputGraphFolder+"Smart_Parking_Stays_JanFebMar_2017_DailyAverageOfParkingsArrivedByHourRange.png", bbox_inches="tight")

## Graph3

plt.figure(figsize=(20, 12))
#sns.set_style("whitegrid")
plt.locator_params(axis='y', nbins=14)

data['DailyParkingPercentageByHours'] = data['DailyParkingPercentageByHours'].apply(lambda y: y*100)

print("Generating Graph 3 - Daily Average of Parking Lots by Hours (Percentage)")
final_data_plot = sns.barplot(x=data['HourRange'], y=data['DailyParkingPercentageByHours'])
final_data_plot.set_xticklabels(final_data_plot.get_xticklabels(), rotation=60, ha='right', fontsize=16)
final_data_plot.set_yticklabels(final_data_plot.get_yticks(), size=16, )

plt.title("Daily Average Parking per Hour (Percentage) - Between 01/01/2017 to 31/03/2017\nAverage Number of Parking Lots per Day: "+str(total_parkings_day_average), fontsize=30, pad=20)
plt.xlabel("Arrived Hour",  labelpad=20,  fontsize=20)
plt.ylabel("Percentage of Parking Lots Started (%)", labelpad=20, fontsize=20)

plt.grid()
plt.show()

fig = final_data_plot.get_figure()
print("Saving Output Graph 1 (Total)...")
fig.savefig(outputGraphFolder+"Smart_Parking_Stays_JanFebMar_2017_DailyPercentageAverageOfParkingsArrivedByHourRange.png")

exit()
