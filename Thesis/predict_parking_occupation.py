# https://www.kaggle.com/mypapit/klccparking

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdt

outpuv_csv = 'OutputDatasets/parking-klcc-standardized.csv'
output_graph_png = "OutputGraphs/parking_occupation_from_sensor.png"
plot_year = 2016

header_list = ["park_name", "spots_available","verification_date","verification_time"]
data = pd.read_csv('parking-klcc-2016-2017.txt', names=header_list, sep=';| ', engine='python')

# Define Numbers for "Open" & "Full" Values (From Dataset)
data = data.replace({'spots_available': 'FULL'}, 0)
data = data.replace({'spots_available': 'OPEN'}, np.nan)

# Remove OPEN (NA Values)
data = data.dropna()

data['spots_available'] = pd.to_numeric(data['spots_available'])
data['verification_date'] = pd.to_datetime(data['verification_date'])
data = data[(data['verification_date'].dt.year == plot_year)]

data['verification_time'] = pd.to_datetime(data['verification_time'])
data['verification_time'] = data['verification_time'].dt.floor('T')
data['verification_time'] = pd.to_datetime(data['verification_time'])
data = data[(data['verification_time'].dt.minute == 0) | (data['verification_time'].dt.minute == 15) |
            (data['verification_time'].dt.minute == 30) | (data['verification_time'].dt.minute == 45) ]


#data = data[(data['verification_time'].dt.minute != 0) & (data['verification_time'].dt.minute != 15) &
#            (data['verification_time'].dt.minute != 30) & (data['verification_time'].dt.minute != 45) ]

data['verification_time'] = pd.to_datetime(data['verification_time'], format='%H:%M').dt.time

data_counts = data['verification_date'].value_counts().rename('date_counts')

data = data.merge(data_counts.to_frame(), left_on='verification_date', right_index=True)

newdata = data[data.date_counts == 96]


# Function by stackoverflowuser2010
# https://stackoverflow.com/questions/30942755/plotting-multiple-time-series-after-a-groupby-in-pandas
def plot_gb_time_series(df, ts_name, gb_name, value_name, figsize=(40,10), title=None):
    '''
    Runs groupby on Pandas dataframe and produces a time series chart.

    Parameters:
    ----------
    df : Pandas dataframe
    ts_name : string
        The name of the df column that has the datetime timestamp x-axis values.
    gb_name : string
        The name of the df column to perform group-by.
    value_name : string
        The name of the df column for the y-axis.
    figsize : tuple of two integers
        Figure size of the resulting plot, e.g. (20, 7)
    title : string
        Optional title
    '''
    xtick_locator = mdt.DayLocator(interval=1)
    xtick_dateformatter = mdt.DateFormatter('%m/%d/%Y')
    fig, ax = plt.subplots(figsize=figsize)
    for key, grp in df.groupby([gb_name]):
        ax = grp.plot(ax=ax, kind='line', x=ts_name, y=value_name, label=key, marker='o')
    ax.xaxis.set_major_locator(xtick_locator)
    ax.xaxis.set_major_formatter(xtick_dateformatter)
    ax.autoscale_view()
    ax.legend(loc='upper left')
    _ = plt.xticks(rotation=90, )
    _ = plt.locator_params(axis='y', nbins=14)
    _ = plt.grid()
    _ = plt.xlabel('')
    _ = plt.ylim(0, df[value_name].max() * 1.25)
    _ = plt.ylabel(value_name)
    if title is not None:
        _ = plt.title(title)
    _ = plt.show()

    ax.figure.savefig(output_graph_png)
    print("Saving Output Graph...")


plot_gb_time_series(newdata, "verification_date", "verification_time", "spots_available")
newdata.to_csv(outpuv_csv, encoding='utf-8-sig', index=False, header=True)

#print(newdata.groupby(['verification_date']).count())
#newdata.groupby(['verification_date']).count()

print("Finished!")