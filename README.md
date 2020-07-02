# Machine-Learning-SmartParking
A collection of Machine Learning algorithms created for a Open SmartParking DataSet.

## Dataset informations

You can acess this dataset from: https://www.data.act.gov.au/Transport/Smart-Parking-Stays/3vsj-zpk7 (356 MB)

This dataset have 3.72 milions of Rows and 8 Columns.

## Objectives of this Project

Prepare the dataset for build any kind of data analysis algorithms and use some machine learning approaches.

Make predictions based in the dataset, like identify:

 - Predict When parking lots will have more or less demand.
 - Predict the number of a parking lots for a given day.
 - Generate parking time/day charts.

## 1. Data Preparing

The first steps in the project was prepare the data.
The datetime format used in the dataset is the 12H with AM/PM like:
<code>01/01/2017 08:13:07 AM</code>

We need to modify all the date formats to a 24H format, for two columns: "Arrived" - "Departed"

Using the <code>DataPreparing.py</code> you set the input dataset and get a output dataset with same number of rows and columns, but all the date columns will be in  the 24H standard.

This code consumes a good amount of time, about 30 minutes to convert all the 3.72 milions of rows to the standard date format.

## 2. Data Retrieve

In this step, using the <code>RetrieveDataFromDataset.py</code> you split the dataset from a given query, to separe a part of the data and do another algorithms in a small dataset (with 237.000 rows - 23mb).

This new dataset has all parking lots between January and March 2017.

I did this to improve the speed of analysis within the data, with a lower dataset, all the query algorithms that will run faster.

Therefore, when the codes used are ready and well implemented, they can be used in the complete dataset created in step 1 (Data preparation).

## 2. Data Visualization

We can visualize the data with codes like <code>plotGraph.py</code> or <code>plotGraph.py</code>

This algorithms plot a graph with Arrived hour in the X Axys and the Departed hour in the Y Axys.

In the <code>plotGraph.py</code> you can specify a day and month in the input header to generate a graph of all parking lots for a specific day.

In the <code>plotGraph2.py</code> you can specify a time in the input header to generate a graph of all parking lots received within a specific interval of 1 hour (for example: 10:00 to 10:59).

#### 2.1 PlotGraph.py



#### 2.2 PlotGraph2.py

