import pandas as pd

"""
Column Names Reference:

SectorCode
 SectorName
 LotName
 BayNumber
 BayName
 Arrived
 Departed
Street
"""

"""
Street Names Reference:

Street               Number Of Parking Lots from Dataset
Bougainville St      82185
Franklin St          39702
Flinders Way         38177
Captain Cook Cres    32714
Multi Storey         29871
Furneaux St          13548
Palmerston Lane       1588

Get This Info with:
df3 = data['Street'].value_counts()
print(df3)
"""

# This Algorithm have 3 Filters, and generates 3 CSV Files.
# The First filter results in a CSV with All Parkings of a Given Place (Street).
# The Second filter results in a CSV based on the first filter, grouped by Date with the Total Parking Lots of Each Date.
# The Third filter results in a CSV based on the second filter and ordered by Date and filtered by a given year (2016).

# Filter 1 - Read Dataset & Filter by one Street
inputCSV = "Smart_Parking_Stays_Formatted.csv"
outputCSV = "OutputDatasets/Smart_Parking_Stays_Formatted_AllParkingsOfGivenPlace.csv"

queryStreet = 'Bougainville St'

print("[Filter 1] Opening Dataset...")
data = pd.read_csv(inputCSV)

print("[Filter 1] Setting the StreetName Filter in Dataset...")
data = data[(data['Street'] == 'Bougainville St')]

print("[Filter 1] Saving Output File...")
data.to_csv(outputCSV, encoding='utf-8-sig', index=False, header=True)

# Filter 2 - Group Dataset By Same DATES (Days)
inputCSV = "OutputDatasets/Smart_Parking_Stays_Formatted_AllParkingsOfGivenPlace.csv"
outputCSV = "OutputDatasets/Smart_Parking_Stays_Formatted_AllParkingsOfGivenPlace_TotalByDay.csv"

print("[Filter 2] Opening Dataset...")
data = pd.read_csv(inputCSV)

data[' Arrived'] = pd.to_datetime(data[' Arrived']).dt.normalize()

save = data[' Arrived'].value_counts()

save.columns = ['Total']
save.index.name = 'Date'

print("[Filter 2] Saving Output File...")
save.to_csv(outputCSV, encoding='utf-8-sig', index=True, header=True)

# Filter 3 - Order By Date & Get Only 1 Year
inputCSV = "OutputDatasets/Smart_Parking_Stays_Formatted_AllParkingsOfGivenPlace_TotalByDay.csv"
outputCSV = "OutputDatasets/Smart_Parking_Stays_Formatted_AllParkingsOfGivenPlace_TotalByDay_Order.csv"

print("[Filter 3] Opening Dataset...")
data = pd.read_csv(inputCSV)

print("[Filter 3] Setting the Pattern in DateTime Columns...")
data['Date'] = pd.to_datetime(data['Date'], errors='ignore')
data['Date'] = pd.to_datetime(data['Date'].dt.strftime('%Y/%m/%d'))

data = data.sort_values(['Date'])

save = data[(data['Date'].dt.year == 2016)]

print("[Filter 3] Saving Output File...")
save.to_csv(outputCSV, encoding='utf-8-sig', index=False, header=True)