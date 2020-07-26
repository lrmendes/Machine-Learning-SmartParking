# https://www.kaggle.com/mypapit/klccparking

import pandas as pd

header_list = ["park_name", "spots_available","verification_date","verification_time"]
data = pd.read_csv('parking-klcc-2016-2017.txt', names=header_list, sep=';| ', engine='python')
print(data.head(30))
