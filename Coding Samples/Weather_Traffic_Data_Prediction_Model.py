import sklearn as sk
import numpy as np
import pandas as pd
import datetime
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#TRAIN/WEATHER DATA MERGE CELL
#https://angwalt12.medium.com/the-hidden-rules-of-pandas-merge-asof-e67293a5318e
#https://stackoverflow.com/questions/63569063/merge-two-dataframes-based-on-nearest-matches-between-pairs-of-column-values    
traindf = pd.read_csv("train.csv")
weatherdf = pd.read_csv("weather.csv")
traindf["start"] = pd.to_datetime(traindf["start"])
weatherdf["start"] = pd.to_datetime(weatherdf["start"]) #changing times out of string into datetime
weatherdf = weatherdf.sort_values(['start']) #sorting time values for merge
df = pd.merge_asof(traindf, weatherdf, on = ['start'], left_by = ['state'], right_by = ['state'], 
                   tolerance = pd.Timedelta('5 hr'), direction = 'nearest') #merge on start times, matching states

#TEST/WEATHER DATA MERGE CELL
testdf = pd.read_csv("test.csv")
test2 = testdf
testdf.rename(columns={"time": "start"}, inplace = True)
testdf["start"] = pd.to_datetime(testdf["start"])
test_df = pd.merge_asof(testdf, weatherdf, on = ['start'], left_by = ['state'], right_by = ['state'], 
                   tolerance = pd.Timedelta('5 hr'), direction = 'nearest')
test_df.drop(columns = ['id_x', 'id_y', 'end', 'state'], inplace=True) #dropping id, end, and state columns
test_df.fillna(0, inplace=True) #chaning nan values to 0
testdf = pd.get_dummies(test_df) #setting dummies for weather so they're not strings (into binary)
testdf['weather_hail'] = testdf['weather_hail'].replace(0) #replacing weather_hail and level_other to 0 into test set since training set has these columns and test doesnt
testdf['level_other'] = testdf['level_other'].replace(0)
testdf.drop(columns = ['lat_y','lng_y', 'side_L', 'side_R'], inplace=True)
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html

#DROP/RENAME NECESSARY COLUMNS IN TRAIN/WEATHER DATAFRAME
df.drop(columns = ['id_x','side', 'id_y', 'end_x', 'end_y', 'state', 'lat_y', 'lng_y'], inplace = True) #simplifying dataframe
df.rename(columns={"start": "time", "lng_x":"lng", "lat_x":"lat"}, inplace = True)

#RENAME AND DROP NECESSARY COLUMNS FROM TEST/WEATHER DATAFRAME
testdf.rename(columns={"start": "time", "lng_x":"lng", "lat_x":"lat"}, inplace = True)
testdf['time'] = testdf['time'].dt.hour
testdf.head()

#SPLIT TRAINING DATAFRAME INTO TEST AND TRAIN SLICES TO TEST MODEL
ytrain = df['event']#set ytrain to event column in training date
x_train = df[['lat', 'lng', 'time', 'weather', 'level']] #columns in x train
x_train.fillna(0, inplace=True) #fill nan with 0
x_train = pd.get_dummies(x_train) #dummies for weather and level data (binary)
X_train, X_test, y_train, y_test = train_test_split(x_train, ytrain, stratify=ytrain, test_size=0.10, random_state=42) #split training data into train and test sets
X_train['time'] = X_train['time'].dt.hour #chaning time to hours (out of 23 since counting 0) instead of datetime
X_test['time'] = X_test['time'].dt.hour
X_train.head()
#https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html
#https://docs.python.org/3/library/datetime.html

#RANDOM FOREST CLASSIFIER FIT, PREDICT (ON ACTUAL TEST DATA), AND SCORE ON TEST SLICE FROM ABOVE
clf = RandomForestClassifier(n_estimators =200) # random forest classifier with 200 estimators
clf.fit(X_train, y_train) #^^grid search to find params??
pred = clf.predict(testdf)
score = clf.score(X_test, y_test)

#PREDICT ON TEST SLICE FROM SPLIT
pred = clf.predict(X_train) #predicting on X-train from split training data

#ACCURACY SCORE TESTING
from sklearn.metrics import accuracy_score
print(accuracy_score(pred, y_train))# testing accuracy of classifier

#RESULTS OUTPUT
results = test2.copy() #copy of test import to get id column
results['event'] = pred
results.to_csv('results.csv', columns=['id', 'event'], index=False)
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html