import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Create your connection with database file.
cnx = sqlite3.connect(r'C:/Users/Akumarx084109/Desktop/Abhineet/DATA_SCIENCE/ACADGILD/Project/database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)

# Data preprosessing

df.dropna(subset=['defensive_work_rate', 'attacking_work_rate', 'preferred_foot'], inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="0"].index,inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="1"].index,inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="2"].index,inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="3"].index,inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="4"].index,inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="5"].index,inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="6"].index,inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="7"].index,inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="8"].index,inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="9"].index,inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="_0"].index,inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="es"].index,inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="ean"].index,inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="o"].index,inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="ormal"].index,inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="tocky"].index,inplace=True)
df.drop(df.loc[df["defensive_work_rate"]=="None"].index,inplace=True)
df.drop(df.loc[df["attacking_work_rate"]=="None"].index,inplace=True)
# Replacing NAN values with median
df.fillna(df.median,inplace=True)

#column "id", "player_fifa_api_id", "player_api_id" and "date" seems  to be irrelevant for prediction. 
#that's why not taking into consideration for model building.

X = df.iloc[:,5:]
Y = df.iloc[:,4]

# Encoding categorical data

labelencoder = LabelEncoder()
X["preferred_foot"] = labelencoder.fit_transform(X["preferred_foot"])
X["attacking_work_rate"] = labelencoder.fit_transform(X["attacking_work_rate"])
X["defensive_work_rate"] = labelencoder.fit_transform(X["defensive_work_rate"])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/4, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#predicting accuracy of model
accuracy = regressor.score(X_test,y_test)
