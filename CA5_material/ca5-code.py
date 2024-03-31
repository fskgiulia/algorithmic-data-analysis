# %% [markdown]
# # Coding Assignment 5 - Analysis of a spatio-temporal dataset

# %%
pip install statsmodels

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

# %% [markdown]
# VARIABLES:
# * instant: record index
# * dteday : date
# * season : season (1:springer, 2:summer, 3:fall, 4:winter)
# * yr : year (0: 2011, 1:2012)
# * mnth : month (1 to 12)
# * hr : hour (0 to 23)
# * holiday : weather day is holiday or not
# * weekday : day of the week
# * workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
# * weathersit :
#     * 1: Clear, Few clouds, Partly cloudy, Partly cloudy
#     * 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#     * 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#     * 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# * temp : Normalized temperature in Celsius. The values are divided to 41 (max)
# * atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
# * hum: Normalized humidity. The values are divided to 100 (max)
# * windspeed: Normalized wind speed. The values are divided to 67 (max)
# * casual: count of casual users
# * registered: count of registered users
# * cnt: count of total rental bikes including both casual and registered

# %%
pd.set_option('display.float_format', '{:.2f}'.format)
plt.rcParams['figure.figsize'] = (10, 6)

# %%
df=pd.read_csv('day.csv')
df["dteday"] = pd.to_datetime(df["dteday"])# converts a column to date format
df.set_index("dteday", inplace=True) #makes the column an index for ease of analysis
df.head()

# %%
df.shape[0] 

# %%
df.info()

# %%
df.isna().sum()

# %%
df[df.duplicated]

# %%
df.describe()

# %%
df[df.cnt == 8714] # The maximum number of rented bicycles

# %%
df[df.cnt == 22] # The minimum number of rented bicycles

# %% [markdown]
# The temperature is normalized so we have to multiply by 41
# 
# * The maximum actual air temperature is 35 degrees Celsius , the minimum temperature is 2 degrees
# * The maximum perceived air temperature is 42 degrees Celsius , the minimum temperature is almost 4 degrees
# * The maximum humidity is 97, the minimum humidity is 0, but this is a measurement error. Most likely the minimum humidity is close to 0 but the accuracy is not
# * The maximum wind speed is 34 meters per second , the minimum wind speed is 1.5 meters per second
# * The maximum number of rented bicycles per day was 3,410 people - it was 2012-09-15. The minimum number of rented bicycles per day was 2 people - it was 2012-10-29. It's all logical. On May 15, it is already warm, dry outside and people go for a ride after spring. And on September 29: Hurricane Sandy, quite rainy and cold, so as not to promote bike rental(I don't see any point in analyzing and drawing conclusions on unregistered and registered users separately, since the overall picture is more important to us)
# 

# %%
plt.figure(figsize=(20, 8))
sns.lineplot(df.cnt)
plt.title('Number of rented bicycles from time to time')
plt.xlabel('Date')
plt.ylabel('Number of rented bicycles')

# %%
df_2011 = df[df.index.year == 2011]# data for 2011
df_2012 = df[df.index.year == 2012]# data for 2012

# %%
plt.plot(df_2011.index, df_2011.cnt.rolling(window=30).mean(), label='2011')
plt.plot(df_2012.index, df_2012.cnt.rolling(window=30).mean(), label='2012')

plt.xlabel('Date')
plt.ylabel('Number of rented bicycles')
plt.title('The data is aggregated by 30 days')
plt.legend()
plt.show()

# %%
plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True)

# %%
df

# %%
from sklearn.ensemble import IsolationForest

features = ['cnt', 'weathersit', 'hum', 'temp', 'atemp']
X_anomaly = df[features]

isolation_forest = IsolationForest(contamination=0.05)  
isolation_forest.fit(X_anomaly)

outliers = isolation_forest.predict(X_anomaly)

df['anomaly'] = outliers

print("Details of Detected Anomalies:")
print(df[df['anomaly'] == -1])


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.scatter(df.index, df['cnt'], c=df['anomaly'], cmap='viridis', marker='o', alpha=0.7)
plt.title('Bike Rental Counts with Anomalies Highlighted')
plt.xlabel('Date')
plt.ylabel('Total Rental Bike Count')
plt.colorbar(label='Anomaly (1: Normal, -1: Anomaly)')
plt.grid(True)
plt.show()


# %%
# different contamination levels
import matplotlib.pyplot as plt

contamination_levels = [0.01, 0.02, 0.05, 0.1]

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
axs = axs.flatten()

for i, contamination in enumerate(contamination_levels):
    isolation_forest = IsolationForest(contamination=contamination, random_state=42)
    isolation_forest.fit(X_anomaly)
    outliers = isolation_forest.predict(X_anomaly)
    
    df['anomaly'] = outliers

    axs[i].scatter(df.index, df['cnt'], c=df['anomaly'], cmap='viridis', marker='o', alpha=0.7)
    axs[i].set_title(f'Contamination Level: {contamination}')
    axs[i].set_xlabel('Date')
    axs[i].set_ylabel('Total Rental Bike Count')
    axs[i].grid(True)

plt.tight_layout()
plt.show()


# %%
# 2011
features = ['cnt', 'weathersit', 'hum', 'temp', 'atemp']
X_anomaly = df_2011[features]

isolation_forest = IsolationForest(contamination=0.05)  
isolation_forest.fit(X_anomaly)

outliers = isolation_forest.predict(X_anomaly)

df_2011['anomaly'] = outliers

# 2012
features = ['cnt', 'weathersit', 'hum', 'temp', 'atemp']
X_anomaly = df_2012[features]

isolation_forest = IsolationForest(contamination=0.05)  
isolation_forest.fit(X_anomaly)

outliers = isolation_forest.predict(X_anomaly)

df_2012['anomaly'] = outliers


# %%
plt.figure(figsize=(16, 14))

plt.subplot(2, 1, 1)
plt.scatter(df_2011.index, df_2011['cnt'], c=df_2011['anomaly'], cmap='viridis', marker='o', alpha=0.7)
plt.title('Anomaly Detection - 2011')
plt.xlabel('Date')
plt.ylabel('Total Rental Bike Count')
plt.colorbar(label='Anomaly (1: Normal, -1: Anomaly)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.scatter(df_2012.index, df_2012['cnt'], c=df_2012['anomaly'], cmap='viridis', marker='o', alpha=0.7)
plt.title('Anomaly Detection - 2012')
plt.xlabel('Date')
plt.ylabel('Total Rental Bike Count')
plt.colorbar(label='Anomaly')
plt.grid(True)

plt.tight_layout()
plt.show()


