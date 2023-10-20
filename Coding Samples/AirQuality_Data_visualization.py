import numpy as np
import scipy as sp
import pandas
import matplotlib.pylab as plt
#This data set explores the air quality over different neighborhoods over time in NYC
def displayOzone(location):
    #extracting data into pandas dataframe
    AQdf = pandas.read_csv('Air_Quality.csv')
    #dropping irrelevant columns
    AQdf.drop('Message', axis=1, inplace=True)
    #sorting df by Time Period
    AQdf.sort_values('Time Period', axis=0, inplace=True)
    #split AQdf into subdataframe to look at Ozone levels over time
    ozoneDf = AQdf[AQdf['Name'] == "Ozone (O3)"]
    #plit ozoneDF into subdataframe to look at ozone levels in specific location
    ozoneDf = ozoneDf[ozoneDf['Geo Place Name'] == location]
    if ozoneDf.empty: #if the location didn't have any ozone measurements
        print("No O3 measurements for this location.")
    else: 
        time = list(ozoneDf['Start_Date']) #list of start dates from df
        ozoneData = list(ozoneDf['Data Value']) #list of data measurements from df
        plt.plot(time, ozoneData) #plot ozone data
        plt.title("Ozone (O3) Levels in " + location + " between 2009-2020")
        plt.xlabel("Year")
        plt.xticks(rotation=90) #readability
        plt.ylabel("Ozone (ppb)")
        plt.show()
    return
#Manhattan Ozone Plot
displayOzone("Manhattan")
#Bronx Ozone Plot
displayOzone("Bronx")
#Brooklyn Ozone Plot
displayOzone("Brooklyn")
#Queens Ozone Plot
displayOzone("Queens")
#Staten Island Ozone Plot
displayOzone("Staten Island")