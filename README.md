# [Time series] Real Estate & Digital Nomads: Is Lisbon the New San Francisco? - By Sebastião Oliveira and Juliette Hanau

# Context 
Our project is about the real estate and the impact of digital nomads. We focused on what happened to the rent prices of Lisbon that had similar characteristics as San Francisco as a growing technological scene. 
In order to do this, we used 2 datasets:

 - For San Francisco, we used the rent index, which is a statistical indicator that measures the rate of change in rent prices of housing. Source: Zillow
 
 - For Lisbon, we used the Price of the rent per square meter. Source: Webscraping the real estate company website of Idealista
 
The data are from 2015 to 2022, per years.

# How did we used those data? 
We turned them into time series data to be able to build a model that can forecast the price. We used the ARIMA model who had a great mean squared error: Lisbon: 0.13 and San Francisco: 16.8 

We also used Dynamic Time Warping to see if they were any relation between the Lisbon and San Francisco prices evolution. 


# Deliverables 
We created a dashboard on Dash that you can find here: [link to add]

Visualization:
- the evolution of the rent index in San Francisco from March 2015 to November 2022.
- the evolution of the rent per squared meter in lisbon from March 2015 to November 2022.
- forecast plot of Lissbon
- forecast plot of SF
- Dynamic Time Warping plot 


