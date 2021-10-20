#####################################################################################
# Time-Series Forecasting 
# Yahoo Stock Price Prediction 
#
# Created by: Pradipti Thakur
#####################################################################################

# Installing and loading required libraries
library(rio)
library(ggplot2)
library(forecast)
library(tseries)
library(tidyverse)

# Reading the file
stock= read.csv("yahoo_stock.csv")

#####################################################################################
# Preliminary Exploratory Data Analysis
#
# We will perform some data cleaning and plot graphs.
#####################################################################################

str(stock) # date is at the factor level
dim(stock) # 1825*7
summary(stock) # no missing values
stock$DATE= as.Date(stock$Date) # converting dates

# Plotting High, Low, Open, Close and Volume for the stock data
ggplot() +
  geom_line(data= stock, aes(DATE, High, colour= "High")) +
  geom_line(data= stock, aes(DATE, Low, colour= "Low")) +
  #geom_line(data= stock, aes(DATE, Open, colour="Open")) +
  #geom_line(data= stock, aes(DATE, Close, colour="Close")) +
  #geom_line(data= stock, aes(DATE, Volume, colour="Volume")) +
  ylab("Stock prices") +
  xlab("")+
  scale_color_manual("",
                     breaks = c("High", "Low", "Open", "Close", "Volume"),
                     values= c("#7CAE00", "#F8766D", "#C77CFF", "#00BFC4", "Black"))

# Creating a time series object 
high_ts= ts(stock[, c("High")])

# tsclean for cleaning this data, removing outliers etc.
stock$Clean = tsclean(high_ts)

# Graph cleaned data
ggplot() +
  geom_line(data= stock, aes(DATE, Clean)) + ylab('Cleaned Daily High') +
  xlab(" ")

# Showing weekly, monthly and daily moving averages for the data
stock$MA_7= ma(stock$Clean, order= 7) # for weekly moving average
stock$MA_30= ma(stock$Clean, order=30) # for monthly moving average

# Plotting all moving averages
ggplot() +
  geom_line(data= stock, aes(DATE, Clean, colour= "Daily average")) +
  geom_line(data= stock, aes(DATE, MA_7, colour= "Weekly moving average")) +
  geom_line(data= stock, aes(DATE, MA_30, colour="Monthly moving average")) +
  ylab("Stock prices") +
  xlab("") +
  scale_color_manual("",
                     breaks = c("Weekly moving average","Monthly moving average","Daily average"),
                     values= c("Blue","Red","Green"))

#####################################################################################
# Decomposition of Data
#
# Our data consists of both seasonality and trend. 
# Therefore, we will decompose the data.
#####################################################################################

org_weekly_d= ts(na.omit(stock$MA_7), frequency = 30)
decomp= stl(org_weekly_d, s.window = "periodic")
deseasonal_d= seasadj(decomp) # we are removing seasonality here
plot(decomp)

# First plot shows our data as it came in, it is a moving average of 7.
# Second plot shows seasonality.
# Third plot shows a trend.
# The fourth plot is a remainder plot taking out seasonality and the trend.

# The data is variant and non-stationary. 
# The seasonal line is very stationary.

######################################
# Case-1
#
# adf, acf and pacf for original data.
######################################

# Test for stationarity
# Augmented Dickey-Fuller test 
adf.test(org_weekly_d, alternative = "stationary")
# Dickey-Fuller = -2.9985, Lag order = 12, p-value = 0.1556
# This test shows that the data is not stationary. 
# Autocorrelations and Choosing Model Order 
# ACF plots display correlation between a series and its lags.
Acf(org_weekly_d, main= "ACF Plot")
# Lot of autocorrelations
# PACF plots display correlation between a series and its lags that are explained by previous lags. 
Pacf(org_weekly_d, main="PACF Plot")

######################################
# Case-2
# 
# Removing trend and checking adf.
######################################

# Removing trend using diff to make data stationary. 
dff_series= diff(org_weekly_d, differences=1) # trend removed, seasonality not removed.
plot(dff_series)

adf.test(dff_series, alternative = "stationary")
# Dickey-Fuller = -10.839, Lag order = 12, p-value = 0.01

# Look for spikes at specific lag points of the differentiated series.
Acf(dff_series, main= "ACF for Differenced Series") # determines value of q
Pacf(dff_series, main= "PACF for Differenced Series") # determines the value of p

######################################
# Case-3
# 
# Removing seasonality and checking adf.
######################################

# Removing seasonality to make data stationary. 
adf.test(deseasonal_d, alternative = "stationary")
# Dickey-Fuller = -2.992, Lag order = 12, p-value = 0.1584
# Look for spikes at specific lag points of the differentiated series.
Acf(deseasonal_d, main= "ACF for Deseasonal Series") # determines value of q
Pacf(deseasonal_d, main= "PACF for Deseasonal Series") # determines the value of p

######################################
# Case-4
# 
# Removing seasonality and trend both.
######################################

# Removing trend and seasonality to make data stationary. 
dff_series_deseason= diff(deseasonal_d, differences=1) # trend removed, seasonality also removed.
plot(dff_series_deseason)
adf.test(dff_series_deseason, alternative = "stationary")
# Dickey-Fuller = -10.816, Lag order = 12, p-value = 0.01
# Look for spikes at specific lag points of the differentiated series.
Acf(dff_series_deseason, main= "ACF for Diff and Deseasoned Series") # determines value of q
Pacf(dff_series_deseason, main= "PACF for Diff and Deseasoned Series") # determines the value of p

#####################################################################################
# Use different models to find the best fit.
#
# 1. We will use a benchmark model called snaive. 
# 2. We will also use the Exponential smoothing model.
# 3. We will use the most famous ARIMA model.
#####################################################################################

# Use a benchmark method to forecast. 
fit_naive= snaive(dff_series_deseason) # Residual sd: 9.5974
print(summary(fit_naive))
checkresiduals(fit_naive)
tsdisplay(residuals(fit_naive), lag.max = 45, main= "Residuals from snaive")

# Fit ETS (Exponential Smoothing Models)
# These model only require the original data.
# sigma given by the model is the Residual sd here.
fit_ets= ets(org_weekly_d) # Residual sd: 0.0012
print(summary(fit_ets))
checkresiduals(fit_ets)
tsdisplay(residuals(fit_ets), lag.max = 45, main= "Residuals from ETS")
# This model performs a bit better than the benchmark model.

# Fit an ARIMA model 
# This model removes seasonality and trend when we use D=1 and d=1 respectively.
# Here a square root of sigma squared gives us the residuals.
fit_arima= auto.arima(org_weekly_d, d=1, D=1, stepwise = FALSE, approximation = FALSE, trace = TRUE)
print(summary(fit_arima)) # Residual sd: 3.9786
checkresiduals(fit_arima)
tsdisplay(residuals(fit_arima), lag.max = 45, main= "Residuals from ARIMA")
# This model performs even better than the ETS model, and we will use this model to forecast.

#####################################################################################
# Forecasting the future
#
# We will forecast for a time period of 10 years ahead.
#####################################################################################

fcast_naive= forecast(fit_naive, h= 10*12)
autoplot(snaive(dff_series))
fcast_ets= forecast(fit_ets, h= 10*12)
autoplot(fcast_ets)
fcast_arima= forecast(fit_arima, h= 10*12)
autoplot(fcast_arima)

#####################################################################################