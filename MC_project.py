
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import datetime as dt
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

## Retrieve stock data
sx = pdr.get_data_yahoo("^STOXX50E", start="2018-01-01", end="2021-04-09")
sp = pdr.get_data_yahoo("^GSPC", start="2018-01-01", end="2021-04-09")
sm = pdr.get_data_yahoo("^SSMI", start="2018-01-01", end="2021-04-09")

## Create DataFrame to store returns of each stock
basket = pd.DataFrame()
basket["sx_close"] = sx['Close'].pct_change().dropna()
basket["sp_close"] = sp['Close'].pct_change().dropna()
basket["sm_close"] = sm['Close'].pct_change().dropna()

## Calculate Correlation matrix
corr_mat = basket.corr() # correlation of returns

## find yearly expected stock returns for each underlying asset

mu_sx = np.mean(np.log(1 + sx['Close'].pct_change().dropna()))*252
mu_sp = np.mean(np.log(1 + sp['Close'].pct_change().dropna()))*252
mu_sm = np.mean(np.log(1 + sm['Close'].pct_change().dropna()))*252
print("returns EURO STOXX 50: ", mu_sx)
print("returns S&P 500: ", mu_sp)
print("returns Swiss Market index: ", mu_sm)

print()

# find annualized stock volatility for each underlying asset
sig_sx = np.std(np.log(1 + sx['Close'].pct_change().dropna()))*(252**.5)
sig_sp = np.std(np.log(1 + sp['Close'].pct_change().dropna()))*(252**.5)
sig_sm = np.std(np.log(1 + sm['Close'].pct_change().dropna()))*(252**.5)

print("volatility EURO STOXX 50: ", sig_sx)
print("volatility S&P 500: ", sig_sp)
print("volatility Swiss Market index: ", sig_sm)

# define annualized dividend yield for each underlying asset
d_sx = 1.72/100 # https://www.marketbeat.com/stocks/NYSEARCA/FEZ/dividend/
d_sp = 1.45/100 # https://ycharts.com/indicators/sp_500_dividend_yield
d_sm = 2.31/100 # http://www.dividendsranking.com/Index/SMI.php

print("divident yield EURO STOXX 50: ", d_sx)
print("divident yield S&P 500: ", d_sp)
print("divident yield Swiss Market index: ", d_sm)

## define key parameters for MC simulation
T = 2
r = 0.01 # the risk free rate is about 1% based on google source
n = 2*252 
dt = T/(2*252)
N = 10**5
C = np.linalg.cholesky(corr_mat)

## initialize array and store initial stock price
X_sx=np.zeros((N,int(n)+1))
X_sx[:,0]=sx['Close'][-1]

X_sp=np.zeros((N,int(n)+1))
X_sp[:,0]=sp['Close'][-1]

X_sm=np.zeros((N,int(n)+1))
X_sm[:,0]=sm['Close'][-1]

coup = 6.25
denom = 1000;
init_fix = np.array([3857.07, 3917.45, 11130.28])
tl = np.array([0.95, 0.85, 0.75, 0.65, 0.59])
br = np.zeros((5,3))
## store the barrier levels for each asset in 5 by 3 matrix
for i in range(0,5):
  br[i,:] = init_fix*tl[i]
## define observation day array
obs = np.array([251])
obs = np.append(obs, obs[-1]+63)
obs = np.append(obs, obs[-1]+63)
obs = np.append(obs, obs[-1]+63)
obs = np.append(obs, obs[-1]+63)


## Simulation of Geometric Brownian Motion
for i in range(1,n+1):
  W = np.random.normal(0, 1, (N,3))
  Z=np.matmul(W,C);    #W*C;

  X_sx[:,i] = X_sx[:,i-1]*np.exp((mu_sx - d_sx - (sig_sx**2)/2)*dt + sig_sx*np.sqrt(dt)*Z[:,0])
  X_sp[:,i] = X_sp[:,i-1]*np.exp((mu_sp - d_sp - (sig_sp**2)/2)*dt + sig_sp*np.sqrt(dt)*Z[:,1])
  X_sm[:,i] = X_sm[:,i-1]*np.exp((mu_sm - d_sm - (sig_sm**2)/2)*dt + sig_sm*np.sqrt(dt)*Z[:,2])

## Define Payoff function that takes each assets' price movements, number of simulations
## and quarterly coupon payment amount in CHF 
def calculatePayoff(X_sx, X_sp, X_sm, N, coup):
  payoff = np.ones(N)*coup*(np.exp(-r*63/252)+np.exp(-r*2*63/252)+np.exp(-r*3*63/252))
  for i in range(0, N):
    for j in range(0,len(obs)):
      if(X_sx[i,obs[j]] > br[j,0] and X_sp[i,obs[j]] > br[j,1] and X_sm[i,obs[j]] > br[j,2]):
        if(j != 4):
          payoff[i] += np.exp(-r*obs[j]/252)*(coup + denom);
        else:
          perf = [X_sx[i,obs[j]]/init_fix[0], X_sp[i,obs[j]]/init_fix[1], X_sm[i,obs[j]]/init_fix[2]]
          worst = np.min(perf)
          payoff[i] += np.exp(-r*obs[j]/252)*(coup + denom*worst)
        break;
      else:
        if(j == 4):
          payoff[i] += np.exp(-r*obs[j]/252)*(coup + denom)
        else:
          payoff[i] += np.exp(-r*obs[j]/252)*(coup)

  return payoff
        
# calculate payoff of the N simulations
payoff = calculatePayoff(X_sx, X_sp, X_sm, N, 6.25)

print("Fair Price (mean) for",N,"simulations:", np.mean(payoff))

print("Standard Error of Estimator for",N,"simulations:",np.std(payoff)/N)

## Coupon Rate mesh to evaluate fair price per each coupon rate
coupon_rates = np.linspace(0, 10, 100, endpoint=True)*250/100

## Calculate fair price per each coupon rate and store in array
prices = np.array([])
for cpr in coupon_rates[1:]:
  prices = np.append(prices, np.mean(calculatePayoff(X_sx, X_sp, X_sm, N, cpr)))
## store coupon rate and respective fair prices in DataFrame
crVp = pd.DataFrame()
crVp['coupon_rates'] = coupon_rates[1:]*100/250
crVp['fair_price'] = prices

## plot coupon rates vs fair prices
ax = sns.lineplot(data=crVp, x="coupon_rates", y="fair_price")
ax.set_title('Coupon Rate vs Fair Price')

## Antithetic Variates Method 
N = 10**5


## Anti-thetic variates method 
X_sx1=np.zeros((int(N/2),int(n)+1))
X_sx1[:,0]=sx['Close'][-1]

X_sp1=np.zeros((int(N/2),int(n)+1))
X_sp1[:,0]=sp['Close'][-1]

X_sm1=np.zeros((int(N/2),int(n)+1))
X_sm1[:,0]=sm['Close'][-1]

X_sx2=np.zeros((int(N/2),int(n)+1))
X_sx2[:,0]=sx['Close'][-1]

X_sp2=np.zeros((int(N/2),int(n)+1))
X_sp2[:,0]=sp['Close'][-1]

X_sm2=np.zeros((int(N/2),int(n)+1))
X_sm2[:,0]=sm['Close'][-1]
## Simulate stock prices
for i in range(1,n+1):
  W1 = np.random.normal(0, 1, (int(N/2),3))
  Z1=np.matmul(W1,C)    #W*C;
  Z2=-1*Z1
  X_sx1[:,i] = X_sx1[:,i-1]*np.exp((mu_sx - d_sx - (sig_sx**2)/2)*dt + sig_sx*np.sqrt(dt)*Z1[:,0])
  X_sp1[:,i] = X_sp1[:,i-1]*np.exp((mu_sp - d_sp - (sig_sp**2)/2)*dt + sig_sp*np.sqrt(dt)*Z1[:,1])
  X_sm1[:,i] = X_sm1[:,i-1]*np.exp((mu_sm - d_sm - (sig_sm**2)/2)*dt + sig_sm*np.sqrt(dt)*Z1[:,2])

  X_sx2[:,i] = X_sx2[:,i-1]*np.exp((mu_sx - d_sx - (sig_sx**2)/2)*dt + sig_sx*np.sqrt(dt)*Z2[:,0])
  X_sp2[:,i] = X_sp2[:,i-1]*np.exp((mu_sp - d_sp - (sig_sp**2)/2)*dt + sig_sp*np.sqrt(dt)*Z2[:,1])
  X_sm2[:,i] = X_sm2[:,i-1]*np.exp((mu_sm - d_sm - (sig_sm**2)/2)*dt + sig_sm*np.sqrt(dt)*Z2[:,2])

# calculate payoffs for each set of simulations
payoff1 = calculatePayoff(X_sx1, X_sp1, X_sm1 , int(N/2), 6.25)
payoff2 = calculatePayoff(X_sx2, X_sp2, X_sm2, int(N/2),  6.25)

total_payoff = np.append(payoff1, payoff2)

print("Fair Price (mean) for",N,"simulations:", np.mean(total_payoff))

print("Standard Error of Estimator for",N,"simulations:",np.std(total_payoff)/N)

## Defing payoff function for ABRC for a single underlying asset
def calculatePayoff_single(X_sx, N, coup):
  payoff = np.ones(N)*coup*(np.exp(-r*63/252)+np.exp(-r*2*63/252)+np.exp(-r*3*63/252))
  for i in range(0, N):
    for j in range(0,len(obs)):
      if(X_sx[i,obs[j]] > br[j,0]):
        if(j != 4):
          payoff[i] += np.exp(-r*obs[j]/252)*(coup + denom);
        else:
          worst = X_sx[i,obs[j]]/init_fix[0]
          payoff[i] += np.exp(-r*obs[j]/252)*(coup + denom*worst)
        break;
      else:
        if(j == 4):
          payoff[i] += np.exp(-r*obs[j]/252)*(coup + denom)
        else:
          payoff[i] += np.exp(-r*obs[j]/252)*(coup)

  return payoff

## apply control variate method
s_payoff = calculatePayoff_single(X_sx, N, coup)
b=np.corrcoef(payoff,s_payoff)[0,1]
payoff_cv = payoff - b*(s_payoff - np.mean(s_payoff))
print("(Control Variates) Fair Price (mean) for",N,"simulations:", np.mean(payoff_cv))
print("(Control Variates) Standard Error of Estimator for",N,"simulations:",np.std(payoff_cv/N))