import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import ta
import pyhht
import scipy
from scipy import fftpack, stats
import matplotlib.ticker as mtick


def plot_distributions(results, ticker, P, x,  rows, cols, save:bool):

    fig, ax = plt.subplots(1, 1, sharex = True, sharey = False, figsize=(20,8))

    ax[0,0].hist(results, bins = 100, color = 'cornflowerblue', label = 'Distribution', density=True)
    ax[0,0].plot(x, P, 'k', linewidth = 1)
    ax[0,0].set_title(f'{ticker} Returns', size=15)
    ax[0,0].tick_params(labelsize = 15)

    if save is True:
        plt.savefig('Distribution of XOM Daily Returns, Simulated Returns & OI.png')

    plt.show()

def scale_dataset(data, min:bool):

    if min is False:
        scaler = StandardScaler()
        scaler.fit(data[:3 * len(data) // 5])
        #scaler.fit(data)
        scaled_dataset = scaler.transform(data)

        scaled_dataset = pd.DataFrame(data=scaled_dataset, index=data.index, columns=data.columns)

    if min is True:
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(data[:3 * len(data) // 5])
        #scaler.fit(data)
        scaled_dataset = scaler.transform(data)

        scaled_dataset = pd.DataFrame(data=scaled_dataset, index=data.index, columns=data.columns)

    return scaled_dataset

def create_datasets(T, T_target, targets, ticker, scaled_data, multi_step:bool):

    if multi_step is True:
        D = len(scaled_data.T)
        X_obs = []
        Y_obs = []
        Target_obs = targets

        for i in range(len(scaled_data) - T - T_target - 1):

            x = scaled_data[i:i + T]
            X_obs.append(x)

            y = Target_obs[(i+T) : (i+T + T_target)]
            Y_obs.append(y)

        X_obs = np.array(X_obs)
        Y_obs = np.array(Y_obs)
        print(f'X shape : {X_obs.shape}')
        print(f'Y shape : {Y_obs.shape}')

        X_obs = X_obs.reshape(-1, T, D)
        Y_obs = Y_obs.reshape(-1, T_target)

        print('Shape of X:', X_obs.shape, 'Shape of Y:', Y_obs.shape)
        
    else:
        D = len(scaled_data.T)
        X_obs = []
        Y_obs = []
        Target_obs = targets

        for i in range(len(scaled_data) - T):
            x = scaled_data[i:i + T]
            X_obs.append(x)

            y = Target_obs[i + T]
            Y_obs.append(y)

        X_obs = np.array(X_obs).reshape(-1, T, D)
        Y_obs = np.array(Y_obs).reshape(-1, 1)

        print('Shape of X:', X_obs.shape, 'Shape of Y:', Y_obs.shape)

    return X_obs, Y_obs


def data_split(X, Y):
    x_train = torch.from_numpy(X[:3 * len(X) // 5].astype(np.float32))
    y_train = torch.from_numpy(Y[:3 * len(Y) // 5].astype(np.float32))

    x_test = torch.from_numpy(X[-1 * len(X) // 5:].astype(np.float32))
    y_test = torch.from_numpy(Y[-1 * len(Y) // 5:].astype(np.float32))

    x_val = torch.from_numpy(X[-2 * len(X) // 5: -1 * len(X) // 5].astype(np.float32))
    y_val = torch.from_numpy(Y[-2 * len(Y) // 5: -1 * len(Y) // 5].astype(np.float32))

    return x_train, y_train, x_val, y_val, x_test, y_test


def plotter(imfs, hilbert):
    t = np.arange(0, len(imfs.T), 1)
    t1 = np.arange(0, len(hilbert.T), 1)
    fig, axs = plt.subplots(len(hilbert), sharex=True, figsize=(8, 24))
    for i in range(len(hilbert)):
        axs[i].plot(t, hilbert[i], label='Hilbert Transform', color='lightcoral', title=f'IMF {i}')
        axs[i].plot(t1, imfs[i], label='IMF Component', color='cornflowerblue', title=f'cIMF {i}')
        axs[i].legend()


def compute_amplitude(imfs, hilbert):
    results = np.zeros((len(imfs), len(imfs.T)))

    for i in range(len(hilbert)):
        Z_t = imfs[i] ** 2 + hilbert[i] ** 2
        Z_t = np.sqrt(Z_t)
        results[i] = Z_t

    return results


def compute_instantaneous_frequency(imfs, hilbert):

    results = np.zeros((len(imfs), len(imfs.T)))

    for i in range(len(hilbert)):
        Z_t = hilbert[i]/imfs[i]
        k = np.arctan(Z_t)
        results[i] = k

    return results

def create_HHT_dataframe(imfs, hilbert, amplitude, inst_freq, returns, timeframe, data):
    if type(imfs) != np.ndarray:
        imfs = np.array(imfs)

    if type(hilbert) != np.ndarray:
        hilbert = np.array(hilbert)

    if type(amplitude) != np.ndarray:
        amplitude = np.array(amplitude)

    if type(inst_freq) != np.ndarray:
        inst_freq = np.array(inst_freq)

    IMF_w_cols = [f'IMF_{i + 1}' for i in range(len(imfs))]
    Complex_IMF_w_cols = [f'Complex IMF_{i + 1}' for i in range(len(hilbert))]
    Amplitude_w_cols = [f'Amp_{i + 1}' for i in range(len(amplitude))]
    IF_w_cols = [f'IF_{i + 1}' for i in range(len(inst_freq))]

    columns = IMF_w_cols + Complex_IMF_w_cols + Amplitude_w_cols + IF_w_cols

    a = pd.DataFrame(data=imfs.reshape(-1, len(imfs)), index=data.index, columns=IMF_w_cols)
    b = pd.DataFrame(data=hilbert.reshape(-1, len(hilbert)), index=data.index, columns=Complex_IMF_w_cols)
    c = pd.DataFrame(data=amplitude.reshape(-1, len(amplitude)), index=data.index, columns=Amplitude_w_cols)
    d = pd.DataFrame(data=inst_freq.reshape(-1, len(inst_freq)), index=data.index, columns=IF_w_cols)
    e = pd.DataFrame(data=returns.reshape(-1), index=data.index, columns=[f'{timeframe} Log Returns'])

    dataframe = a.join((b, c, d), how='left')

    return dataframe


def find_lagged_correlations(dataframe, returns):
    lagged = dataframe.shift(1)
    check = pd.Series(returns, index=dataframe.index)
    for i in lagged.columns:
        print(f'Correlation between {i} and the Returns: {lagged[i].corr(check)}')


def create_features(df, ticker):
    df[f'{ticker} Previous Close'] = df[f'{ticker} Closing price'].shift(1)
    df[f'{ticker} Daily Returns'] = df[f'{ticker} Closing price'] / df['Previous Close'] - 1
    df[f'{ticker} 2 Day Price Return'] = df[f'{ticker} Closing price'] / df[f'{ticker} Closing price'].shift(2) - 1
    df[f'{ticker} 3 Day Price Return'] = df[f'{ticker} Closing price'] / df[f'{ticker} Closing price'].shift(3) - 1
    df[f'{ticker} 5 Day Price Return'] = df[f'{ticker} Closing price'] / df[f'{ticker} Closing price'].shift(5) - 1
    df[f'{ticker} CCI'] = ta.trend.cci(df[f'{ticker} High price'], df[f'{ticker} Low price'],
                                       df[f'{ticker} Closing price'], window=20, constant=0.015)
    # XOM_full_data['XOM Price-Dividend Ratio'] =  XOM_full_data['XOM Dividend_payout_ratio'] / XOM_full_data['XOM Price_earning ratio']

    return df


def compute_moments(returns):
    mean = np.mean(returns)
    var = np.var(returns)
    std = returns.std()
    min = returns.min()
    max = returns.max()
    x = np.linspace(min, max, len(returns))
    P = scipy.stats.norm.pdf(x, True_returns_mean, True_returns_std)

    skew = scipy.stats.skew(returns)
    kurtosis = scipy.stats.kurtosis(returns)

    print(
        f"Mean: {mean:.4f}, Variance: {var:.4f}, Standard Deviation: {std:.4f}, Skewness: {skew:.4f}, Kurtosis: {kurtosis:.4f}")

    return mean, var, std, P, skew, kurtosis, x

def make_predictions(model, x, targets, T, D, Output_size, multi_step = bool):
    if multi_step is True:

        preds = np.zeros((len(targets), Output_size))

        i = 0

        while i + 1 < len(targets):
            input = x[i].reshape(1, T, D)
            # print(input)

            output = model(input)  # [0, 0].item()
            # print(output)

            i += 1

            preds[i] = output.detach().cpu().numpy()

    else:
        preds = []

        i = 0

        while len(preds) < len(targets):
            input = x[i].reshape(1, T, D)

            output = model(input)[0, 0].item()

            i += 1

            preds.append(output)

    return preds



