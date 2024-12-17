import numpy as np
import scienceplots
import glob
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

root = os.getcwd()
plt.style.use(['science', 'nature', 'std-colors'])

def conc_data_files(x):
    return(x[-18:])

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def fit_gaussian(xdata, ydata):
    
    # xdata = xdata[ydata>= np.max(ydata)*0.80]
    # ydata = ydata[ydata>= np.max(ydata)*0.80]
    amplitude_guess = np.max(ydata) - np.min(ydata)
    initial_guess = [amplitude_guess, xdata[np.argmax(ydata)], np.std(xdata)]
    
    popt, pcov = curve_fit(gaussian, xdata, ydata, p0=initial_guess)
    return popt, pcov

filepath = 'conc_data/'
paths = [f for f in glob.glob(f'{filepath}'+'*.csv')]
files = sorted(paths, key=conc_data_files)

data = []
peak_vals_array = []
peak_yvals_array = []
fits_array = []

for f in files:
    filename = f'{f}'
    resmaxfilepath = os.path.join(
        root,
        filename)
    spectrum = np.genfromtxt(
        fname=resmaxfilepath,
        delimiter=",",
        skip_header=1,
        unpack=True) 
    data.append(spectrum)

labels = ['0 uL', '10 uL', '20 uL', '30 uL', '40 uL', '50 uL']
# labels = ['0 ul', '10 ul', '20 ul', '30 ul']

max_array = []
for i in range(len(data)):
    x = np.linspace(3,-3,len(data[i]))
    
    popt, pcov = fit_gaussian(x, data[i])
    peak = popt[1]
    peak_y = gaussian(peak, *popt)
    yfit = gaussian(x, *popt)
    max_peak = x[np.argmax(data[i])]
    
    max_array.append(max_peak)
    peak_vals_array.append(peak)
    peak_yvals_array.append(peak_y)
    
    data[i] = data[i] - np.min(data[i])
    yfit = yfit - np.min(yfit)
    fits_array.append(yfit)

colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']
x_conc = [0, 10, 20, 30, 40, 50]

fig, (ax, ax1) = plt.subplots(2, 1, figsize=[3, 4.5])
for i in range(len(data)):
    x_volts = np.linspace(3,-3,len(data[i]))
    
    ax.plot(x_volts, data[i], alpha=0.5)
    ax.plot(x_volts, fits_array[i], '--', color=colors[i], label=labels[i])
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.legend()

ax1.plot(x_conc, peak_vals_array, label='Gauss fit')
ax1.plot(x_conc, max_array, label='Max val')
# ax1 = ax.twinx()
# ax1.plot(x_conc, peak_yvals_array, color='r')
ax1.set_xlabel('Concentration (ul)')
ax1.set_ylabel('Peak value (V)')
ax1.legend()

plt.savefig('conc_plot.png', dpi=600)