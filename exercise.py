############################################################
# Modelling Forecast Uncertainty (Workshop Exercises)
# Harsha Halgamuwe Hewage | DL4SG, Cardiff University
# Date: 2026-02-25
#
# Goal:
#   1) Model-based uncertainty (ARIMAX distributions + prediction intervals)
#   2) Bootstrap uncertainty (sample paths + bootstrap intervals)
#   3) Conformal prediction (split/cv style calibration -> valid PI)
#
# How to use this script:
#   - Run section-by-section.
#   - Do NOT rush to the end. Each "YOUR TURN" is an exercise.
#
# Dependencies:
#   pip install -r requirements.txt
############################################################


############################
# 0) Housekeeping
############################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
np.random.seed(123)

# Nicer plot defaults
plt.rcParams.update({
    'figure.figsize': (12, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
})
sns.set_style('whitegrid')


############################
# 1) Load data + create time series
############################

# This script expects the dataset:
#   data/brazil_dengue.csv

brazil_dengue = pd.read_csv('data/brazil_dengue.csv')

# Parse time_period to datetime (monthly)
brazil_dengue['time_period'] = pd.to_datetime(brazil_dengue['time_period'])

# Quick check
print(brazil_dengue.head())
print("\nLocations and row counts:")
print(brazil_dengue.groupby('location').size().reset_index(name='n'))

# ---- Plot one series (Bahia) ----
bahia_all = brazil_dengue[brazil_dengue['location'] == 'Bahia'].copy()

fig, ax = plt.subplots()
ax.plot(bahia_all['time_period'], bahia_all['disease_cases'])
ax.set_xlabel('Month')
ax.set_ylabel('Disease Cases')
ax.set_title('Dengue Cases – Bahia')
plt.tight_layout()
plt.show()

# =========================
# YOUR TURN (Exercise 1)
# =========================
# 1) Change "Bahia" to another location you see in the data.
# 2) Plot disease_cases again.
# 3) Describe in 1 sentence: Is it seasonal? Are there spikes?


############################
# 2) Train / Future split (simple)
############################

# Train: up to 2015-12
# Future: 2016-01 onwards (exogenous variables available, disease_cases removed)

train = brazil_dengue[brazil_dengue['time_period'] <= '2015-12-31'].copy()
future = brazil_dengue[brazil_dengue['time_period'] >= '2016-01-01'].copy()

print("Train period:", train['time_period'].min(), "–", train['time_period'].max())
print("Future period:", future['time_period'].min(), "–", future['time_period'].max())


############################################################
# 3) MODEL-BASED UNCERTAINTY (ARIMAX via statsmodels)
############################################################

# Filter Bahia training data
bahia_train = train[train['location'] == 'Bahia'].sort_values('time_period').reset_index(drop=True)
bahia_future = future[future['location'] == 'Bahia'].sort_values('time_period').reset_index(drop=True)

# Fit SARIMAX with automatic order selection via pmdarima
try:
    import pmdarima as pm
    auto_fit = pm.auto_arima(
        bahia_train['disease_cases'],
        exogenous=bahia_train[['rainfall', 'mean_temperature']],
        seasonal=True, m=12,
        suppress_warnings=True,
        stepwise=True,
        error_action='ignore',
    )
    order = auto_fit.order
    seasonal_order = auto_fit.seasonal_order
    print(f"Auto-selected order: {order}, seasonal_order: {seasonal_order}")
except ImportError:
    print("pmdarima not installed – using default ARIMA(1,1,1)(1,1,1,12)")
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

# Fit statsmodels SARIMAX (used for forecasting, intervals, etc.)
model = SARIMAX(
    bahia_train['disease_cases'],
    exog=bahia_train[['rainfall', 'mean_temperature']],
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
fit_arimax = model.fit(disp=False)
print(fit_arimax.summary())

# ---- Forecast + distribution visualisation ----

# Forecast on the future period
fc = fit_arimax.get_forecast(
    steps=len(bahia_future),
    exog=bahia_future[['rainfall', 'mean_temperature']],
)
fc_mean = fc.predicted_mean
fc_ci_95 = fc.conf_int(alpha=0.05)
fc_ci_80 = fc.conf_int(alpha=0.20)

# Fitted values for the training period
fitted_vals = fit_arimax.fittedvalues

# Plot: distribution as shaded intervals
fig, ax = plt.subplots(figsize=(14, 6))

# Show last year of training data
last_year_mask = bahia_train['time_period'] >= '2015-01-01'
ax.plot(bahia_train.loc[last_year_mask, 'time_period'],
        fitted_vals.values[last_year_mask],
        color='#E69F00', marker='o', markersize=4, label='Fitted')

# Forecast distribution: 95% and 80% PI shading
fc_dates = bahia_future['time_period'].values
ax.fill_between(fc_dates,
                np.maximum(0, fc_ci_95.iloc[:, 0]),
                fc_ci_95.iloc[:, 1],
                alpha=0.15, color='#0072B2', label='95% PI')
ax.fill_between(fc_dates,
                np.maximum(0, fc_ci_80.iloc[:, 0]),
                fc_ci_80.iloc[:, 1],
                alpha=0.30, color='#0072B2', label='80% PI')

# Point forecast
ax.plot(fc_dates, np.maximum(0, fc_mean), color='black', label='Point Forecast')

# Actual future values
ax.plot(bahia_future['time_period'], bahia_future['disease_cases'],
        color='#0072B2', marker='o', markersize=4, label='Actual')

ax.set_xlabel('Month')
ax.set_ylabel('Disease Cases')
ax.legend()
ax.set_title('ARIMAX Forecast with Prediction Intervals – Bahia')
plt.tight_layout()
plt.show()

# =========================
# YOUR TURN (Exercise 2)
# =========================
# 1) Fit ARIMAX for a different location.
#    Hint: replace location == "Bahia" with your chosen location.
# 2) Re-run the forecast distribution plot.
# 3) Does truncation at 0 change how you interpret uncertainty?


############################
# 4) Prediction intervals (model-based)
############################

# Extract 80% and 95% PIs
pi_95 = fc.conf_int(alpha=0.05)
pi_80 = fc.conf_int(alpha=0.20)

pi_tbl = pd.DataFrame({
    'time_period': bahia_future['time_period'].values,
    'point_forecast': fc_mean.values,
    'lower_80': pi_80.iloc[:, 0].values,
    'upper_80': pi_80.iloc[:, 1].values,
    'lower_95': pi_95.iloc[:, 0].values,
    'upper_95': pi_95.iloc[:, 1].values,
})
print(pi_tbl.head(6))

# Plot
fig, ax = plt.subplots(figsize=(14, 6))

# History (last year of training)
bahia_history = bahia_train[bahia_train['time_period'] >= '2015-01-01']
ax.plot(bahia_history['time_period'], bahia_history['disease_cases'],
        color='black', label='History')

# PIs
ax.fill_between(pi_tbl['time_period'],
                pi_tbl['lower_95'], pi_tbl['upper_95'],
                alpha=0.15, color='#0072B2', label='95% PI')
ax.fill_between(pi_tbl['time_period'],
                pi_tbl['lower_80'], pi_tbl['upper_80'],
                alpha=0.30, color='#0072B2', label='80% PI')

# Point forecast + actuals
ax.plot(pi_tbl['time_period'], pi_tbl['point_forecast'],
        color='#0072B2', label='Point Forecast')
ax.plot(bahia_future['time_period'], bahia_future['disease_cases'],
        color='black', linestyle='--', label='Actual')

ax.set_xlabel('Month')
ax.set_ylabel('Disease Cases')
ax.legend()
ax.set_title('Model-Based Prediction Intervals – Bahia')
plt.tight_layout()
plt.show()

# =========================
# YOUR TURN (Exercise 3)
# =========================
# 1) Change PI level from 95% to 90% and 99%.
# 2) Plot them.
# 3) Write one sentence: what happens as the level increases?


############################################################
# 5) BOOTSTRAP UNCERTAINTY (sample paths + bootstrap PI)
############################################################

# statsmodels doesn't have built-in bootstrap forecast,
# so we implement it manually:
#   1. Get residuals from the fitted model
#   2. Resample residuals with replacement
#   3. Simulate future paths by adding resampled residuals to the point forecast

residuals = fit_arimax.resid.dropna().values
n_paths = 5  # increase to 100+ for smoother plots
h = len(bahia_future)

# Generate bootstrap sample paths
sim_paths = np.zeros((n_paths, h))
for i in range(n_paths):
    boot_resid = np.random.choice(residuals, size=h, replace=True)
    sim_paths[i, :] = fc_mean.values + boot_resid

# Plot: actual history + simulated futures
fig, ax = plt.subplots(figsize=(14, 6))

bahia_hist = brazil_dengue[
    (brazil_dengue['location'] == 'Bahia') &
    (brazil_dengue['time_period'] >= '2015-01-01')
].sort_values('time_period')

ax.plot(bahia_hist['time_period'], bahia_hist['disease_cases'], color='black', label='History / Actual')

for i in range(n_paths):
    ax.plot(bahia_future['time_period'], sim_paths[i, :],
            alpha=0.6, linewidth=0.8, label=f'Sim {i+1}' if i < 5 else None)

ax.set_xlabel('Month')
ax.set_ylabel('Disease Cases')
ax.legend(loc='upper left')
ax.set_title('Bootstrap Sample Paths – Bahia')
plt.tight_layout()
plt.show()

# ---- Bootstrap-based prediction intervals ----

n_boot = 1000  # more = more stable empirical intervals
boot_paths = np.zeros((n_boot, h))
for i in range(n_boot):
    boot_resid = np.random.choice(residuals, size=h, replace=True)
    boot_paths[i, :] = fc_mean.values + boot_resid

boot_lower_80 = np.percentile(boot_paths, 10, axis=0)
boot_upper_80 = np.percentile(boot_paths, 90, axis=0)
boot_lower_95 = np.percentile(boot_paths, 2.5, axis=0)
boot_upper_95 = np.percentile(boot_paths, 97.5, axis=0)

# Plot
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(bahia_hist['time_period'], bahia_hist['disease_cases'], color='black', label='History / Actual')

ax.fill_between(bahia_future['time_period'],
                boot_lower_95, boot_upper_95,
                alpha=0.15, color='#D55E00', label='95% Bootstrap PI')
ax.fill_between(bahia_future['time_period'],
                boot_lower_80, boot_upper_80,
                alpha=0.30, color='#D55E00', label='80% Bootstrap PI')

ax.plot(bahia_future['time_period'], fc_mean.values, color='#D55E00', label='Point Forecast')

ax.set_xlabel('Month')
ax.set_ylabel('Disease Cases')
ax.legend()
ax.set_title('Bootstrap Prediction Intervals – Bahia')
plt.tight_layout()
plt.show()

# =========================
# YOUR TURN (Exercise 4)
# =========================
# 1) Change n_boot to 50, 200, 1000 and compare the PI stability.
# 2) Try another location.
# 3) Does bootstrap give wider or narrower intervals than model-based?


############################################################
# 6) CONFORMAL PREDICTION
############################################################

# We will:
#  - Pick one location (Bahia)
#  - Create train / calibration / test splits by dates
#  - Rolling CV: refit ARIMA on expanding window, collect 1-step-ahead errors
#  - Build conformal PI: yhat +/- q_alpha where q_alpha is a quantile of |errors|

bahia = brazil_dengue[brazil_dengue['location'] == 'Bahia'].sort_values('time_period').reset_index(drop=True)

y_all = bahia['disease_cases'].values
xreg_all = bahia[['rainfall', 'mean_temperature']].values
dates_all = bahia['time_period'].values

# Define split dates
train_end = pd.Timestamp('2014-12-31')
calib_start = pd.Timestamp('2015-01-01')
calib_end = pd.Timestamp('2015-12-31')
test_start = pd.Timestamp('2016-01-01')
test_end = pd.Timestamp('2016-12-31')

i_train_end = np.max(np.where(dates_all <= train_end))
i_calib_start = np.min(np.where(dates_all >= calib_start))
i_calib_end = np.max(np.where(dates_all <= calib_end))
i_test_start = np.min(np.where(dates_all >= test_start))
i_test_end = np.max(np.where(dates_all <= test_end))

print(f"Train: indices 0–{i_train_end}  ({len(y_all[:i_train_end+1])} obs)")
print(f"Calib: indices {i_calib_start}–{i_calib_end}  ({i_calib_end - i_calib_start + 1} obs)")
print(f"Test:  indices {i_test_start}–{i_test_end}  ({i_test_end - i_test_start + 1} obs)")

# ---- Calibration residuals via rolling CV ----
# For each calibration month, fit ARIMA on all data up to that point,
# then produce a 1-step-ahead forecast. Collect absolute errors.

calib_errors = []

for t in range(i_calib_start, i_calib_end + 1):
    # Training window: all data up to index t-1
    y_win = y_all[:t]
    x_win = xreg_all[:t]
    x_next = xreg_all[t:t+1]

    try:
        m = SARIMAX(
            y_win, exog=x_win,
            order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        fc_1 = m.get_forecast(steps=1, exog=x_next)
        yhat_1 = fc_1.predicted_mean.iloc[0]
    except Exception:
        # Fallback: use a simpler model if convergence fails
        m = SARIMAX(
            y_win, exog=x_win,
            order=(1, 1, 1), seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        fc_1 = m.get_forecast(steps=1, exog=x_next)
        yhat_1 = fc_1.predicted_mean.iloc[0]

    actual_1 = y_all[t]
    calib_errors.append(abs(actual_1 - yhat_1))

scores = np.array(calib_errors)
print(f"Collected {len(scores)} conformity scores (absolute errors)")
print(f"Score summary: min={scores.min():.1f}, median={np.median(scores):.1f}, max={scores.max():.1f}")

# Choose alpha = 0.05 for a 95% conformal interval
q_95 = np.quantile(scores, 0.95)
print(f"\nq_95 = {q_95:.1f}  (conformal half-width)")

# ---- Fit final model on train+calib, forecast test ----

y_train_calib = y_all[:i_calib_end + 1]
x_train_calib = xreg_all[:i_calib_end + 1]
x_test = xreg_all[i_test_start:i_test_end + 1]

final_model = SARIMAX(
    y_train_calib, exog=x_train_calib,
    order=order, seasonal_order=seasonal_order,
    enforce_stationarity=False, enforce_invertibility=False,
).fit(disp=False)

h_test = i_test_end - i_test_start + 1
fc_test = final_model.get_forecast(steps=h_test, exog=x_test)
yhat_test = fc_test.predicted_mean.values

test_dates = dates_all[i_test_start:i_test_end + 1]
actual_test = y_all[i_test_start:i_test_end + 1]

# Conformal PI
conf_lower = np.maximum(0, yhat_test - q_95)
conf_upper = yhat_test + q_95

# ---- Plot ----
fig, ax = plt.subplots(figsize=(14, 6))

# History (2014 Jan – calibration end)
hist_mask = (dates_all >= np.datetime64('2014-01-01')) & (dates_all <= np.datetime64(calib_end))
ax.plot(dates_all[hist_mask], y_all[hist_mask], color='black', linewidth=1.2, label='Actual')

# Conformal PI
ax.fill_between(test_dates, conf_lower, conf_upper,
                alpha=0.2, color='#0072B2', label='95% Conformal PI')

# Point forecast
ax.plot(test_dates, yhat_test, color='#0072B2', linewidth=1, label='Point Forecast')

# Actual test
ax.plot(test_dates, actual_test, color='black', linewidth=1.2)

ax.set_xlabel('Month')
ax.set_ylabel('Disease Cases')
ax.legend()
ax.set_title('Conformal Prediction Intervals – Bahia')
plt.tight_layout()
plt.show()

# =========================
# YOUR TURN (Exercise 5)
# =========================
# 1) Change the split:
#    - Make calibration 24 months instead of 12 months (e.g., 2015-2016).
#    - Move the test year forward.
# 2) Recompute q_95 and replot.
# 3) Does the interval get tighter or wider? Why?
