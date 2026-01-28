import pandas as pd
import numpy as np
from cuml.svm import SVR
from cuml.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from metrics import evaluate_performance, print_performance_metrics

if __name__ == "__main__":
	print("Loading data for Ensemble Learning...")
	try:
		df = pd.read_csv("cleaned_data_compressed.csv")
	except FileNotFoundError:
		print("Error: 'cleaned_data_compressed.csv' not found.")
		exit(1)

	# Load ground truth
	gt = pd.read_csv("data/train_hh_gt.csv")
	df = df.merge(gt[['hhid', 'cons_ppp17']], on='hhid')

	# Prepare data
	weight = df["weight"].to_numpy()
	X = df.drop(columns=['hhid', 'cons_ppp17', 'weight', 'survey_id'], errors='ignore').to_numpy().astype(np.float32)
	y = df['cons_ppp17'].to_numpy().astype(np.float32)

	# Split
	X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, weight, test_size=0.2, random_state=42)
	w_train = w_train / np.mean(w_train)

	# Scaling (Important for SVR, helpful for others)
	scaler_x = StandardScaler()
	X_train_scaled = scaler_x.fit_transform(X_train)
	X_test_scaled = scaler_x.transform(X_test)

	# Target Log Transform & Scaling (Critical for SVR)
	y_train_log = np.log(y_train)
	scaler_y = StandardScaler()
	y_train_log_scaled = scaler_y.fit_transform(y_train_log.reshape(-1, 1)).flatten()

	print("-" * 30)
	
	# 1. SVR (Using robust parameters)
	print("Training SVR (RBF)...")
	svr = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')
	svr.fit(X_train_scaled, y_train_log_scaled, sample_weight=w_train)
	
	pred_svr_scaled = svr.predict(X_test_scaled)
	pred_svr_log = scaler_y.inverse_transform(pred_svr_scaled.reshape(-1, 1)).flatten()
	pred_svr = np.exp(pred_svr_log)
	rmse_svr = np.sqrt(mean_squared_error(y_test, pred_svr, sample_weight=w_test))
	print(f"SVR RMSE: {rmse_svr:.4f}")

	# 2. XGBoost
	print("Training XGBoost...")
	xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=6, n_jobs=-1)
	xg_reg.fit(X_train_scaled, y_train_log, sample_weight=w_train)
	
	pred_xgb_log = xg_reg.predict(X_test_scaled)
	pred_xgb = np.exp(pred_xgb_log)
	rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb, sample_weight=w_test))
	print(f"XGBoost RMSE: {rmse_xgb:.4f}")

	# 3. Random Forest
	print("Training Random Forest...")
	# Note: cuML RF currently doesn't support sample_weight in fit, but is very fast
	rf = RandomForestRegressor(n_estimators=200, max_depth=16, random_state=42)
	rf.fit(X_train_scaled, y_train_log) 
	
	pred_rf_log = rf.predict(X_test_scaled)
	pred_rf = np.exp(pred_rf_log)
	rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf, sample_weight=w_test))
	print(f"Random Forest RMSE: {rmse_rf:.4f}")

	# 4. Ensemble (Average)
	print("-" * 30)
	print("Calculating Ensemble...")
	# Simple averaging often works best
	pred_ensemble = (pred_svr + pred_xgb + pred_rf) / 3.0

	metrics = evaluate_performance(y_test, pred_ensemble, w_test)
	print_performance_metrics(metrics, model_name="Ensemble (SVR+XGB+RF)")