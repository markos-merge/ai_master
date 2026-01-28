import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import os
import zipfile
from generate_submission import prepareSubmission
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import itertools
import cupy as cp
from tqdm import tqdm


if __name__ == "__main__":
	train_path = "cleaned_data_compressed.csv"
	test_path = "cleaned_data_test_compressed.csv"
	gt_path = "data/train_hh_gt.csv"

	df_train = pd.read_csv(train_path)
	df_test = pd.read_csv(test_path)
	gt = pd.read_csv(gt_path)
	
	df_train = df_train.merge(gt[['hhid', 'cons_ppp17']], on='hhid')

	weight_train = df_train["weight"].to_numpy()
	weight_train = weight_train / np.mean(weight_train)

	# Drop metadata to isolate features
	cols_to_drop = ['hhid', 'cons_ppp17', 'weight', 'survey_id']
	X_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns], errors='ignore')
	y_train = df_train['cons_ppp17'].to_numpy()
	
	test_ids = df_test['hhid']
	test_survey_ids = df_test['survey_id']

	X_test = df_test[X_train.columns]

	y_train_log = np.log(y_train)
	scaler_y = StandardScaler()
	y_train_scaled = scaler_y.fit_transform(y_train_log.reshape(-1, 1)).flatten()

	param_grid = {
		# 'n_estimators': [500, 1000, 2000],
		'n_estimators': [2000, 3000, 4000],
		'learning_rate': [0.01, 0.05, 0.1],
		'max_depth': [4, 6, 8, 10],
		'subsample': [0.7, 0.8, 0.9],
		'colsample_bytree': [0.7, 0.8, 0.9],
		'tree_method': ['hist'],
		'device': ['cuda'],
	}

	keys, values = zip(*param_grid.items())
	param_candidates = [dict(zip(keys, v)) for v in itertools.product(*values)]

	best_score = float('inf')
	best_params = {}
	X_train_np = X_train.to_numpy()
	kf = KFold(n_splits=5, shuffle=True, random_state=42)

	for i, params in enumerate(tqdm(param_candidates)):
		mse_scores = []
		for train_idx, val_idx in kf.split(X_train_np):
			X_tr, X_val = X_train_np[train_idx], X_train_np[val_idx]
			y_tr, y_val = y_train_scaled[train_idx], y_train_scaled[val_idx]
			w_tr, w_val = weight_train[train_idx], weight_train[val_idx]
			
			X_tr = cp.asarray(X_tr)
			y_tr = cp.asarray(y_tr)
			w_tr = cp.asarray(w_tr)
			X_val_gpu = cp.asarray(X_val)

			model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, **params)
			model.fit(X_tr, y_tr, sample_weight=w_tr)
			y_pred_val = model.predict(X_val_gpu)
			y_pred_val = cp.asnumpy(y_pred_val)
			mse_scores.append(mean_squared_error(y_val, y_pred_val, sample_weight=w_val))
		
		avg_mse = np.mean(mse_scores)
		if avg_mse < best_score:
			best_score = avg_mse
			best_params = params
			print(f"New Best: {best_score:.4f} with {best_params}")

	X_train_gpu = cp.asarray(X_train_np)
	y_train_gpu = cp.asarray(y_train_scaled)
	w_train_gpu = cp.asarray(weight_train)
	X_test_gpu = cp.asarray(X_test.to_numpy())

	model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, **best_params)
	model.fit(X_train_gpu, y_train_gpu, sample_weight=w_train_gpu)

	y_pred_scaled = model.predict(X_test_gpu)
	y_pred_scaled = cp.asnumpy(y_pred_scaled)

	y_pred_log = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
	y_pred = np.exp(y_pred_log)

	prepareSubmission( df_test, y_pred, "submission")
