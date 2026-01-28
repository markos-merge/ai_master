from cuml.svm import SVR
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from metrics import evaluate_performance, print_performance_metrics
from generate_submission import prepareSubmission

if __name__ == "__main__":
	df = pd.read_csv("cleaned_data_compressed.csv")
	gt = pd.read_csv("data/train_hh_gt.csv")
	df = df.merge(gt[['hhid', 'cons_ppp17']], on='hhid')
	
	weight = df["weight"].to_numpy()
	X = df.drop(columns=['hhid', 'cons_ppp17', 'weight', 'survey_id'], errors='ignore').to_numpy()
	y = df['cons_ppp17'].to_numpy()
	
	X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, weight, test_size=0.2, random_state=42)
	
	w_train = w_train / np.mean(w_train)

	y_train_log = np.log(y_train)
	scaler_y = StandardScaler()
	y_train_scaled = scaler_y.fit_transform(y_train_log.reshape(-1, 1)).flatten()
	
	param_candidates = []

	for C in [10, 100]:
		for epsilon in [0.01, 0.1, 0.2]:
			for gamma in ['scale', 0.1]:
				param_candidates.append({'kernel': 'rbf', 'C': C, 'epsilon': epsilon, 'gamma': gamma})

	# for C in [10, 100]:
	# 	for epsilon in [0.01, 0.1]:
	# 		for degree in [2, 3]:
	# 			param_candidates.append({'kernel': 'poly', 'C': C, 'epsilon': epsilon, 'gamma': 'scale', 'degree': degree})

	best_score = float('inf')
	best_params = {}
	kf = KFold(n_splits=5, shuffle=True, random_state=42)
	splits = list(kf.split(X_train))

	print(f"Starting Grid Search with {len(param_candidates)} candidates...")

	for params in param_candidates:
		mse_scores = []
		for train_idx, val_idx in splits:
			X_tr, X_val = X_train[train_idx], X_train[val_idx]
			y_tr, y_val = y_train_scaled[train_idx], y_train_scaled[val_idx]
			w_tr = w_train[train_idx]
			w_val = w_train[val_idx]
			
			model = SVR(**params)
			model.fit(X_tr, y_tr, sample_weight=w_tr)
			
			y_pred_val = model.predict(X_val)
			mse_scores.append(mean_squared_error(y_val, y_pred_val, sample_weight=w_val))
		
		avg_mse = np.mean(mse_scores)
		p_str = ", ".join([f"{k}={v}" for k, v in params.items()])
		print(f"Params: [{p_str}], CV MSE={avg_mse:.4f}")
		
		if avg_mse < best_score:
			best_score = avg_mse
			best_params = params
	
	print(f"Best parameters found: {best_params}")
	print(f"Best CV score (MSE): {best_score:.4f}")

	best_svr = SVR(**best_params)
	best_svr.fit(X_train, y_train_scaled, sample_weight=w_train)
	
	y_pred_scaled = best_svr.predict(X_test)
	y_pred_log = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

	y_pred = np.exp(y_pred_log)

	metrics = evaluate_performance(y_test, y_pred, w_test)
	print_performance_metrics(metrics, model_name="SVR (RBF)")


	test_file = "cleaned_data_test_compressed.csv"

	df_test = pd.read_csv(test_file)
	print(f"Loaded test data: {df_test.shape}")

	train_cols = df.drop(columns=['hhid', 'cons_ppp17', 'weight', 'survey_id'], errors='ignore').columns
	X_test_sub = df_test[train_cols].to_numpy()
	

	scaler_y_full = StandardScaler()
	y_log_full = np.log(y)
	y_scaled_full = scaler_y_full.fit_transform(y_log_full.reshape(-1, 1)).flatten()

	weight_normalized = weight / np.mean(weight)
	print(f"Retraining SVR on full dataset with params: {best_params}")
	final_model = SVR(**best_params)
	final_model.fit(X, y_scaled_full, sample_weight=weight_normalized)
	print( best_params )

	y_pred_sub_scaled = final_model.predict(X_test_sub)
	y_pred_sub_log = scaler_y_full.inverse_transform(y_pred_sub_scaled.reshape(-1, 1)).flatten()
	y_pred_sub = np.exp(y_pred_sub_log)

	prepareSubmission(df_test, y_pred_sub, "svm_submission")

