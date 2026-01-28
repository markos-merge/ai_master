import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from metrics import evaluate_performance, print_performance_metrics
from generate_submission import prepareSubmission

np.random.seed(42)
tf.random.set_seed(42)

def build_model(input_dim, hidden_layers=[128, 64], dropout=0.0, learning_rate=0.001):
	model = Sequential()
	model.add(Input(shape=(input_dim,)))
	for units in hidden_layers:
		model.add(Dense(units, activation='relu'))
		if dropout > 0:
			model.add(Dropout(dropout))
	model.add(Dense(1, activation='linear'))
	
	optimizer = Adam(learning_rate=learning_rate)
	model.compile(optimizer=optimizer, loss='mse')
	return model

if __name__ == "__main__":
	print("Loading data for MLP training (TensorFlow)...")

	gpus = tf.config.list_physical_devices('GPU')
	if gpus:
		print(f"GPU detected: {gpus}")
	else:
		print("No GPU detected. Running on CPU.")

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

	print("Initializing MLP and Manual Grid Search...")

	param_candidates = [
		{'hidden_layers': [128, 64], 'dropout': 0.1, 'learning_rate': 0.001, 'batch_size': 32, 'epochs': 100},
		{'hidden_layers': [256, 128, 64], 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 64, 'epochs': 100},
		{'hidden_layers': [128, 64, 32], 'dropout': 0.1, 'learning_rate': 0.0005, 'batch_size': 32, 'epochs': 100},
		{'hidden_layers': [512, 256, 128], 'dropout': 0.3, 'learning_rate': 0.0005, 'batch_size': 64, 'epochs': 100 }
	]

	best_score = float('inf')
	best_params = {}
	kf = KFold(n_splits=3, shuffle=True, random_state=42)
	
	input_dim = X_train.shape[1]

	print(f"Starting Grid Search with {len(param_candidates)} candidates...")

	for params in param_candidates:
		mse_scores = []
		for train_idx, val_idx in kf.split(X_train):
			X_tr, X_val = X_train[train_idx], X_train[val_idx]
			y_tr, y_val = y_train_scaled[train_idx], y_train_scaled[val_idx]
			w_tr = w_train[train_idx]
			
			model = build_model(input_dim, params['hidden_layers'], params['dropout'], params['learning_rate'])
			
			early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
			
			model.fit(
				X_tr, y_tr, 
				sample_weight=w_tr,
				validation_data=(X_val, y_val),
				epochs=params['epochs'],
				batch_size=params['batch_size'],
				callbacks=[early_stop],
				verbose=0
			)
			
			y_pred_val = model.predict(X_val, verbose=0).flatten()
			mse_scores.append(mean_squared_error(y_val, y_pred_val))
		
		avg_mse = np.mean(mse_scores)
		print(f"Params: {params}, CV MSE={avg_mse:.4f}")
		
		if avg_mse < best_score:
			best_score = avg_mse
			best_params = params

	best_model = build_model(input_dim, best_params['hidden_layers'], best_params['dropout'], best_params['learning_rate'])

	X_tr_final, X_val_final, y_tr_final, y_val_final, w_tr_final, w_val_final = train_test_split(X_train, y_train_scaled, w_train, test_size=0.1, random_state=42)

	early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

	history = best_model.fit(
		X_tr_final, y_tr_final,
		sample_weight=w_tr_final,
		validation_data=(X_val_final, y_val_final),
		epochs=best_params['epochs'] + 50,
		batch_size=best_params['batch_size'],
		callbacks=[early_stop],
		verbose=1
	)
	
	y_pred_scaled = best_model.predict(X_test).flatten()
	y_pred_log = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
	y_pred = np.exp(y_pred_log)
	
	metrics = evaluate_performance(y_test, y_pred, w_test)
	print_performance_metrics(metrics, model_name="MLP Regressor (TF)")
	
	test_file = "cleaned_data_test_compressed.csv"
	df_test = pd.read_csv(test_file)
	train_cols = df.drop(columns=['hhid', 'cons_ppp17', 'weight', 'survey_id'], errors='ignore').columns
	X_test_sub = df_test[train_cols].to_numpy()
	
	test_ids = df_test['hhid']
	test_survey_ids = df_test['survey_id'] if 'survey_id' in df_test.columns else np.zeros(len(df_test), dtype=int)
	
	
	scaler_y_full = StandardScaler()
	y_log_full = np.log(y)
	y_scaled_full = scaler_y_full.fit_transform(y_log_full.reshape(-1, 1)).flatten()
	
	X_tr_all, X_val_all, y_tr_all, y_val_all, w_tr_all, w_val_all = train_test_split(
		X, y_scaled_full, weight/np.mean(weight), test_size=0.1, random_state=42
	)

	final_model = build_model(input_dim, best_params['hidden_layers'], best_params['dropout'], best_params['learning_rate'])
	
	final_model.fit(
		X_tr_all, y_tr_all,
		sample_weight=w_tr_all,
		validation_data=(X_val_all, y_val_all),
		epochs=best_params['epochs'] + 50,
		batch_size=best_params['batch_size'],
		callbacks=[early_stop],
		verbose=1
	)

	print( best_params )
	
	y_pred_sub_scaled = final_model.predict(X_test_sub).flatten()
	y_pred_sub_log = scaler_y_full.inverse_transform(y_pred_sub_scaled.reshape(-1, 1)).flatten()
	y_pred_sub = np.exp(y_pred_sub_log)
	
	prepareSubmission(df_test, y_pred_sub, "mlp_submission")