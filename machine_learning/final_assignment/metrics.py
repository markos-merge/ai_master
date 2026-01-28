import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def calculate_poverty_rate(y, weights, line):
	"""
	Calculates the weighted poverty rate for a given poverty line.
	"""
	if len(y) == 0:
			return 0.0
	
	# Ensure inputs are numpy arrays
	y = np.array(y)
	weights = np.array(weights)
	
	# Identify poor households
	poor_mask = y < line
	
	# Calculate weighted ratio
	total_weight = np.sum(weights)
	if total_weight == 0:
			return 0.0
			
	poor_weight = np.sum(weights[poor_mask])
	rate = poor_weight / total_weight
	return rate

def evaluate_performance(y_true, y_pred, weights, poverty_lines=[2.15, 3.65, 6.85]):
	"""
	Calculates consumption metrics (RMSE, R2) and Poverty Rate errors.
	"""
	metrics = {}
	
	# 1. Consumption Metrics (Weighted)
	mse = mean_squared_error(y_true, y_pred, sample_weight=weights)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_true, y_pred, sample_weight=weights)
	
	metrics['RMSE'] = rmse
	metrics['R2'] = r2
	
	# 2. Poverty Rate Metrics
	# World Bank Poverty Lines (2017 PPP): $2.15, $3.65, $6.85
	for line in poverty_lines:
			true_rate = calculate_poverty_rate(y_true, weights, line)
			pred_rate = calculate_poverty_rate(y_pred, weights, line)
			
			metrics[f'Poverty_Rate_{line}_True'] = true_rate
			metrics[f'Poverty_Rate_{line}_Pred'] = pred_rate
			metrics[f'Poverty_Rate_{line}_Error'] = pred_rate - true_rate
			metrics[f'Poverty_Rate_{line}_AbsError'] = abs(pred_rate - true_rate)

	return metrics

def print_performance_metrics(metrics, model_name="Model"):
	print(f"\n{'='*10} {model_name} Performance {'='*10}")
	print(f"Weighted RMSE: {metrics['RMSE']:.4f}")
	print(f"Weighted R2:   {metrics['R2']:.4f}")
	print("-" * 40)
	print("Poverty Rate Estimates (Weighted):")
	print(f"{'Line ($)':<10} | {'True':<10} | {'Pred':<10} | {'Error':<10}")
	print("-" * 46)
	
	# Extract lines from keys
	lines = sorted(list(set([float(k.split('_')[2]) for k in metrics.keys() if 'Poverty_Rate' in k])))
	
	for line in lines:
			true_val = metrics[f'Poverty_Rate_{line}_True']
			pred_val = metrics[f'Poverty_Rate_{line}_Pred']
			error_val = metrics[f'Poverty_Rate_{line}_Error']
			print(f"{line:<10.2f} | {true_val:<10.4f} | {pred_val:<10.4f} | {error_val:<10.4f}")
	print("=" * 46 + "\n")