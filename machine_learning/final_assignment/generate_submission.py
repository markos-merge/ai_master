import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import os
import zipfile

def prepareSubmission(df_test, y_pred_sub, zip_name):
	test_ids = df_test['hhid']
	test_survey_ids = df_test['survey_id']

	consumption_df = pd.DataFrame({
		'survey_id': test_survey_ids,
		'household_id': test_ids,
		'cons_ppp17': y_pred_sub
	})

	consumption_filename = "predicted_household_consumption.csv"
	consumption_df.to_csv(consumption_filename, index=False)
	print(f"Created {consumption_filename}")

	thresholds = np.array([3.17, 3.94, 4.60, 5.26, 5.88, 6.47, 7.06, 7.7, 8.4, 9.13, 9.87, 10.70, 11.62, 12.69, 14.03, 15.64, 17.76, 20.99, 27.37])
	threshold_cols = [f"pct_hh_below_{t:.2f}" for t in thresholds]

	if "weight" in df_test.columns:
		test_weights = df_test["weight"].to_numpy()
	else:
		test_weights = np.ones(len(df_test))
		
	poverty_data = []
	unique_surveys = consumption_df['survey_id'].unique()
	
	for sid in unique_surveys:
		mask = consumption_df['survey_id'] == sid
		survey_preds = y_pred_sub[mask]
		survey_weights = test_weights[mask]
		
		row = {'survey_id': sid}
		total_weight = np.sum(survey_weights)
		
		for t, col in zip(thresholds, threshold_cols):
			poor_weight = np.sum(survey_weights[survey_preds < t])
			row[col] = poor_weight / total_weight if total_weight > 0 else 0.0
		poverty_data.append(row)
		
	poverty_df = pd.DataFrame(poverty_data)
	poverty_filename = "predicted_poverty_distribution.csv"
	poverty_df.to_csv(poverty_filename, index=False)
	print(f"Created {poverty_filename}")

	zip_filename = zip_name + ".zip"
	with zipfile.ZipFile(zip_filename, 'w') as zf:
		zf.write(consumption_filename)
		zf.write(poverty_filename)
	print("Saved 'submission.zip'.")

	os.remove(consumption_filename)
	os.remove(poverty_filename)