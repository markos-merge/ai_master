import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def fillMissingData( df ):
	cat_cols = df.select_dtypes(include=['object']).columns
	num_cols = df.select_dtypes(exclude=['object']).columns

	if len(cat_cols) > 0:
		cat_imputer = SimpleImputer(strategy='most_frequent')
		df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

	if len(num_cols) > 0:
		num_imputer = SimpleImputer(strategy='median')
		df[num_cols] = num_imputer.fit_transform(df[num_cols])

def scaleFeatures( df ):
	scaler = StandardScaler()
	num_cols = df.select_dtypes(exclude=['object']).columns
	cols_to_scale = [col for col in num_cols if df[col].nunique() != 2]
	if len(cols_to_scale) > 0:
		df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

def objectToOneHot( df ):
	cat_cols = df.select_dtypes(include=['object']).columns

	cols_to_encode = [col for col in cat_cols if df[col].nunique() > 3]
	cols_to_factorize = [col for col in cat_cols if df[col].nunique() <= 3]

	if len(cols_to_encode) > 0:
		encoder = OneHotEncoder(sparse_output = False )
		encoded_features = encoder.fit_transform(df[cols_to_encode])
		encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cols_to_encode))
		df = pd.concat([df, encoded_df], axis=1)
		df = df.drop(columns=cols_to_encode)

	for col in cols_to_factorize:
		df[col] = pd.factorize(df[col])[0]

	return df

def cleanData( df ):
	hhid = df['hhid']
	weight = df['weight'] if 'weight' in df.columns else None
	survey = df['survey_id'] if 'survey_id' in df.columns else None
	
	cols_to_drop = [c for c in ['hhid', 'com', 'survey_id', 'weight'] if c in df.columns]
	df = df.drop( columns = cols_to_drop )

	fillMissingData( df )
	df['num_children'] = df['num_children5'] + df['num_children10'] + df['num_children18']
	df['workers'] = df['sworkershh']
	df = df.drop( columns = ['num_children5', 'num_children10', 'num_children18', 'sworkershh', 'sfworkershh'] )
	scaleFeatures( df )
	df = objectToOneHot( df )
	
	df['hhid'] = hhid
	if weight is not None:
		df['weight'] = weight
	if survey is not None:
		df['survey_id'] = survey

	return df

if __name__ == "__main__":
	train_file = "data/train_hh_features.csv"
	test_file = "data/test_hh_features.csv"
	
	df_train = pd.read_csv(train_file)
	
	try:
		df_test = pd.read_csv(test_file)
		print("Found test data. Processing train and test together...")
		df_train['is_train'] = 1
		df_test['is_train'] = 0
		# Concat to ensure consistent encoding
		df_all = pd.concat([df_train, df_test], ignore_index=True)
		
		df_all_clean = cleanData(df_all)
		
		df_train_clean = df_all_clean[df_all_clean['is_train'] == 1].drop(columns=['is_train'])
		df_test_clean = df_all_clean[df_all_clean['is_train'] == 0].drop(columns=['is_train'])
		
		df_train_clean.to_csv("cleaned_data.csv", index=False)
		df_test_clean.to_csv("cleaned_data_test.csv", index=False)
		print("Saved cleaned_data.csv and cleaned_data_test.csv")
		
	except FileNotFoundError:
		print("Test file not found. Processing only train data...")
		df = cleanData(df_train)
		df.to_csv("cleaned_data.csv", index=False)