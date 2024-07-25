import joblib
import inflection
import pandas as pd
import json  # Import the json module

class Fraud:
    
    def __init__(self):
        try:
            self.onehotencoder = joblib.load('./onehotencoder_cycle1.joblib')
        except FileNotFoundError:
            print("File not found: './onehotencoder_cycle1.joblib'. Please check the file path and try again.")
            self.onehotencoder = None  # Handle the absence of the encoder gracefully
        
        try:
            self.minmaxscaler = joblib.load('./minmaxscaler_cycle1.joblib')
        except FileNotFoundError:
            print("File not found: './minmaxscaler_cycle1.joblib'. Please check the file path and try again.")
            self.minmaxscaler = None  # Handle the absence of the scaler gracefully
        
    def data_cleaning(self, df1):
        cols_old = df1.columns.tolist()
        
        snakecase = lambda i: inflection.underscore(i)
        cols_new = list(map(snakecase, cols_old))
        
        df1.columns = cols_new
        
        return df1
    
    def feature_engineering(self, df2):
        df2['step_days'] = df2['step'].apply(lambda i: i/24)
        df2['step_weeks'] = df2['step'].apply(lambda i: i/(24*7))
        df2['diff_new_old_balance'] = df2['newbalance_orig'] - df2['oldbalance_org']
        df2['diff_new_old_destiny'] = df2['newbalance_dest'] - df2['oldbalance_dest']
        df2['name_orig'] = df2['name_orig'].apply(lambda i: i[0])
        df2['name_dest'] = df2['name_dest'].apply(lambda i: i[0])
        
        return df2.drop(columns=['name_orig', 'name_dest', 'step_weeks', 'step_days'], axis=1)
    
    def data_preparation(self, df3):
        if self.onehotencoder:
            df3 = self.onehotencoder.transform(df3)
            df3 = pd.DataFrame(df3, columns=self.onehotencoder.get_feature_names_out())
        
        num_columns = ['amount', 'oldbalance_org', 'newbalance_orig', 'oldbalance_dest', 
                       'newbalance_dest', 'diff_new_old_balance', 'diff_new_old_destiny']
        if hasattr(self, 'minmaxscaler') and self.minmaxscaler:
            df3[num_columns] = self.minmaxscaler.transform(df3[num_columns])
            
            # Manually clip the data
            df3[num_columns] = df3[num_columns].clip(0, 1)
        
        final_columns_selected = ['step', 'oldbalance_org', 'newbalance_orig', 'newbalance_dest', 
                                  'diff_new_old_balance', 'diff_new_old_destiny', 'type_TRANSFER']
        
        # Check if all final columns are present in the DataFrame
        missing_columns = [col for col in final_columns_selected if col not in df3.columns]
        if missing_columns:
            print(f"Missing columns after transformation: {missing_columns}")
            # Handle missing columns (e.g., add them with default values)
            for col in missing_columns:
                df3[col] = 0  # or some other default value
                
        return df3[final_columns_selected]
    
    def get_prediction(self, model, original_data, test_data):
        pred = model.predict(test_data)
        original_data['prediction'] = pred
        
        # Convert the DataFrame to a JSON object
        result = original_data.to_dict(orient="records")
        
        return result


