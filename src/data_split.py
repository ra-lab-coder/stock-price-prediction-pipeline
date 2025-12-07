import os
import argparse
import pandas as pd

class DataSplit:
    def load_cleaned_data(self, filename):
            """
            Load cleaned data from Task 6
            
            Args:
                filename (str): Name of cleaned CSV file
                
            Returns:
                pandas.DataFrame: Loaded and prepared data
            """
            try:
                filepath = os.path.join(self.data_dir, filename)
                data = pd.read_csv(filepath)
                
                # Convert Date column if it exists
                if 'Date' in data.columns:
                    data['Date'] = pd.to_datetime(data['Date'])
                    data = data.sort_values('Date').reset_index(drop=True)
                return data
            except Exception as e:
                return None
            
    def split_data(self, df, ticker):
            """
            Split data into training, validation, and testing sets
            
            Returns:
                train, test, validation set csv file
            """        
            self.logger.info("Splitting train, val, test sets.")
            
            n = len(df)

            train_df = df.iloc[: int(0.6 * n) ]
            val_df   = df.iloc[int(0.6 * n) : int(0.8 * n)]
            test_df  = df.iloc[int(0.8 * n) : ]
            
            train_df.to_csv(f"train_{ticker}.csv", index=False)
            val_df.to_csv(f"val_{ticker}.csv", index=False)
            test_df.to_csv(f"test_{ticker}.csv", index=False)
            
            return train_df, val_df, test_df
    
    #def main(args):
    #    data = load_cleaned_data(args.data_dir, args.filename)    
    #  return split_data(data)
        
    #if __name__ == '__main__':
    #    parser = argparse.ArgumentParser(description="Data Split")
    #    parser.add_argument("--data_dir", type="str")
    #    parser.add_arguement("--filename", type="str")
    #    args = parser.parse_args()
    #    main(args)
        

        

        
        