import pandas as pd

def check_data_quality(filepath):
    """
    Checks for missing values and duplicates in a DataFrame.
    Returns dictionaries of missing value counts per column and the total count of duplicate rows.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Missing values info
        missing_info = df.isnull().sum().to_dict()
        
        # Duplicate rows count
        duplicate_count = df.duplicated().sum()

        # Get list of columns for front-end forms
        columns = df.columns.tolist()

        return missing_info, duplicate_count, columns
    except Exception as e:
        print(f"Error checking data quality: {e}")
        return None, 0, []

def handle_preprocessing_action(filepath, action, form_data):
    """
    Performs a specified preprocessing action on the DataFrame.
    Returns True on success, False otherwise.
    """
    try:
        df = pd.read_csv(filepath)

        if action == "drop_duplicates":
            df.drop_duplicates(inplace=True)
            df.to_csv(filepath, index=False)
            return True

        elif action == "fill_missing":
            fill_method = form_data.get("fill_method")
            if fill_method == "drop_rows":
                df.dropna(inplace=True)
            else:
                for col in df.columns:
                    if df[col].isnull().any():
                        # Fill based on data type and method
                        if pd.api.types.is_numeric_dtype(df[col]):
                            if fill_method == "mean":
                                df[col] = df[col].fillna(df[col].mean())
                            elif fill_method == "median":
                                df[col] = df[col].fillna(df[col].median())
                            elif fill_method == "mode":
                                df[col] = df[col].fillna(df[col].mode().iloc[0])
                        else:
                            # For non-numeric, only mode is applicable — skip mean/median
                            if fill_method == "mode":
                                df[col] = df[col].fillna(df[col].mode().iloc[0])
                            # mean/median: skip non-numeric columns entirely
            df.to_csv(filepath, index=False)
            return True

        elif action == "drop_columns":
            columns_to_drop = form_data.getlist("drop_columns")
            if columns_to_drop:
                df.drop(columns=columns_to_drop, inplace=True)
                df.to_csv(filepath, index=False)
                return True

        elif action == "rename_column":
            old_name = form_data.get("rename_column")
            new_name = form_data.get("new_name")
            if old_name and new_name and old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
                df.to_csv(filepath, index=False)
                return True
        
        return False
    except Exception as e:
        print(f"Error handling preprocessing action: {e}")
        return False

def get_current_columns(filepath):
    """Fetches current column names from the CSV file."""
    try:
        df = pd.read_csv(filepath)
        return df.columns.tolist()
    except Exception as e:
        print(f"Error fetching columns: {e}")
        return []