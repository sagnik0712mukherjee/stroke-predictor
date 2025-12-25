from sklearn.preprocessing import LabelEncoder

def preprocess_df(df, categorical_cols, fill_bmi=0):
    """
    Preprocess the dataframe by encoding categorical variables and filling missing BMI values.
    """
    df_processed = df.copy()
    le = LabelEncoder()
    
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            
    if 'bmi' in df_processed.columns:
        df_processed['bmi'] = df_processed['bmi'].fillna(fill_bmi)
        
    return df_processed

def get_features_targets(df, drop_cols, target_col):
    """
    Split the dataframe into features and target.
    """
    x = df.drop(columns=drop_cols)
    y = df[target_col]
    return x, y
