import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
    
# Creating multiple columns split from columns like category, attributes, venue, etc.
def encode_multilabel_field(df, col):
    # Parse the string representation into actual lists
    df[col] = df[col].apply(
        lambda x: [item.strip().strip("'\"") for item in str(x).strip('[]').split(',')] 
        if pd.notna(x) and x not in ['[]', '', 'nan'] 
        else []
    )
    
    # Use MultiLabelBinarizer to create individual binary columns
    mlb = MultiLabelBinarizer()
    encoded = pd.DataFrame(
        mlb.fit_transform(df[col]),
        columns=[f"{col}_{c}" for c in mlb.classes_],
        index=df.index
    )
    
    print(f"{col}: {len(mlb.classes_)} unique values found")
    
    # Drop original and concatenate encoded
    df = df.drop(col, axis=1)
    df = pd.concat([df, encoded], axis=1)
    
    return df


# Check variance for all binary/dummy columns
def remove_low_variance_features(df, threshold=0.90, exclude_cols=['business_id', 'month', 'demand']):
    low_variance_features = []
    
    # Get all columns except the ones to exclude
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            # Check if it's a binary column (0/1)
            unique_vals = df[col].nunique()
            
            if unique_vals <= 10:  # Binary or low cardinality
                value_counts = df[col].value_counts(normalize=True)
                max_proportion = value_counts.max()
                
                if max_proportion >= threshold:
                    low_variance_features.append({
                        'column': col,
                        'max_proportion': max_proportion,
                        'dominant_value': value_counts.idxmax()
                    })
    
    # Create summary dataframe
    low_var_df = pd.DataFrame(low_variance_features)
    
    if len(low_var_df) > 0:
        print(f"\nFound and dropped {len(low_var_df)} low-variance features:")
        print(low_var_df.sort_values('max_proportion', ascending=False))
        
        # Drop these columns
        cols_to_drop = low_var_df['column'].tolist()
        df = df.drop(columns=cols_to_drop)
    else:
        print(f"No low-variance features found with threshold {threshold}")
    
    return df, low_var_df