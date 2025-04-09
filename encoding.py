import pandas as pd
import numpy as np

def calculate_encoding(df: pd.DataFrame, column: str, binary = False, alpha = 1.0) -> pd.DataFrame:
    """
    Calculate the mean and standard deviation of a specified column,
    and create a new column 'encoding' based on the given conditions.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to calculate the mean and standard deviation on.
        
    Returns:
        pd.DataFrame: The DataFrame with the new 'encoding' column.
    """
    mean_value = df[column].mean()
    std_dev = df[column].std()

    if binary == True:
        df[f'binary_encoding_{column}'] = 0
    else:
        df[f'encoding_{column}_{alpha}'] = 0
    
    for i in range(1, df.shape[0]):
        current_value = df.at[df.index[i], column]
        previous_value = df.at[df.index[i-1], column]

        if binary == True:
            if current_value > previous_value:
                df.at[df.index[i], f'binary_encoding_{column}'] = 1
            elif current_value < previous_value:
                df.at[df.index[i], f'binary_encoding_{column}'] = -1
        else:
            if current_value > mean_value + std_dev * alpha:
                df.at[df.index[i], f'encoding_{column}_{alpha}'] = 1
            elif current_value < mean_value - std_dev * alpha:
                df.at[df.index[i], f'encoding_{column}_{alpha}'] = -1
            
    return df

def calculate_entropy(df: pd.DataFrame, column: str) -> float:
    """
    Calculate the entropy of a specified column in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to calculate the entropy on.

    Returns:
        float: The entropy of the specified column.
    """
    value_counts = df[column].value_counts(normalize=True)

    entropy_value = -np.sum(value_counts * np.log2(value_counts + np.finfo(float).eps))
    if 0 in value_counts:
        print('')

    return entropy_value
