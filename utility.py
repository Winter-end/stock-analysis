import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week

    return df

def correlation(x: pd.Series, y: pd.Series, stock: str) -> None:
    """
    Calculates correlation and visualize it with best fit line on scatter plot.
    """
    correlation_matrix = np.corrcoef(x, y)
    correlation = correlation_matrix[0,1]
    print(f"Współczynniki korelacji Pearsona pomiędzy wartością zamknięcia a wolumenem: {correlation:.2f}")

    coefficients = np.polyfit(x, y, 1)
    print("Współczynniki linii najlepszego dopasowania: ", coefficients)
    best_fit_line = np.poly1d(coefficients)

    plt.scatter(x, y, alpha = 0.7, label = "Punkty danych")
    plt.title(f"Wykres punktowy cena zamknięcie - wolumen ({stock})")
    plt.xlabel("Cena zamknięcia")
    plt.ylabel("Wolumen")
    plt.plot(x, best_fit_line(x), label='Linia najlepszego dopasowania', color = 'r')
    plt.legend()
    plt.show()