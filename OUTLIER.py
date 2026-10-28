import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("tips.xlsx")

def detect_outliers_total_bill(df, column_name="total_bill", z_threshold=2):
    data_values = df[column_name].dropna().values
    mean = np.mean(data_values)
    stddev = np.std(data_values)
    z_scores = (data_values - mean) / stddev
    z_outliers = np.where(np.abs(z_scores) > z_threshold)[0]
    
    q1 = np.percentile(data_values, 25)
    q3 = np.percentile(data_values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    iqr_outliers = np.where((data_values < lower_bound) | (data_values > upper_bound))[0]
    
    print(f"\n==== {column_name.upper()} ====")
    print(f"Mean: {mean:.2f}")
    print(f"Standard Deviation: {stddev:.2f}")
    print(f"IQR: {iqr:.2f} | Lower Bound: {lower_bound:.2f} | Upper Bound: {upper_bound:.2f}")
    print(f"Outliers using Z-Score: {data_values[z_outliers]}")
    print(f"Outliers using IQR: {data_values[iqr_outliers]}")
    
    plt.figure(figsize=(6, 5))
    sns.boxplot(y=data_values, color="skyblue")
    plt.title(f"Boxplot of {column_name}")
    plt.ylabel(column_name)
    plt.grid(True)
    plt.show()

detect_outliers_total_bill(df, "total_bill")
