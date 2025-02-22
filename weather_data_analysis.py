# Weather Data Analysis Script

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Data Exploration

df = pd.read_csv('/content/drive/MyDrive/basic/weather.csv')

print(df.head())
print(df.info())
print(df.describe())
print(df.columns)
# Step 2: Data Visualization
sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
plt.show()

# Step 3: Feature Engineering
if 'MinTemp' in df.columns and 'MaxTemp' in df.columns:
    df['TempDiff'] = df['MaxTemp'] - df['MinTemp']
# Step 4: Data Analysis
if 'WindGustDir' in df.columns and 'MaxTemp' in df.columns:
    avg_max_temp = df.groupby('WindGustDir')['MaxTemp'].mean().sort_values(ascending=False)
    print(avg_max_temp)

# Step 5: Data Visualization (Part 2)
avg_max_temp.plot(kind='bar', figsize=(10, 6))
plt.xlabel('WindGust Direction')
plt.ylabel('Average Max Temperature')
plt.title('Average Max Temperature by WindGust Direction')
plt.show()

# Step 6: Additional Visualizations
# Scatter plot of TempDiff vs Rainfall
if 'TempDiff' in df.columns and 'Rainfall' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='TempDiff', y='Rainfall', data=df)
    plt.xlabel('Temperature Difference (MaxTemp - MinTemp)')
    plt.ylabel('Rainfall')
    plt.title('Temperature Difference vs Rainfall')
    plt.show()
# Step 7: Advanced Analysis (Rainfall Prediction)
if {'MinTemp', 'MaxTemp', 'Rainfall'}.issubset(df.columns):
    X = df[['MinTemp', 'MaxTemp']]
    y = df['Rainfall']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error for Rainfall Prediction: {mse}')
    print(f'R-squared Score: {r2}')

    results_df = pd.DataFrame({'Actual Rainfall': y_test, 'Predicted Rainfall': y_pred})
    print(results_df.head())

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.xlabel('Actual Rainfall')
    plt.ylabel('Predicted Rainfall')
    plt.title('Predicted vs Actual Rainfall')
    plt.grid(True)
    plt.show()

# Step 8: Conclusions and Insights
print('\nConclusions and Insights:')
print('- Data exploration and visualization highlight relationships between weather variables.')
print('- Linear regression predicts rainfall using MinTemp and MaxTemp, with performance measured by MSE and R-squared.')
print('- Scatter plots and bar charts provide insights into temperature differences and their impact on rainfall.')
