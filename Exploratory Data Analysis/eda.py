import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../Datacleaning/pakwheels_preprocessed_data.csv')


print("Shape:",df.shape,'\n')
print("Dimensions: ", df.ndim,'\n')


print("Describe Data")
print(df.describe(),'\n')
print("Info Data")
print(df.info(),'\n')
print("Finding Unique values for each column")
print(df.nunique(),'\n')



#EDA using graphs and plots
#price vs year
plt.figure(figsize=(12,6))
sns.lineplot(y='price', x='year', data=df)

#histograms
df.hist(bins=50, figsize=(12,6))
plt.show()


# Create box plots for numerical variables
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# Year
sns.boxplot(x=df['year'], ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Box Plot of Year')
# Engine
sns.boxplot(x=df['engine'], ax=axes[0, 1], color='lightgreen')
axes[0, 1].set_title('Box Plot of Engine Capacity (cc)')
# Mileage
sns.boxplot(x=df['mileage'], ax=axes[1, 0], color='salmon')
axes[1, 0].set_title('Box Plot of Mileage')
# Price
sns.boxplot(x=df['price'], ax=axes[1, 1], color='gold')
axes[1, 1].set_title('Box Plot of Price')
plt.tight_layout()
plt.show()



sns.set(style="whitegrid")
# Create histograms for numerical variables
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# Year
sns.histplot(df['year'], bins=20, kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Distribution of Year')
# Engine
sns.histplot(df['engine'], bins=20, kde=True, ax=axes[0, 1], color='lightgreen')
axes[0, 1].set_title('Distribution of Engine Capacity (cc)')
# Mileage
sns.histplot(df['mileage'], bins=20, kde=True, ax=axes[1, 0], color='salmon')
axes[1, 0].set_title('Distribution of Mileage')
# Price
sns.histplot(df['price'], bins=20, kde=True, ax=axes[1, 1], color='gold')
axes[1, 1].set_title('Distribution of Price')
plt.tight_layout()
plt.show()




# Create scatter plots to examine relationships between numerical variables
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
# Price vs Year
sns.scatterplot(x=df['year'], y=df['price'], ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Price vs Year')
# Price vs Engine
sns.scatterplot(x=df['engine'], y=df['price'], ax=axes[0, 1], color='lightgreen')
axes[0, 1].set_title('Price vs Engine Capacity (cc)')
# Price vs Mileage
sns.scatterplot(x=df['mileage'], y=df['price'], ax=axes[1, 0], color='salmon')
axes[1, 0].set_title('Price vs Mileage')
# Engine vs Mileage
sns.scatterplot(x=df['engine'], y=df['mileage'], ax=axes[1, 1], color='gold')
axes[1, 1].set_title('Engine Capacity vs Mileage')
plt.tight_layout()
plt.show()



