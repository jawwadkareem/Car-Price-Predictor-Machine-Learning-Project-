import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../Datacleaning/pakwheels_preprocessed_data.csv')


transmission_count = df['transmission'].value_counts().reset_index()
plt.figure(figsize=(12,6))
plt.pie(x='count', labels='transmission', autopct='%1.1f%%', data=transmission_count)
plt.tight_layout()
plt.show()


fuel_count = df['fuel'].value_counts().reset_index()
plt.figure(figsize=(12,6))
plt.pie(x='count', labels='fuel', autopct='%1.1f%%', data=fuel_count)
plt.tight_layout()
plt.show()

makers_count = df['make'].value_counts().reset_index()
plt.figure(figsize=(12,6))
plt.pie(x='count', labels='make', autopct='%1.1f%%', data=makers_count)
plt.tight_layout()
plt.show()



#Mean mileage by year
mean_mileage_by_year = df.groupby('year')['mileage'].mean()
plt.figure(figsize=(12, 6))
plt.plot(mean_mileage_by_year.index, mean_mileage_by_year.values, marker='o', linestyle='-')
plt.xlabel('Year of Production')
plt.ylabel('Mean Mileage (kms)')
plt.title('Relationship Between Year of Production and Mean Mileage')
plt.grid(True)
plt.show()



# Average Engine Displacement by Car Make
engine_make_avg = df.groupby(['make'])['engine'].mean().reset_index()
plt.figure(figsize=(12,6))
sns.barplot(x='make', y='engine', data=engine_make_avg)
plt.xticks(rotation=90)
plt.xlabel('Car Make')
plt.ylabel('Average Engine Displacement (cc)')
plt.title('Average Engine Displacement by Car Make')
plt.tight_layout()
plt.show()

# Average Price by Car Make
make_avg_price = df.groupby('make')['price'].mean()
plt.figure(figsize=(12, 6))
make_avg_price.plot(kind='bar')
plt.xlabel('Car Make')
plt.ylabel('Average Price')
plt.title('Average Price by Car Make')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



# Top 10 car makers in dataset
top_10 = df['make'].value_counts().head(10).reset_index()
top_10.columns  = ['Cars','Counts']
top_10.index+=1
print(top_10)


# Bottom 10 car makers in dataset
bot_10 = df['make'].value_counts().tail(10).reset_index()
bot_10.columns  = ['Cars','Counts']
bot_10.index+=1
print(bot_10)

fuel_count = df['fuel'].value_counts().reset_index()
print(fuel_count)
