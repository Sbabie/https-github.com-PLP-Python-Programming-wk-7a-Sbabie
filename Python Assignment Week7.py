from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch the Iris dataset from UCI ML Repository
iris = fetch_ucirepo(id=53)

# Extract features and target
X = iris.data  # Features as DataFrame
y = iris.target  # Target labels

# Convert features to a DataFrame for easier handling if not already
X = pd.DataFrame(X, columns=iris.variables['features'])
y = pd.Series(y, name='species')

# Display metadata and variable info
print("Metadata:")
print(iris.metadata, '\n')

print("Variable Information:")
print(iris.variables, '\n')

# Display first few rows
print("First few rows of features:")
print(X.head())

# Check data types and missing values
print("\nData info:")
print(X.info())

print("\nMissing values per column:")
print(X.isnull().sum())

# Basic statistical description
print("\nBasic Statistics of Features:")
print(X.describe())

# Map numeric target to species names if available
# Check if variable info has species names
species_names = None
for var in iris.variables:
    if var.get('name') == 'species' and 'values' in var:
        species_names = var['values']
        break

if species_names:
    y = y.map({i: name for i, name in enumerate(species_names)})

# Add species labels to the feature DataFrame for visualization
X['species'] = y

# Visualization Setup
sns.set(style='whitegrid')

# 1. Pairplot (scatterplot matrix) of features by species
sns.pairplot(X, hue='species', markers=["o", "s", "D"])
plt.suptitle('Pairplot of Iris Features by Species', y=1.02)
plt.show()

# 2. Bar chart: Average petal length per species
plt.figure(figsize=(8,6))
sns.barplot(x='species', y='petal length (cm)', data=X)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# 3. Histogram: Distribution of sepal width
plt.figure(figsize=(8,6))
sns.histplot(X['sepal width (cm)'], bins=20, kde=True)
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Line plot: Se pal length across samples (just for visualization)
plt.figure(figsize=(10,4))
plt.plot(X['sepal length (cm)'], label='Sepal Length')
plt.title('Sepal Length across Samples')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()
