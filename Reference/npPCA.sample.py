'''
https://towardsdatascience.com/pca-with-numpy-58917c1d0391
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                  header=None)
iris.columns = ["sepal_length","sepal_width",
                'petal_length','petal_width','species']
iris.dropna(how='all', inplace=True)
iris.head()

# Plotting data using seaborn
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12,8)
sns.scatterplot(x = iris.sepal_length, y=iris.sepal_width,
               hue = iris.species, style=iris.species)


def standardize_data(arr): 
	'''
	This function standardze an aray, its substracts mean value, 
	and then divide the standard deviation. 

	param 1: array 
	return: standardized array 
	''' 
	rows, columns = arr.shape 

	standardizedArray = np.zeros(shape=(rows, columns)) 
	tempArray = np.zeros(rows)  

	for column in range(columns): 

		mean = np.mean(X[:, column]) 
		std = np.std(X[:, column]) 
		tempArray = np.empty(0) 

		for element in X[:, column]: 

			tempArray = np.append(tempArray, ((element-mean)/std)) 

		standardizedArray[:, column] = tempArray 

	return standardizedArray 


# Standardizing data 

X = iris.iloc[:, 0:4].values 
y = iris.species.values 

X = standardize_data(X)  


# Calculating the covariance matrix 

covariance_matrix = np.cov(X.T)  


# Using np.linalg.eig function  

eigen_values, eigene_vectors = np.linalg.eig(covariance_matrix) 
print("Eigenvector: \n", eigene_vectors, "\n") 
print("Eigenvalues: \n", eigen_values, "\n") 


# calculating the explained variance on each of compnents 

variance_explained = [] 
for i in eigen_values: 
	variance_explained.append((i / sum(eigen_values)) * 100) 

print(variance_explained) 


# Identifying compnents tahta explain at least 95%  

cumulative_variance_explained = np.cumsum(variance_explained) 
print(cumulative_variance_explained) 


# Visualizing the eigenvalues and finding the "elbow" in the graphic
sns.lineplot(x = [1,2,3,4], y=cumulative_variance_explained)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Explained variance vs Number of components")


# Using two first components (because those explain more than 95%)
projection_matrix = (eigen_vectors.T[:][:2]).T
print(projection_matrix)


# Getting the product of original standardized X and the eigenvectors 
X_pca = X.dot(projection_matrix)
print(X_pca)

