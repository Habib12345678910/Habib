#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# Chargement des données
df = pd.read_csv('books.csv')
last_col_index = df.columns.get_loc(df.columns[-1])
df = df.iloc[:, :-1]
df1= pd.read_csv('ratings.csv')
data = pd.merge(df,df1)

# Sélection d'un échantillon de données
data = data.iloc[:1000000, :]

# Création d'une table pivot
user_movie_table = data.pivot_table(index=["title"], columns=["user_id"], values="rating").fillna(0)

# Division des données en ensembles d'entraînement et de test
train_data, test_data = train_test_split(user_movie_table, test_size=0.2, random_state=42)

# Conversion de la table en une matrice creuse
train_matrix = csr_matrix(train_data.values)

# Calcul des similarités avec l'algorithme KNN
model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
model_knn.fit(train_matrix)

# Récupération des indices des livres similaires pour chaque livre de test
test_indices = []
for i in range(len(test_data)):
    query_index = i
    distances, indices = model_knn.kneighbors(test_data.iloc[query_index, :].values.reshape(1, -1), n_neighbors=8)
    test_indices.append(indices.flatten()[1:])

# Regroupement des livres en clusters avec l'algorithme K-means
kmeans = KMeans(n_clusters=5)
kmeans.fit(train_data.values)

# Pour chaque livre de test, on récupère les indices des livres similaires, leur cluster, 
# on filtre les livres similaires pour ne garder que ceux du même cluster que le livre de test,
# puis on calcule la précision et le rappel
precision = 0
recall = 0
for i, indices in enumerate(test_indices):
    query_index = i
    clusters = kmeans.labels_
    similar_books_indices = indices
    similar_books_clusters = clusters[similar_books_indices]
    filtered_books_indices = similar_books_indices[similar_books_clusters == clusters[query_index]]
    recommended_books = train_data.index[filtered_books_indices]
    actual_books = test_data.iloc[i, :].nonzero()[0]
    tp = len(set(recommended_books) & set(actual_books))
    fp = len(set(recommended_books) - set(actual_books))
    fn = len(set(actual_books) - set(recommended_books))
    precision += tp / (tp + fp)
    recall += tp / (tp + fn)

# Calcul de la précision et du rappel moyens
precision /= len(test_data)
recall /= len(test_data)

# Affichage des résultats
print(f"Précision : {precision:.2f}")
print(f"Rappel : {recall:.2f}")


# In[3]:


import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# Chargement des données
df = pd.read_csv('books.csv')
last_col_index = df.columns.get_loc(df.columns[-1])
df = df.iloc[:, :-1]
df1= pd.read_csv('ratings.csv')
data = pd.merge(df,df1)

# Sélection d'un échantillon de données
data = data.iloc[:1000000, :]

# Création d'une table pivot
user_movie_table = data.pivot_table(index=["title"], columns=["user_id"], values="rating").fillna(0)

# Division des données en ensembles d'entraînement et de test
train_data, test_data = train_test_split(user_movie_table, test_size=0.2, random_state=42)

# Conversion de la table en une matrice creuse
train_matrix = csr_matrix(train_data.values)

# Calcul des similarités avec l'algorithme KNN
model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
model_knn.fit(train_matrix)

# Récupération des indices des livres similaires pour chaque livre de test
test_indices = []
for i in range(len(test_data)):
    query_index = i
    distances, indices = model_knn.kneighbors(test_data.iloc[query_index, :].values.reshape(1, -1), n_neighbors=8)
    test_indices.append(indices.flatten()[1:])

# Regroupement des livres en clusters avec l'algorithme K-means
kmeans = KMeans(n_clusters=5)
kmeans.fit(train_data.values)

# Pour chaque livre de test, on récupère les indices des livres similaires, leur cluster, 
# on filtre les livres similaires pour ne garder que ceux du même cluster que le livre de test,
# puis on calcule la précision et le rappel
precision = 0
recall = 0
for i, indices in enumerate(test_indices):
    query_index = i
    clusters = kmeans.labels_
    similar_books_indices = indices
    similar_books_clusters = clusters[similar_books_indices]
    filtered_books_indices = similar_books_indices[similar_books_clusters == clusters[query_index]]
    recommended_books = train_data.index[filtered_books_indices]
    actual_books = test_data.iloc[i, :].to_numpy().nonzero()[0]
    tp = len(set(recommended_books) & set(actual_books))
    fp = len(set(recommended_books) - set(actual_books))
    fn = len(set(actual_books) - set(recommended_books))
    if tp + fp == 0:
        precision += 0
    else:
        precision += tp / (tp + fp)
    recall += tp / (tp + fn)

# Calcul de la précision et du rappel moyens
precision /= len(test_data)
recall /= len(test_data)

# Affichage des résultats
print(f"Précision : {precision:.2f}")
print(f"Rappel : {recall:.2f}")


# In[4]:


import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Chargement des données
df = pd.read_csv('books.csv')
last_col_index = df.columns.get_loc(df.columns[-1])
df = df.iloc[:, :-1]
df1= pd.read_csv('ratings.csv')
data = pd.merge(df,df1)

# Sélection d'un échantillon de données
data = data.iloc[:1000000, :]

# Création d'une table pivot
user_movie_table = data.pivot_table(index=["title"], columns=["user_id"], values="rating").fillna(0)

# Division des données en ensembles d'entraînement et de test
train_data, test_data = train_test_split(user_movie_table, test_size=0.2, random_state=42)

# Conversion de la table en une matrice creuse
train_matrix = csr_matrix(train_data.values)

# Calcul des similarités avec l'algorithme KNN
model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
model_knn.fit(train_matrix)

# Récupération des indices des livres similaires pour chaque livre de test
test_indices = []
for i in range(len(test_data)):
    query_index = i
    distances, indices = model_knn.kneighbors(test_data.iloc[query_index, :].values.reshape(1, -1), n_neighbors=8)
    test_indices.append(indices.flatten()[1:])

# Regroupement des livres en clusters avec l'algorithme K-means
kmeans = KMeans(n_clusters=5)
kmeans.fit(train_data.values)

# Pour chaque livre de test, on récupère les indices des livres similaires, leur cluster, 
# on filtre les livres similaires pour ne garder que ceux du même cluster que le livre de test,
# puis on calcule le RMSE
rmse = 0
for i, indices in enumerate(test_indices):
    query_index = i
    clusters = kmeans.labels_
    similar_books_indices = indices
    similar_books_clusters = clusters[similar_books_indices]
    filtered_books_indices = similar_books_indices[similar_books_clusters == clusters[query_index]]
    recommended_books = train_data.iloc[filtered_books_indices, :]
    actual_books = test_data.iloc[i, :]
    rmse += mean_squared_error(actual_books[actual_books.nonzero()], recommended_books[actual_books.nonzero()], squared=False)

# Calcul du RMSE moyen
rmse /= len(test_data)

# Affichage du résultat
print(f"RMSE : {rmse:.2f}")


# In[5]:


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Chargement des données
df = pd.read_csv('books.csv')
last_col_index = df.columns.get_loc(df.columns[-1])
df = df.iloc[:, :-1]
df1 = pd.read_csv('ratings.csv')
data = pd.merge(df, df1)

# Sélection d'un échantillon de données
data = data.iloc[:1000000, :]

# Création d'une table pivot
user_movie_table = data.pivot_table(index=["title"], columns=["user_id"], values="rating").fillna(0)

# Division des données en ensembles d'entraînement et de test
train_data, test_data = train_test_split(user_movie_table, test_size=0.2, random_state=42)

# Conversion de la table en une matrice creuse
train_matrix = csr_matrix(train_data.values)

# Calcul des similarités avec l'algorithme KNN
model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
model_knn.fit(train_matrix)

# Récupération des indices des livres similaires pour chaque livre de test
test_indices = []
for i in range(len(test_data)):
    query_index = i
    distances, indices = model_knn.kneighbors(test_data.iloc[query_index, :].values.reshape(1, -1), n_neighbors=8)
    test_indices.append(indices.flatten()[1:])

# Regroupement des livres en clusters avec l'algorithme K-means
kmeans = KMeans(n_clusters=5)
kmeans.fit(train_data.values)

# Pour chaque livre de test, on récupère les indices des livres similaires, leur cluster,
# on filtre les livres similaires pour ne garder que ceux du même cluster que le livre de test,
# puis on calcule la RMSE
rmse = 0
for i, indices in enumerate(test_indices):
    query_index = i
    clusters = kmeans.labels_
    similar_books_indices = indices
    similar_books_clusters = clusters[similar_books_indices]
    filtered_books_indices = similar_books_indices[similar_books_clusters == clusters[query_index]]
    recommended_books = train_data.iloc[filtered_books_indices, :]
    actual_books = test_data.iloc[i, :]
    predicted_ratings = recommended_books.mean(axis=0)
    predicted_ratings = predicted_ratings.fillna(0)
    actual_ratings = actual_books.to_numpy()
    rmse += mean_squared_error(actual_ratings, predicted_ratings, squared=False)

# Calcul de la RMSE moyenne
rmse /= len(test_data)

# Affichage du résultat
print(f"RMSE : {rmse:.2f}")


# In[6]:


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Chargement des données
df = pd.read_csv('books.csv')
last_col_index = df.columns.get_loc(df.columns[-1])
df = df.iloc[:, :-1]
df1 = pd.read_csv('ratings.csv')
data = pd.merge(df, df1)

# Sélection d'un échantillon de données
data = data.iloc[:1000000, :]

# Création d'une table pivot
user_movie_table = data.pivot_table(index=["title"], columns=["user_id"], values="rating").fillna(0)

# Division des données en ensembles d'entraînement et de test
train_data, test_data = train_test_split(user_movie_table, test_size=0.2, random_state=42)

# Conversion de la table en une matrice creuse
train_matrix = csr_matrix(train_data.values)

# Calcul des similarités avec l'algorithme KNN
model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
model_knn.fit(train_matrix)

# Récupération des indices des livres similaires pour chaque livre de test
test_indices = []
for i in range(len(test_data)):
    query_index = i
    distances, indices = model_knn.kneighbors(test_data.iloc[query_index, :].values.reshape(1, -1), n_neighbors=8)
    test_indices.append(indices.flatten()[1:])

# Regroupement des livres en clusters avec l'algorithme K-means
kmeans = KMeans(n_clusters=5)
kmeans.fit(train_data.values)

# Pour chaque livre de test, on récupère les indices des livres similaires, leur cluster,
# on filtre les livres similaires pour ne garder que ceux du même cluster que le livre de test,
# puis on calcule la RMSE et le MAE
rmse = 0
mae = 0
for i, indices in enumerate(test_indices):
    query_index = i
    clusters = kmeans.labels_
    similar_books_indices = indices
    similar_books_clusters = clusters[similar_books_indices]
    filtered_books_indices = similar_books_indices[similar_books_clusters == clusters[query_index]]
    recommended_books = train_data.iloc[filtered_books_indices, :]
    actual_books = test_data.iloc[i, :]
    predicted_ratings = recommended_books.mean(axis=0)
    predicted_ratings = predicted_ratings.fillna(0)
    actual_ratings = actual_books.to_numpy()
    rmse += mean_squared_error(actual_ratings, predicted_ratings, squared=False)
    mae += mean_absolute_error(actual_ratings, predicted_ratings)

# Calcul de la RMSE et du MAE moyens
rmse /= len(test_data)
mae /= len(test_data)

# Affichage des résultats
print(f"RMSE : {rmse:.2f}")
print(f"MAE : {mae:.2f}")


# In[ ]:




