from sklearn import datasets
import pandas as pd
import polars as pl
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

wine = datasets.load_wine()
# type(wine)

# print(wine.DESCR)
# print(wine.target_names)
# print(wine.feature_names)

df = pd.DataFrame(data=wine.data,columns=wine.feature_names)
df["target"] = wine.target
df.head()

# print(df)

df = pl.DataFrame(
    data=wine.data, schema=wine.feature_names).with_columns(
    target=pl.Series(wine.target)
)
df.head()

# print(df)

X_wine, y_wine = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.3)
y_train


# plt.hist(y_train, align="right", label="train") 
# plt.hist(y_test, align="left", label="test")
# plt.legend()
# plt.xlabel("Classe")
# plt.ylabel("Nombre d'exemples")
# plt.title("Répartition des classes") 
# plt.show()

X2_train, X2_test, y2_train, y2_test = train_test_split(X_wine, y_wine, test_size=0.25, stratify=y_wine)
# plt.hist(y_train, align="right", label="train") 
# plt.hist(y_test, align="left", label="test") 
# plt.legend()
# plt.xlabel("Classe")
# plt.ylabel("Nombre d'exemples")
# plt.title("Répartition des classes avec échantillonnage stratifié") 
# plt.show()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = []

for train_index, test_index in skf.split(X_wine, y_wine):
    X_train_fold, X_test_fold = X_wine[train_index], X_wine[test_index]
    y_train_fold, y_test_fold = y_wine[train_index], y_wine[test_index]
    
    # Initialiser et entraîner le modèle
    clf_fold = LinearSVC(dual="auto")
    clf_fold.fit(X_train_fold, y_train_fold)
    
    # Prédire les étiquettes sur l'ensemble de test
    y_pred_fold = clf_fold.predict(X_test_fold)
    
    # Calculer le score d'exactitude sur l'ensemble de test
    score_fold = accuracy_score(y_test_fold, y_pred_fold)
    
    # Stocker le score
    scores.append(score_fold)

    # print(classification_report(y_test_fold, y_pred_fold))

# Entraînement

clf = LinearSVC(dual="auto")
clf.fit(X_train, y_train)
clf.predict(X_test)

clf2 = LinearSVC(dual="auto")
clf2.fit(X2_train, y2_train)
clf2.predict(X2_test)

# print(clf.score(X_test, y_test))
# print(clf2.score(X2_test, y2_test))


# Évaluation

y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))

y2_pred = clf2.predict(X2_test)
# print(classification_report(y2_test, y2_pred))

mean_score = sum(scores) / len(scores)
# print("Scores de chaque fold :", scores)
# print("Score moyen :", mean_score)

# Validation croisée

# print(cross_validate(LinearSVC(dual="auto"), X_wine, y_wine)) # infos d'accuracy mais aussi de temps
# print(cross_val_score(LinearSVC(dual="auto"), X_wine, y_wine)) # uniquement accuracy


# Optimisation des hyperparamètres



param_grid =  {'C': [0.1, 0.5, 1, 10, 100, 1000], 'kernel':['linear']}
grid = GridSearchCV(SVC(), param_grid, cv = 5, scoring = 'accuracy')
estimator = grid.fit(X_wine, y_wine)
print(estimator.cv_results_)
df = pd.DataFrame(estimator.cv_results_)
df.sort_values('rank_test_score')
print(df)

# plt.plot(estimator.cv_results_['param_C'], estimator.cv_results_['mean_test_score'])
# plt.xlabel('C')
# plt.ylabel('Accuracy')
# plt.title('Accuracy en fonction de C')
# plt.show()


# Clustering

# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X_wine)
# y_kmeans = kmeans.predict(X_wine)

# print(silhouette_score(X_wine, y_kmeans))

# plt.scatter(X_wine[:, 0], X_wine[:, 1], c=y_kmeans, s=50, cmap='viridis')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
# plt.show()


categories = [
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
]

data_train = fetch_20newsgroups(
    subset="train",
    categories=categories,
    shuffle=True,
)

data_test = fetch_20newsgroups(
    subset="test",
    categories=categories,
    shuffle=True,
)

# print(len(data_train.data))
# print(len(data_test.data))

vectorizer = CountVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(data_train.data) # données de train vectorisées
y_train = data_train.target
X_train.shape

# print(X_train[0, :])

X_test = vectorizer.transform(data_test.data)
y_test = data_test.target

clf = LinearSVC(C=0.5, dual='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))

vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    max_df=0.5,
    stop_words='english'
)
X_train = vectorizer.fit_transform(data_train.data) # données de train vectorisées
y_train = data_train.target
X_train.shape

X_test = vectorizer.transform(data_test.data)
y_test = data_test.target

clf = LinearSVC(C=0.5, dual='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
