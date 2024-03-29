import os
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import spacy
import en_core_web_sm
import pandas as pd
import polars as pl
from sklearn.model_selection import RandomizedSearchCV, cross_validate, cross_val_score, train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC, LinearSVR, NuSVC, NuSVR
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from gensim.models import Word2Vec

nlp = en_core_web_sm.load()

data = []
labels = []

for category in ['pos', 'neg']:
    folder_path = os.path.join('imdb_smol', category)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            data.append(file.read())
            labels.append(category)

def preprocess_text(text):
    doc = nlp(text)
    
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    
    return tokens

# def preprocess_text(text):

#     tokens = [token.lower() for token in text.split() if token.isalpha()]
    
#     return tokens

if (os.path.exists('pre_processed_data.txt')):
    print("Reading pre-processed data...")
    with open('pre_processed_data.txt', 'r', encoding='utf-8') as file:
        data = file.readlines()
 
else:
    print("Pre-processing data...")
    for i in range(len(data)):
        data[i] = ' '.join(preprocess_text(data[i]))

    with open('pre_processed_data.txt', 'w', encoding='utf-8') as file:
        for i in range(len(data)):
            file.write(data[i] + '\n')

df = pd.DataFrame({'text': data, 'label': labels})

X = df['text']
y = df['label']

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X = vectorizer.fit_transform(X)

# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(X)


models = {
    'LinearSVC': LinearSVC(dual=False, C=0.344),
    'LogisticRegression': LogisticRegression(max_iter=10000, C = 0.483),
    'SVCKernel': SVC(kernel='rbf', C = 0.6),
    'NuSVC': NuSVC(nu = 0.301),
    'RandomForest': RandomForestClassifier()
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'C': np.around(np.linspace(0.301, 0.6, 300), 3).tolist()
}


for name, model in models.items():
    print()
    print(f"Evaluating {name}...")
    y_pred = cross_val_predict(model, X, y, cv=cv)
    print(classification_report(y, y_pred, digits=4))
    

    print(accuracy_score(y, y_pred))


    # grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    # grid_search.fit(X, y)
    # results = pd.DataFrame(grid_search.cv_results_)
    # print(results[['params', 'mean_test_score', 'rank_test_score']].sort_values('rank_test_score'))
