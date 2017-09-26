import cPickle as pkl
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.externals import joblib

def get_best_score(X_tr, y_tr, classifier='SVC', n_jobs=2):

    if classifier is 'SVC':
        parametros = [{
            'kernel': ['linear'],
            'C': [1, 10, 100, 1000],
            'cache_size': [7000],
            'max_iter': [50000],
        }]

        clf = GridSearchCV(SVC(), parametros, cv=5, n_jobs = n_jobs, scoring='roc_auc')
        clf.fit(X_tr, y_tr)

    elif classifier is 'KNeighbors':
        parametros = [{'n_neighbors': [10, 50, 100]}]

        clf = GridSearchCV(KNeighborsClassifier(), parametros, cv=5, n_jobs = n_jobs)
        clf.fit(X_tr, y_tr)

    return clf


def load_best_score(path, X_tr, y_tr):
    if os.path.exists(path):
        print "Cargando modelo existente."
        modelo = joblib.load(path)
    else:
        print "Creando mejor modelo."
        modelo = get_best_score(X_tr, y_tr, classifier='SVC', n_jobs=4)
        joblib.dump(modelo, path)
        print "Guardando mejor modelo."

    return modelo
