import cPickle as pkl
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.externals import joblib

def get_best_score(X_tr, y_tr, clasifier='SVC'):
    
    if clasifier is 'SVC':
        parametros = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        score = 'precision'
    
        clf = GridSearchCV(SVC(C=1), parametros, cv=5, scoring='%s_macro' % score, n_jobs = 2)
        clf.fit(X_tr, y_tr)

    elif clasifier is 'KNeighbors':
        parametros = [{'n_neighbors': [10, 50, 100]}]
        score = 'precision'
        
        clf = GridSearchCV(KNeighborsClassifier(), parametros, cv=5, scoring='%s_macro' % score, n_jobs = 2)
        clf.fit(X_tr, y_tr)
        
    return clf


def load_best_score(path, X_tr, y_tr):
    if os.path.exists(path):
        print "Cargando modelo existente."
        modelo = joblib.load(path)
    else:
        print "Creando mejor modelo."
        modelo = get_best_score(X_tr, y_tr)
        joblib.dump(modelo, path)
        print "Guardando mejor modelo."

    return modelo
