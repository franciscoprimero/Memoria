import time
import timeit
import numpy as np
import os
#from sklearn.neighbors import KNeighborsClassifier
from utils.clasificacion import get_best_score

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.datasets import mnist
from keras import optimizers

from bob.learn.linear import GFKTrainer

from sklearn.decomposition import PCA

from mSDA import msda, md
from mSDA.msda_theano import mSDATheano

from sklearn.externals import joblib
import theano.tensor as T

###########
### GFK ###
###########

def adapt_gfk(X_src, X_tgt, n_subs, subs_dim):
    gfk_trainer = GFKTrainer(n_subs, subspace_dim_source=subs_dim, subspace_dim_target=subs_dim)
    gfk_machine = gfk_trainer.train(X_src, X_tgt, norm_inputs=False)

    return gfk_machine

def transform_gfk(X, gfk_machine):
    K = gfk_machine.G
    return np.dot(X, K)


def gfk_compute_accuracy(K, Xs, Ys, Xt, Yt):
    np.random.seed(10)

    source = np.diag(np.dot(np.dot(Xs, K), Xs.T))
    source = np.reshape(source, (Xs.shape[0], 1))
    source = np.matlib.repmat(source, 1, Yt.shape[0])

    target = np.diag(np.dot(np.dot(Xt, K), Xt.T))
    target = np.reshape(target, (Xt.shape[0], 1))
    target = np.matlib.repmat(target, 1, Ys.shape[0]).T

    dist = source + target - 2 * np.dot(np.dot(Xs, K), Xt.T)

    indices = np.argmin(dist, axis=0)
    prediction = Ys[indices]

    accuracy = sum(prediction == Yt) / float(Yt.shape[0])

    return accuracy

def gfk_compute_knn(K, Xs, Ys, Xt, Yt):
    Xs = np.dot(Xs, K)
    Xt = np.dot(Xt, K)

    neigh = KNeighborsClassifier().fit(Xs, Ys)

    return neigh.score(Xt, Yt)

def gfk_compute_svc(K, Xs, Ys, Xt, Yt):
    Xs = np.dot(Xs, K)
    Xt = np.dot(Xt, K)

    svc = get_best_score(Xs, Ys)
    return svc.score(Xt, Yt)

def gfk_train_all(X_src, X_tgt, parameters, folder_path, prefix=""):
    """
    Entrena distintos modelos de GFK con diferentes parametros.



    parametros: {
        dims: [int],
        n_subs: [int]

    }


    """
    i=0
    paths_list = []

    # se itera sobre la combinaciones posibles
    for dim in parameters['dims']:
        for n_subs  in parameters['n_subs']:
            # se entrena un adaptador con GFK
            print "\tEntrenando modelo %d" % i
            gfk = adapt_gfk(X_src, X_tgt, n_subs, dim)

            # se une la ruta de la carpeta de modelos, el prefijo y el nombre
            # para guardar el modelo
            full_path = os.path.join(folder_path, "%s%d.pkl" % (prefix, i))
            print "\tGuardando modelo en %s" % full_path
            joblib.dump(gfk, full_path)

            paths_list.append(full_path)
            i = i+1

    return paths_list

def gfk_grid_search(X_src, y_src, X_tgt, parameters, n_jobs=2):
    """
    gfk_grid_search

    Realiza Grid Search con los parametros dados y
    obtiene el mejor modelo realizando CV con los datos del dominio fuente.

    parameters: {
        dims: [int],so
        n_subs: [int]

    }
    """
    best_gfk = None
    best_score = None

    for dim in parameters['dims']:
        for n in parameters['n_subs']:
            print "\tn: %d - d: %d" % (n, dim),
            gfk = adapt_gfk(X_src, X_tgt, n, dim)
            X_src2 = transform_gfk(X_src, gfk)

            clf = get_best_score(X_src2, y_src, 'SVC', n_jobs=n_jobs)

            print clf.best_score_

            if best_score is None:
                best_score = clf.best_score_
                best_gfk = gfk
            elif clf.best_score_ > best_score:
                best_score = clf.best_score_
                best_gfk = gfk

    return best_gfk, best_score

#############
#### SDA ####
#############

def create_SDA(input_size, layers, noise):
    input_layer = Input(shape=(input_size,))
    encoded = None
    decoded = None

    for layer in layers:
        if encoded is None:
            encoded = Dense(layer, activation='sigmoid')(input_layer)
            #masking noise
            encoded = Dropout(noise)(encoded)
        else:
            encoded = Dense(layer, activation='softplus')(encoded)


    for layer in layers[-2::-1]:
        if decoded is None:
            decoded = Dense(layer, activation='relu')(encoded)
        else:
            decoded = Dense(layer, activation='relu')(decoded)


    if len(layers) == 1:
            decoded = Dense(input_size, activation='sigmoid')(encoded)
    else:
        decoded = Dense(input_size, activation='sigmoid')(decoded)

    encoder = Model(input_layer, encoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='sgd', loss='binary_crossentropy')

    return autoencoder, encoder

def sda_pseudo_grid_search(X_train, X_val, parameters, folder_path, prefix=""):
    i=0
    saved_paths = []
    dims = X_train.shape[1]

    for noise in parameters['noises']:
        for layers in parameters['layers']:
            for epoch in parameters['epochs']:
                print "pr: %.3f - epochs: %d - layers: %s" % (noise, epoch, layers)

                #se crea un sda
                autoencoder, encoder = create_SDA(dims, layers, noise)

                print "\tEntrenando autoencoder..."
                autoencoder.fit(X_train, X_train,
                    epochs=epoch,
                    batch_size=256,
                    shuffle=True,
                    verbose=0,
                    validation_data=(X_val, X_val))


                #guardar el modelo
                ae_save_path = os.path.join(folder_path, "%sae_%d.h5" % (prefix, i))
                e_save_path = os.path.join(folder_path, "%se_%d.h5" % (prefix, i))

                print "\tGuardando autoencoder en %s" % ae_save_path
                autoencoder.save(ae_save_path)
                print "\tGuardando encoder en %s" % e_save_path
                encoder.save(e_save_path)

                saved_paths.append({
                    'autoencoder': ae_save_path,
                    'encoder': e_save_path
                })

                i = i+1

    return saved_paths

def sda_grid_search(X_train, X_val, X_test, y_test, parameters, n_jobs=2):
    """
    sda_grid_search

    Realiza Grid Search con los parametros dados y
    obtiene el mejor modelo realizando CV con los datos del dominio fuente.

    parameters: {
        'noises': [float],
        'layers': [int],
        'epochs': [int],
    }
    """
    best_ae = None
    best_e = None
    best_score = None

    dims = X_train.shape[1]

    for noise in parameters['noises']:
        for layers in parameters['layers']:
            for epoch in parameters['epochs']:
                print "\tpr: %.3f - epochs: %d - layers: %s" % (noise, epoch, layers)

                #se crea un sda
                autoencoder, encoder = create_SDA(dims, layers, noise)

                print "\tEntrenando autoencoder..."
                autoencoder.fit(X_train, X_train,
                    epochs=epoch,
                    batch_size=256,
                    shuffle=True,
                    verbose=0,
                    validation_data=(X_val, X_val))


                # adaptar datos de prueba
                X_test2 = encoder.predict(X_test)

                # probar datos de prueba
                clf = get_best_score(X_test2, y_test, 'SVC', n_jobs=n_jobs)

                print "\t%.3f\n" % clf.best_score_

                if best_score is None:
                    best_score = clf.best_score_
                    best_ae = autoencoder
                    best_e = encoder
                elif clf.best_score_ > best_score:
                    best_score = clf.best_score_
                    best_ae = autoencoder
                    best_e = encoder

    return best_ae, best_e, best_score

##########
## mSDA ##
##########

def adapt_msda(x_src, pr=0.5, n_layers=1):
    """
    Funcion encargada de realizar la adaptacion con mSDA.

    x_src: matriz X del dominio fuente
    x_tgt: matriz X del dominio objetivo
    pr: probabilidad de corrupcion
    n_layers: numero de capas ocultas del mSDA

    t_adaptar: tiempo demorado en realizar la adaptacion
    train_mappings: mapeos de la red que permiten transformar los datos.
    """

    i = time.time()
    # se entrena un mSDA con los datos de ambos dominios juntos
    train_mappings, train_reps = msda.mSDA(x_src, pr, n_layers)

    f = time.time()
    t_adaptar = f-i

    return t_adaptar, train_mappings


# se entrenan varios mSDA con distintos parametros utilizando
# todos los datos disponibles
def msda_theano_pseudo_grid_search(X, parameters, folder_path, prefix=""):
    i = 0
    paths_list = []
    x = T.dmatrix('x')

    for noise in parameters['noises']:
        for layer in parameters['layers']:
            print "\tpr: %.3f - l: %d" % (noise, layer)

            #entrenar el mSDA
            new_msda = mSDATheano(x, layer, noise)
            t_adaptar = new_msda.fit(X)

            # se crea un diccionario para almacenar el modelo y sus datos
            new_model = {
                'pr': noise,
                'l': layer,
                'model': new_msda,
                'time': t_adaptar,
            }

            #guardar el modelo
            full_path = os.path.join(folder_path, "%s%d.pkl" % (prefix, i))

            print "\tGuardando modelo en %s" % full_path
            joblib.dump(new_model, full_path)
            paths_list.append(full_path)
            i = i+1

    return paths_list


def msda_theano_grid_search(X_train, X_test, y_test, parameters, n_jobs=2):
    """
    msda_theano_grid_search

    Realiza Grid Search con los parametros dados y
    obtiene el mejor modelo realizando CV con los datos del dominio fuente.

    parameters: {
        layers: [int],
        noises: [double]

    }
    """
    x = T.dmatrix('x')

    best_msda = None
    best_score = None

    for noise in parameters['noises']:
        for layer in parameters['layers']:
            print "\tpr: %.3f - l: %d" % (noise, layer)
            #entrenar el mSDA
            new_msda = mSDATheano(x, layer, noise)
            t_adaptar = new_msda.fit(X_train)

            # adaptar datos del dominio fuente
            X_test2 = new_msda.predict(X_test)
            clf = get_best_score(X_test2, y_test, 'SVC', n_jobs=n_jobs)

            print "\t%.3f\n" % clf.best_score_

            if best_score is None:
                best_score = clf.best_score_
                best_msda = new_msda
            elif clf.best_score_ > best_score:
                best_score = clf.best_score_
                best_msda = new_msda

    return best_msda, best_score

##########
## PCA ###
##########

def pca_pseudo_grid_search(X_train, parameters, folder_path, prefix=""):
    i = 0
    saved_paths = []

    for n_components in parameters['n_components']:
        print "\tn_components: %d" % (n_components)

        #se crea un modelo PCA
        new_model = PCA(n_components=n_components)

        print "\tEntrenando modelo PCA..."
        new_model.fit(X_train)

        model_save_path = os.path.join(folder_path, "%s%d.pkl" % (prefix, i))
        print "\tGuardando modelo en %s\n" % model_save_path

        joblib.dump(new_model, model_save_path)

        saved_paths.append(model_save_path)
        i =i+1

    return saved_paths

def pca_grid_search(X_train, X_test, y_test, parameters, n_jobs=2):
    """
    pca_grid_search

    Realiza Grid Search con los parametros dados y
    obtiene el mejor modelo realizando CV con los datos del dominio fuente.

    parameters: {
        n_components: [int],
    }
    """
    best_pca = None
    best_score = None

    for n_components in parameters['n_components']:
        print "\tn_components: %d" % (n_components)

        # entrenar PCA
        new_model = PCA(n_components=n_components)
        new_model.fit(X_train)

        # adaptar datos del dominio fuente
        X_test2 = new_model.transform(X_test)
        clf = get_best_score(X_test2, y_test, 'SVC', n_jobs=n_jobs)

        print "\t%.3f\n" % clf.best_score_

        if best_score is None:
            best_score = clf.best_score_
            best_pca = new_model
        elif clf.best_score_ > best_score:
            best_score = clf.best_score_
            best_pca = new_model

    return best_pca, best_score
