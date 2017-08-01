import time

import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from utils.clasificacion import get_best_score

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.datasets import mnist
from keras import optimizers

from bob.learn.linear import GFKTrainer


from mSDA import msda, md

from sklearn.externals import joblib


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


def gfk_grid_search(parameters, X_src, y_src, X_tgt):
    """
    gfk_grid_search
    
    Realiza Grid Search con los parametros dados y
    obtiene el mejor modelo realizando CV con los datos del dominio fuente.
    
    parameters: {
        dims: [int],
        n_subs: [int]
    
    }
    """
    best_gfk = None
    best_score = None
    
    for dim in parameters['dims']:
        for n in parameters['n_subs']:
            #print "n: %d - d: %d" % (n, dim),
            gfk = adapt_gfk(X_src, X_tgt, n, dim)
            X_src2 = transform_gfk(X_src, gfk)
            
            clf = get_best_score(X_src2, y_src, 'KNeighbors')
            
            #print clf.best_score_
            
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


def sda_pseudo_grid_search(X, parameters, models_path, tipo, dataset_name):
    i=0
    saved_paths = []
    dims = X.shape[1]
    
    for noise in parameters['noise']:
        for layers in parameters['layers']:
            for epoch in parameters['epochs']:
                print "pr: %.3f - epochs: %d - layers: %s" % (noise, epoch, layers)

                #se crea un sda
                autoencoder, encoder = create_SDA(dims, layers, noise)
                
                #TODO: tomar el tiempo
                #TODO: datos de validacion
                print "\tEntrenando autoencoder..."
                autoencoder.fit(X, X,
                   epochs=epoch,
                   batch_size=256,
                   shuffle=True,
                   verbose=0,
                   validation_data=(X, X))
                
                ########################
                # esto se puede borrar #
                ########################
                new_model = {
                    'noise': noise,
                    'layers': layers,
                    'autoencoder': autoencoder,
                    'encoder': encoder,
                }
                
                #guardar el modelo
                ae_save_path = os.path.join(models_path, tipo, "%s_ae_%d.h5" % (dataset_name, i))
                e_save_path = os.path.join(models_path, tipo, "%s_e_%d.h5" % (dataset_name, i))
                
                print "\tGuardando autoencoder en %s" % ae_save_path
                #autoencoder.save(ae_save_path)
                print "\tGuardando encoder en %s" % e_save_path
                encoder.save(e_save_path)
                
                saved_paths.append({
                    'autoencoder': ae_save_path,
                    'encoder': e_save_path
                })
                
                i = i+1
                
    return saved_paths

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
def msda_pseudo_grid_search(X, parameters, models_path, tipo, dataset_name):
    i = 0
    saved_paths = []
    
    for noise in parameters['noises']:
        for layer in parameters['layers']:
            print "pr: %.3f - l: %d" % (noise, layer)
            
            #entrenar el mSDA
            t_adaptar, train_mappings = adapt_msda(X, pr=noise, n_layers=layer)
            
            # se crea un diccionario para almacenar el modelo y sus datos
            new_model = {
                'pr': noise,
                'l': layer,
                'mapping': train_mappings,
                'time': t_adaptar,
            }
            
            #guardar el modelo
            msda_save_path = os.path.join(models_path, tipo, "%s_%d.pkl" % (dataset_name, i))
            
            
            print "Guardando modelo en %s" % msda_save_path
            joblib.dump(new_model, msda_save_path)
            saved_paths.append(msda_save_path)
            i = i+1
    
    return saved_paths

##########
## PCA ###
##########

def adapt_pca(X, dims):
    return X