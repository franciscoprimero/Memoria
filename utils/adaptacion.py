import time

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.noise import GaussianNoise
from keras.datasets import mnist
from keras import optimizers

from bob.learn.linear import GFKTrainer


from mSDA import msda, md


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


#############
#### SDA ####
#############

def create_SDA(input_size, layers):
    input_layer = Input(shape=(input_size,))    
    encoded = None
    decoded = None
    
    for layer in layers:
        if encoded is None:
            encoded = Dense(layer, activation='sigmoid')(input_layer)
            #masking noise
            encoded = GaussianNoise(1)(encoded)
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

##########
## PCA ###
##########

def adapt_pca(X, dims):
    return X