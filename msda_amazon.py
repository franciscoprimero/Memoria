import os
import argparse

#carga de datasets
from utils.DatasetStorage import Dataset
from utils.paths import *

#clasificadores
from utils.clasificacion import *

#adaptacion
from mSDA import msda
from utils.adaptacion import *

#otros
import numpy as np
import pandas as pd

tipo = pruebas[1]
dataset_name = datasets[0]
dims = dimensions[dataset_name]

print "\n\n############\n\n"


def crear_modelos(X, parametros):
    saved_paths = msda_pseudo_grid_search(X, parametros, models_path, tipo, dataset_name)
    print "\nModelos creados.\n"

    return saved_paths

def main(modo):

    # cargando dataset Amazon
    print "Cargando dataset ", dataset_name
    dataset_path = os.path.join(data_path, dataset_name+'.pkl')
    dataset_object = Dataset().load(dataset_path)

    dataset_object.split_dataset(test_size=0.2)

    labeled = dataset_object.labeled
    unlabeled = dataset_object.unlabeled
    domains = dataset_object.domains

    # se obtienen todos los valores X disponibles para realizar adaptacion
    X = dataset_object.get_all_X()
    X = np.asarray(X)
    X = X[:15000,:]
    print "Todos los datos disponibles obtenidos"

    msda_paths = os.path.join(models_path, tipo, "%s_paths.pkl" % dataset_name)

    if modo == 0:
        if os.path.exists(msda_paths):
            print "Modelos adaptados ya existen."
            return


        noises = [0.3, 0.5, 0.8]
        layers_sizes = [1, 3, 5]

        parametros = {
            'noises': noises,
            'layers': layers_sizes
        }

        saved_paths = crear_modelos(X.transpose(), parametros)
        joblib.dump(saved_paths, msda_paths)
        print "Rutas de los modelos guardadas en ", msda_paths

    print "\nEjecucion terminada."

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--modo",
                        type=int,
                        default=0,
                        help="Acciones a realizar: 0: Crear modelos para adaptar\n1: Obtener mejor modelo por cada dominio\n2: Realizar pruebas con los mejores modelos.\n")

    args = parser.parse_args()

    main(args.modo)
