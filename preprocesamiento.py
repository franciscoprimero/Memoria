"""preprocesamiento."""
import os
import argparse

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from nltk.corpus import stopwords
import pandas as pd

from utils.DatasetStorage import Dataset
from utils.paths import *


def read_amazon_file(path, labeled=True):
    """read_amazon_file."""
    file = open(path)
    comentarios = []
    labels = []
    for line in file:
        line = line.split("#label#:")
        labels.append(line[1][:-1])

        pares = line[0].split(" ")
        comentario = ""
        for par in pares:
            palabra = par.split(":")[0]
            palabra = str.replace(palabra, "_", " ")
            comentario = comentario + palabra + " "
        comentario = comentario + "."
        comentarios.append(comentario)

    if labeled:
        return comentarios, labels
    else:
        return comentarios


def read_all_amazon_domains(path):
    """read_all_amazon_domains."""
    file_names = ['positive.review', 'negative.review', 'unlabeled.review']

    domains = []

    labeled = {}
    unlabeled = {}

    print 'Leyendo dominio: '
    for folder in os.listdir(path):
        print "- %s" % folder

        instances = []
        labels = []
        for file_name in file_names[0:2]:
            file_path = os.path.join(path, folder, file_name)
            new_instances, new_labels = read_amazon_file(file_path)
            instances += new_instances
            labels += new_labels

        labeled[folder] = {
            'X': instances,
            'y': labels,
        }

        # datos sin etiquetas
        file_path = os.path.join(path, folder, file_names[2])
        instances = read_amazon_file(file_path, labeled=False)

        unlabeled[folder] = {
            'X': instances,
        }

        domains.append(folder)

    return labeled, unlabeled, domains


def read_twitter_files(dir_path, three_labels=False):
    """read_twitter_files."""


    domains = []
    labeled = {}

    print "Leyendo dominio: "
    # TODO: arreglar esto
    if three_labels:
        dir_path = raw_path+twitter_3_path


    for xls_file in os.listdir(dir_path):

        instances = []
        labels = []

        full_path = os.path.join(dir_path, xls_file)

        if ".xls" in full_path:
            domain = xls_file.split(".")[0]
            print "- %s" % domain

            data = pd.read_excel(full_path, header=1)[['Cuerpo', 'Sentido']]

            # eliminar retweets
            data.drop_duplicates("Cuerpo", keep='first')

            # dataframe a lista
            temp_labels = data.Sentido.values.tolist()
            temp_instances = data.Cuerpo.values.tolist()

            for instance, label in zip(temp_instances, temp_labels):
                words = instance.split(" ")
                words = [x.lower() for x in words if ("@" not in x and "#" not in x and "http" not in x and "RT" not in x)]
                instance = " ".join(words)

                if instance not in instances:
                    instances.append(instance)
                    labels.append(label)

            domains.append(domain)

            labeled[domain] = {
                'X': instances,
                'y': labels,
            }

    return labeled, domains


def preprocesar(labeled, unlabeled, dims, stop_words=None):
    """preprocesar."""

    instances = []
    labels = []
    ant = 0
    labeled_lims ={}
    unlabeled_lims = {}

    for v_l in labeled.values():
        instances += v_l['X']
        labels += v_l['y']

    if unlabeled is not None:
        for v_ul in unlabeled.values():
            instances += v_ul['X']



    x_cv = CountVectorizer(max_features=dims, ngram_range=(1, 2), binary=True, stop_words=stop_words)
    x_cv.fit(instances)

    y_cv = CountVectorizer()
    y_cv.fit(labels)


    for d_l in labeled:
        labeled[d_l]['X'] = x_cv.transform(labeled[d_l]['X'])
        labeled[d_l]['y'] = y_cv.transform(labeled[d_l]['y'])

    if unlabeled is not None:
        for d_ul in unlabeled:
            unlabeled[d_ul]['X'] = x_cv.transform(labeled[d_ul]['X'])

    return labeled, unlabeled


def main(dataset, dims):
    """main."""
    new_path = os.path.join(raw_path, raw_folders[dataset])
    if dataset == 'amazon':
        try:
            print 'Leyendo directorio %s' % new_path
            labeled, unlabeled, domains = read_all_amazon_domains(new_path)

            print 'Procesando datos.'
            labeled, unlabeled = preprocesar(labeled, unlabeled, dims)

            #dataset_path = data_path + 'amazon.pkl'
            dataset_path = os.path.join(data_path, dataset+'.pkl')
            print 'Guardando datos en %s' % dataset_path
            dataset_object = Dataset(labeled, unlabeled, domains)
            dataset_object.save(dataset_path)

            print 'Operacion terminada.'

        except Exception:
            print 'Error leyendo los datos de amazon.'

    elif dataset == 'twitter':
        try:
            print 'Leyendo directorio %s' % new_path
            labeled, domains = read_twitter_files(new_path)

            print 'Procesando datos.'
            stop_words = stopwords.words('spanish')
            labeled, unlabeled = preprocesar(labeled, None, dims)


            dataset_path = os.path.join(data_path, dataset+'.pkl')
            print 'Guardando datos en %s' % dataset_path
            dataset_object = Dataset(labeled, None, domains)
            dataset_object.save(dataset_path)

            print 'Operacion terminada.'

        except Exception:
            print 'Error leyendo los datos de twitter.'

    #elif dataset == 'twitter_3':
    #    try:
    #        print 'Leyendo directorio %s' % raw_path + twitter_3_path
    #        instances, labels, domains = read_twitter_files(True)

    #         print 'Procesando datos.'
    #        stop_words = stopwords.words('spanish')
    #        x_cv, x, y_cv, y = preprocesar(instances, labels, domains, dims, stop_words)

#            dataset_path = data_path + 'twitter_3.pkl'
#            print 'Guardando datos en %s' % dataset_path
#            dataset_object = Dataset(x, y, domains)
#            dataset_object.save(dataset_path)
#
#        except Exception:
#            print 'Error leyendo los datos de twitter.'
    else:
        print 'Dataset no encontrado'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        type=str,
                        default="amazon",
                        help="Data a preprocesar: amazon|twitter|twitter_3\n")
    parser.add_argument("--dims",
                        type=int,
                        default=2000,
                        help="N de dimensiones que se van a conservar.")

    args = parser.parse_args()

    main(args.dataset, args.dims)
