# Memoria: Técnicas de Adaptación entre dominios para aprendizaje automático
================================================================


### Autor: Francisco :european_castle: @franciscoprimero

#### Técnicas implementadas:

* PCA
* SDA
* mSDA
* GFK

#### Requerimientos:

Para poder ejecutar las pruebas correctamente se requiere instalar [Anaconda](https://www.continuum.io/downloads), y luego una serie de paquetes mediante un entorno virtual.

Al tener Anaconda instalado, se require ejecutar los siguientes comandos:

```
$ conda env create -f env_memoria.yml
$ python -m nltk.downloader stopwords
```

#### Para iniciar el entorno virtual y ejecutar las pruebas:

```
$ source activate env_memoria
(env_memoria) $ jupyter notebook
```

#### Archivos de Jupyter Notebook:

Para ejecutar el preprocesamiento de los datasets, las pruebas de adaptación y la creación de gráficos es necesario utilizar Notebooks de Python, los cuales contienen todos los comandos para esto.
