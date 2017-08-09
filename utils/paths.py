raw_path = 'raw_data'
data_path = 'data'
scores_path = 'scores'
models_path = 'models'

datasets = ['amazon', 'twitter']

pruebas = ['baseline', 'msda', 'gfk', 'pca', 'sda']

dataframe_columns = ['Adaptacion',
                     'Tarea',
                     'Fuente',
                     'Objetivo',
                     'Baseline error', 
                     'Transfer error', 
                     'Transfer loss']


raw_folders = {
    'amazon': 'multi-domain/processed_acl',
    'twitter': 'twitter',
    'twitter_3': 'twitter_3_etiquetas',
}

dimensions = {
 'amazon': 3000,
 'twitter': 2000
}


#amazon_path = 'multi-domain/processed_acl'
#twitter_3_path = 'twitter_3_etiquetas/'
#twitter_path = 'twitter'
