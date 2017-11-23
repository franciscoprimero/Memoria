"""DatasetStorage."""
import cPickle as pickle
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset():
    """Dataset."""

    # Dataset
    # domains = [dominio_1, ... , dominio_m]
    # labeled = {
    #    dominio_1: {'X': X, 'y': y}
    #    ...
    #    dominio_m: {'X': X, 'y': y}
    # }
    # unlabeled = {
    #    dominio_1: {'X': X}
    #    ...
    #    dominio_m: {'X': X}
    # }

    def __init__(self, labeled=None, unlabeled=None, domains=None):
        """__init__."""
        self.domains = domains

        self.labeled = labeled
        self.unlabeled = unlabeled
        self.splitted = False

        if labeled is None and unlabeled is None:
            self.loaded = False
        else:
            self.loaded = True

    def save(self, path):
        """save."""
        pickle.dump(self, open(path, "wb"))

    def load(self, path):
        """load."""
        data = pickle.load(open(path, "rb"))
        return data

    def split_dataset(self, test_size=0.2, seed=42):
        """train_test_split"""

        if self.splitted:
            print "Dataset already splitted"
            return

        for v_l in self.labeled:
            X = self.labeled[v_l]['X']
            y = self.labeled[v_l]['y']
            X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=test_size, random_state=seed)

            self.labeled[v_l] = {
                'X_tr': X_tr,
                'X_ts': X_ts,
                'y_tr': y_tr,
                'y_ts': y_ts
            }

        self.splitted = True
        return

    def get_all_domain_X(self, domain, test_data=False):
        """get_all_domain_X.
           
        """
        v_l = self.labeled[domain]
        
        
        if self.splitted:
            X_l_tr = v_l['X_tr'].todense()
            
            if test_data:
                X_l_ts = v_l['X_ts'].todense()
                X_l = np.concatenate((X_l_tr, X_l_ts))
            else:
                X_l = X_l_tr
                
        else:
            X_l = v_l['X'].todense()

            
        if self.unlabeled is not None:
            v_ul = self.unlabeled[domain]
            X_ul = v_ul['X'].todense()

            return np.concatenate((X_l, X_ul))
        else:
            return X_l
    
    
    def get_all_X(self, test_data=False):
        instances = None

        if self.unlabeled is None:
            for v_l in self.labeled.values():
                if self.splitted:
                    X_l_tr = v_l['X_tr'].todense()
                    
                    
                    if test_data:
                        X_l_ts = v_l['X_ts'].todense()
                        X_l = np.concatenate((X_l_tr, X_l_ts))
                    else:
                        X_l = X_l_tr
                else:
                    X_l = v_l['X'].todense()

                if instances is None:
                    instances = X_l
                else:
                    instances = np.concatenate((instances, X_l))

        else:
            for v_l, v_ul in zip(self.labeled.values(), self.unlabeled.values()):
                if self.splitted:
                    X_l_tr = v_l['X_tr'].todense()
                    
                    if test_data:
                        X_l_ts = v_l['X_ts'].todense()
                        X_l = np.concatenate((X_l_tr, X_l_ts))
                    else:
                        X_l = X_l_tr
                else:
                    X_l = v_l['X'].todense()

                X_ul = v_ul['X'].todense()

                if instances is None:
                    instances = np.concatenate((X_l, X_ul))
                else:
                    instances = np.concatenate((instances, X_l, X_ul))

        return instances
