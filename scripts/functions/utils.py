import random
from torch.autograd import Variable
import torch
import numpy as np
import pandas as pd

# evaluation
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error, matthews_corrcoef, accuracy_score, roc_auc_score

from maayanlab_bioinformatics.dge.characteristic_direction import characteristic_direction

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

# feature x sample matrix
import scipy.stats as ss
import warnings
from maayanlab_bioinformatics.normalization import quantile_normalize

def CPM(data):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = (data/data.sum())*10**6
        data = data.fillna(0)
        
    return data
def logCPM(data):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = (data/data.sum())*10**6
        data = data.fillna(0)
        data = np.log10(data+1)

    # Return
    return data
def log(data):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = data.fillna(0)
        data = np.log10(data+1)

    return data
def qnormalization(data):

    X_quantile_norm = quantile_normalize(data)
    return X_quantile_norm  

def normalization(data, logCPM_normalization=False, log_normalization=False, z_normalization=False, q_normalization=False):
    if logCPM_normalization == True:  
        data = logCPM(data)
        
    if log_normalization == True:   
        data = log(data)
        
    if z_normalization == True: 
        data = data.apply(ss.zscore, axis=0).dropna()
    if q_normalization == True:
        data = qnormalization(data)
    return data



def save(data, filename = "./y_true.txt", shuffle=False):
    if shuffle == True:
        filename = "shuffle_"+filename
    if filename.endswith(".txt"):
        print("Saving in txt...")
        with open(filename, "w") as f1:
            for i in range(len(data)):
                f1.write("\t".join(map(str, data[i])))
                f1.write("\n")
    elif filename.endswith(".f"):
        print("Saving in feather...")
        data.reset_index().to_feather(filename)
    print("Saved!", filename)

def save_df(data, filename = "./y_true.txt", shuffle=False):
    if shuffle == True:
        filename = "shuffle_"+filename
    if filename.endswith(".txt"):
        print("Saving in txt...")
        data.to_csv(filename, sep="\t")
    elif filename.endswith(".f"):
        print("Saving in feather...")
        data.reset_index().to_feather(filename)
    print(data.head())
    print("Saved!", filename)

def load_feather(filename):
    df = pd.read_feather(filename)
    first_col = df.columns.tolist()[0]
    df = df.set_index(first_col)
    
    # sort dataframe by column names
    df = df.sort_index(axis = 1) 
    return df
def load_csv(filename, sep):
    df = pd.read_csv(filename, sep=sep, index_col=0)    
    # sort dataframe by column names
    df = df.sort_index(axis = 1) 
    return df
from scipy.spatial import distance

def jaccard(a, b):
    if len(set(a).union(set(b))) > 0:
        return (0.0 + len(set(a).intersection(set(b)))) / len(set(a).union(set(b)))
    else:
        return 0
    # return 1-distance.jaccard(a, b) 

def precision(true, pred):
    if len(set(pred)) > 0:
        return (0.0 + len(set(true).intersection(set(pred)))) / len(set(pred))
    else:
        return 0

def recall(true, pred):

    if len(set(true)) > 0:
        return (0.0 + len(set(true).intersection(set(pred)))) / len(set(true))
    else:
        return 0
def f1score(true, pred):
    prec = precision(true, pred)
    recl = recall(true, pred)

    if prec > 0 or recl > 0:
        return (2 * prec * recl )/(prec+recl)
    else:
        return 0


def get_scores(y_true, y_pred):

    y_true_flatten = np.array(y_true).ravel()
    y_pred_prob_flatten = np.array(y_pred).ravel()
    scores_dict = dict()

    scores_dict["r2"] = r2_score(y_true, y_pred)
    scores_dict["rmse"] = mean_squared_error(y_true, y_pred, squared=True)
    scores_dict["pearson"] = pearsonr(y_true_flatten, y_pred_prob_flatten)[0]
    scores_dict["spearmanr"] = spearmanr(y_true_flatten, y_pred_prob_flatten)[0]

    return scores_dict

def get_average(ls):
    return sum(ls)/len(ls)



