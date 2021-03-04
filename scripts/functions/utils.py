import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np

from torchvision.utils import save_image


# evaluation
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error, matthews_corrcoef, accuracy_score, roc_auc_score


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



def save(data, folder="./", filename = "y_true.txt", shuffle=False):
    if shuffle == True:
        filename = "shuffle_"+filename

    with open(folder+filename, "w") as f1:
        for i in range(len(data)):
            f1.write("\t".join(map(str, data[i])))
            f1.write("\n")
    print("Saved!", folder+filename)
        
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

    # scores_dict["r2"] = list()
    # scores_dict["rmse"] = list()
    # scores_dict["pearson"] = list()
    # scores_dict["spearmanr"] = list()
    # scores_dict["jaccard_up"] = list()
    # scores_dict["jaccard_down"] = list()
    # scores_dict["precision_up"] = list()
    # scores_dict["precision_down"] = list()
    # scores_dict["recall_up"] = list()
    # scores_dict["recall_down"] = list()
    # scores_dict["f1_up"] = list()
    # scores_dict["f1_down"] = list()


    # for i in range(len(y_true)):
    #     print(i)
    #     tmp_y_true = y_true[i]
    #     tmp_y_pred = y_pred[i]
        # scores_dict["r2"].append(r2_score(tmp_y_true, tmp_y_pred))
        # scores_dict["rmse"].append(mean_squared_error(tmp_y_true, tmp_y_pred, squared=False))
        # scores_dict["pearson"].append(pearsonr(tmp_y_true, tmp_y_pred)[0])
        # scores_dict["spearmanr"].append(spearmanr(tmp_y_true, tmp_y_pred)[0])
   
    # temporary commented out
    # true_zscore_all = normalization(pd.DataFrame(y_true), z_normalization=True).values
    # pred_zscore_all = normalization(pd.DataFrame(y_pred), z_normalization=True).values
    
    # for i in range(len(true_zscore_all)):
    
    #     true_zscore = true_zscore_all[i]
    #     pred_zscore = pred_zscore_all[i]

    #     true_zscore_up = [i for i, x in enumerate(true_zscore) if x > 2]
    #     true_zscore_down = [i for i, x in enumerate(true_zscore) if x < -2]

    #     pred_zscore_up = [i for i, x in enumerate(pred_zscore) if x > 2]
    #     pred_zscore_down = [i for i, x in enumerate(pred_zscore) if x > -2]
    #     scores_dict["jaccard_up"].append(jaccard(true_zscore_up, pred_zscore_up))
    #     scores_dict["jaccard_down"].append(jaccard(true_zscore_down, pred_zscore_down))

    #     scores_dict["precision_up"].append(precision(true_zscore_up, pred_zscore_up))
    #     scores_dict["precision_down"].append(precision(true_zscore_down, pred_zscore_down))

    #     scores_dict["recall_up"].append(recall(true_zscore_up, pred_zscore_up))
    #     scores_dict["recall_down"].append(recall(true_zscore_down, pred_zscore_down))

    #     scores_dict["f1_up"].append(f1score(true_zscore_up, pred_zscore_up))
    #     scores_dict["f1_down"].append(f1score(true_zscore_down, pred_zscore_down))



        # break
    # scores_dict["r2"] = get_average(scores_dict["r2"])
    # scores_dict["rmse"] = get_average(scores_dict["rmse"])
    scores_dict["r2"] = r2_score(y_true, y_pred)
    scores_dict["rmse"] = mean_squared_error(y_true, y_pred, squared=True)
    scores_dict["pearson"] = pearsonr(y_true_flatten, y_pred_prob_flatten)[0]
    scores_dict["spearmanr"] = spearmanr(y_true_flatten, y_pred_prob_flatten)[0]

    # scores_dict["pearson"] = get_average(scores_dict["pearson"])
    # scores_dict["spearmanr"] = get_average(scores_dict["spearmanr"])
    # scores_dict["jaccard_up"] = get_average(scores_dict["jaccard_up"])
    # scores_dict["jaccard_down"] = get_average(scores_dict["jaccard_down"])
    # scores_dict["precision_up"] = get_average(scores_dict["precision_up"])
    # scores_dict["precision_down"] = get_average(scores_dict["precision_down"])
    # scores_dict["recall_up"] = get_average(scores_dict["recall_up"])
    # scores_dict["recall_down"] = get_average(scores_dict["recall_down"])
    # scores_dict["f1_up"] = get_average(scores_dict["f1_up"])
    # scores_dict["f1_down"] = get_average(scores_dict["f1_down"])

    return scores_dict

def get_average(ls):
    return sum(ls)/len(ls)


