import argparse
import os
import numpy as np
import pandas as pd
import json
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torch.autograd import Variable

from models_transcript import *
from utils import *
import torchnlp
import torch.nn as nn
import torch.nn.functional as F
import torch

from sklearn.model_selection import train_test_split

# evaluation
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error, matthews_corrcoef, accuracy_score, roc_auc_score

import wandb
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--epoch_resume", type=int, default=0,
                    help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200,
                    help="number of epochs of training")
parser.add_argument("--input_dataset_name", type=str,
                    default="L1000", help="name of the dataset")
parser.add_argument("--output_dataset_name", type=str,
                    default="L1000", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100,
                    help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--input_dim", type=int, default=10,
                    help="size of input dimensions")
parser.add_argument("--hidden_dim", type=int, metavar='N', nargs='+' , default=10,
                    help="size of intermediate layer")
parser.add_argument("--output_dim", type=int,default=10,
                    help="size of latent space")

parser.add_argument("--num_samples", type=int,default=10,
                    help="number of samples in input dataset")
parser.add_argument("--sample_interval", type=int, default=100,
                    help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=10,
                    help="interval between saving model checkpoints")

parser.add_argument("--eval_input_dataset_name", type=str, default="GTEx",
                    help="GTEx")
parser.add_argument("--eval_output_dataset_name", type=str, default="GTEx",
                    help="GTEx")

                
parser.add_argument("--eval_exp_index", type=int, default="8",
                    help="eval_exp_index")


parser.add_argument("--exp_index", type=int, default=0,
                    help="index of experiment")                    
parser.add_argument('--ispredicting',  default=False, help='predict or train?', action='store_true')  # predict or                 
parser.add_argument("--cell_line", type=str, help="cell_line")  
parser.add_argument("--gamma", type=float, default=0.5,
                    help="step lr ratio")
parser.add_argument('--shuffle',  default=False, help='shuffle', action='store_true')          
parser.add_argument("--valid_ratio", type=float, default=0.1,
                    help="validation set ratio")
parser.add_argument("--test_ratio", type=float, default=0.1,
                    help="test set ratio")
parser.add_argument("--y_true_output_filename", type=str, help="y_true.txt")
parser.add_argument("--y_pred_output_filename", type=str, help="y_pred.txt")  
parser.add_argument('--early_stopping',  default=False, help='early_stopping', action='store_true')          
parser.add_argument('--early_stopping_epoch',  default=3, help='early_stopping_epoch', type=int)        
parser.add_argument("--early_stopping_tol", type=float, default=0.1,
                    help="early_stopping_tol")  

parser.add_argument('--evaluation',  default=False, help='calculate scores or just predict?', action='store_true')        

parser.add_argument("--prediction_folder", type=str, default="./", help="prediction_folder")

parser.add_argument("--output_gene_list", type=str, default="../data/ARCHS4/high_count_gene_list.txt", help="output_gene_list")

opt = parser.parse_args()

dataset_dict = {

    "ARCHS4_50000_input": "../data/processed/ARCHS4/human_matrix_v9_filtered_n50000x962_v2.f",
    "ARCHS4_50000_output": "../data/processed/ARCHS4/human_matrix_v9_filtered_n50000x25312_v2.f",


    "L1000_MCF7": "../data/Evaluation/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n203x962_celllineMCF7.f",
    "ARCHS4_MCF7": "../data/Evaluation/ARCHS4_human_matrix_v9_n203x25312_celllineMCF7.f",
    "ARCHS4_MCF7_landmark": "../data/Evaluation/ARCHS4_human_matrix_v9_n203x962_celllineMCF7.f",

    "L1000_PC3": "../data/Evaluation/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n31x962_celllinePC3.f",
    "ARCHS4_PC3": "../data/Evaluation/ARCHS4_human_matrix_v9_n31x25312_celllinePC3.f",
    "ARCHS4_PC3_landmark": "../data/Evaluation/ARCHS4_human_matrix_v9_n31x962_celllinePC3.f",

    "L1000_A375": "../data/Evaluation/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n30x962_celllineA375.f",
    "ARCHS4_A375": "../data/Evaluation/ARCHS4_human_matrix_v9_n30x25312_celllineA375.f",
    "ARCHS4_A375_landmark": "../data/Evaluation/ARCHS4_human_matrix_v9_n30x962_celllineA375.f",
    
    "L1000_HEPG2": "../data/Evaluation/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n7x962_celllineHEPG2.f",
    "ARCHS4_HEPG2": "../data/Evaluation/ARCHS4_human_matrix_v9_n7x25312_celllineHEPG2.f",
    "ARCHS4_HEPG2_landmark": "../data/Evaluation/ARCHS4_human_matrix_v9_n7x962_celllineHEPG2.f",
    
    "L1000_VCAP": "../data/Evaluation/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n4x962_celllineVCAP.f",
    "ARCHS4_VCAP": "../data/Evaluation/ARCHS4_human_matrix_v9_n4x25312_celllineVCAP.f",
    "ARCHS4_VCAP_landmark": "../data/Evaluation/ARCHS4_human_matrix_v9_n4x962_celllineVCAP.f",

    "GTEx_L1000": "../data/processed/GTEx/GSE92743_Broad_GTEx_L1000_Level3_Q2NORM_filtered_n2929x962_v2.f",
    "GTEx_RNAseq_landmark": "../data/processed/GTEx/GSE92743_Broad_GTEx_RNAseq_Log2RPKM_q2norm_filtered_n2929x962_v2.f",
    "GTEx_RNAseq": "../data/processed/GTEx/GSE92743_Broad_GTEx_RNAseq_Log2RPKM_q2norm_filtered_n2929x25312_v2.f",
    "step1": "../output/{}/prediction/{}"

}

# Create sample and checkpoint directories
model_folder = f"saved_models_step2/{opt.exp_index}/"
log_folder = f"../output_step2/{opt.exp_index}/logs/"
# prediction_folder = f"../output_step2/{opt.exp_index}/prediction/"
prediction_folder = opt.prediction_folder
visualization_folder = f"../output_step2/{opt.exp_index}/viz/"
if opt.ispredicting == False:
    wandb.init(project="L1000toRNAseq_step2")
    os.makedirs(model_folder, exist_ok=False)
    os.makedirs(log_folder, exist_ok=False)
    os.makedirs(prediction_folder, exist_ok=False)
    os.makedirs(visualization_folder, exist_ok=False)

    with open(log_folder+"args.txt", "w") as f:
        # f.write(repr(opt))
        json.dump(opt.__dict__, f, indent=2)

else:
    
    with open(log_folder+"args.txt", "r") as f:
        saved_opt = json.load(f)
        print(saved_opt)
        
        opt.input_dim = saved_opt["input_dim"]
        opt.hidden_dim = saved_opt["hidden_dim"]
        opt.output_dim = saved_opt["output_dim"]
        opt.num_samples = saved_opt["num_samples"]

print(opt)
cuda = torch.cuda.is_available()

def data_split(input, label, n_samples, random_seed=42):

    X_train, X_valid, y_train, y_valid = train_test_split(input, label, test_size=n_samples, random_state=random_seed)
    return X_train, X_valid, y_train, y_valid

def main():
    
    # Losses
    criterion_E = torch.nn.MSELoss()
    # criterion_E = torch.nn.L1Loss()

    
    print("Cuda available", cuda)
    input_shape = (opt.num_samples, opt.input_dim)
    intermediate_dim = opt.hidden_dim
    output_dim = opt.output_dim

    # Initialize generator and discriminator
    E = Extrapolator(input_shape, intermediate_dim, output_dim)
    # Log metrics with wandb
    if opt.ispredicting == False:
        wandb.watch(E)
    if cuda:
        E = E.cuda()
        criterion_E.cuda()

    if opt.epoch_resume != 0:
        # Load pretrained models
        E.load_state_dict(torch.load(model_folder+"E_%d.pth" % (opt.epoch_resume)))
        
    else:
        # Initialize weights
        E.apply(weights_init_normal)

    # Optimizers
    optimizer_E = torch.optim.Adam(
        E.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    lr_scheduler_E = torch.optim.lr_scheduler.StepLR(
        optimizer_E, step_size=opt.decay_epoch, gamma=opt.gamma
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Training data loader
    if opt.ispredicting == False:

        input_data_file = dataset_dict[opt.input_dataset_name]
        output_data_file = dataset_dict[opt.output_dataset_name]
        
        input_data = pd.read_feather(input_data_file)
        first_col = input_data.columns.tolist()[0]
        input_data.set_index(first_col, inplace=True)

        output_data = pd.read_feather(output_data_file)
        first_col = output_data.columns.tolist()[0]
        output_data.set_index(first_col, inplace=True)
        
        print(input_data_file)
        print(input_data.shape)
        print(input_data)
        print(output_data_file)
        print(output_data.shape)
        print(output_data)
        # data split
        input_data_train, input_data_test, output_data_train, output_data_test = data_split(input_data, output_data, n_samples = int(input_data.shape[0]*opt.test_ratio))
        input_data_train, input_data_valid, output_data_train, output_data_valid = data_split(input_data_train, output_data_train, n_samples = int(input_data.shape[0]*opt.valid_ratio))
        
        input_tensor_train = torch.FloatTensor(input_data_train.values)
        output_tensor_train = torch.FloatTensor(output_data_train.values)


        loss_E_perEpoch = list()

        current_best_valid_loss = np.inf

        for epoch in range(opt.epoch_resume, opt.n_epochs):
            print("---------------------------")
            train_loss = train_step(E, criterion_E, optimizer_E, input_tensor_train, output_tensor_train)
            print("[Epoch %d/%d] Train loss: %f" % (
                    epoch+1,
                    opt.n_epochs,
                    train_loss,
                ))     

            
            # Save loss_E per epoch
            loss_E_perEpoch.append(train_loss)

            # ------------------
            #  Eval Validation set
            # ------------------
            for valid_mode in ["valid", "test"]:
                if valid_mode == "valid":
                    input_tensor_test = torch.FloatTensor(input_data_valid.values)
                    output_tensor_test = torch.FloatTensor(output_data_valid.values)
                elif valid_mode == "test":
                    input_tensor_test = torch.FloatTensor(input_data_test.values)
                    output_tensor_test = torch.FloatTensor(output_data_test.values)
                
                val_labels, val_pred, scores_dict = eval_step(E, criterion_E, input_tensor_test, output_tensor_test, valid_mode)
                print("[Epoch %d/%d] %s loss: %f" % (
                    epoch+1,
                    opt.n_epochs,
                    valid_mode,
                    scores_dict["val_loss"],
                ))     

                print("/".join(scores_dict.keys()))
                print("\t".join(map(str, [item for item in scores_dict.values()])))
                
                if valid_mode == "valid":
                    if scores_dict["val_loss"]+opt.early_stopping_tol < current_best_valid_loss:
                        print("Current Best!!")
                        current_best_valid_loss = scores_dict["val_loss"]                        
                        early_stopping_count = 0
                        # Save model checkpoints
                        torch.save(E.state_dict(), model_folder+"E.pth")
                        save(val_pred, filename=prediction_folder+opt.y_pred_output_filename, shuffle=opt.shuffle)
                        save(val_labels, filename=prediction_folder+opt.y_true_output_filename, shuffle=opt.shuffle)
                        
                    else:
                        early_stopping_count += 1
                        print(f"{early_stopping_count} out of {opt.early_stopping_epoch}")
                    
                if valid_mode == "test":
                    wandb.log({"Test Pearson": scores_dict["pearson"], "Test Loss": scores_dict["val_loss"]})
            if opt.early_stopping==True and early_stopping_count == opt.early_stopping_epoch:
                break
            
            # Update learning rates
            lr_scheduler_E.step()
            
                


        # Write loss arrays to file
        E_loss_file = os.path.join(log_folder+"E_loss.txt")
        with open(E_loss_file, "w") as f:
            f.write("\n".join(map(str, loss_E_perEpoch)))


    else: 
        # ----------
        #  Predicting
        # ----------
        # load datasets

        if opt.evaluation == False:

            if opt.eval_input_dataset_name in dataset_dict:
                data_fileA = dataset_dict[opt.eval_input_dataset_name]
            else:
                data_fileA = opt.eval_input_dataset_name

            print(data_fileA)

            if data_fileA.endswith(".f"):
                input = pd.read_feather(data_fileA)
                first_col = input.columns.tolist()[0]
                input.set_index(first_col, inplace=True)
            else:
                input = pd.read_csv(data_fileA, sep="\t", index_col=0)
            print(input)
            input_tensor = torch.FloatTensor(input.values)
            
            dataloaderA = DataLoader(
                TensorDataset(input_tensor),
                batch_size=opt.batch_size,
                num_workers=opt.n_cpu,
            )
            
            # load model
            E.load_state_dict(torch.load(model_folder+"E.pth"))
            E.eval()

            predictions = list()
            for i, (batchA) in enumerate(dataloaderA):
                # Set model input
                input_batch = batchA[0].cuda()  # number of samples in category A  
                pred = E(input_batch)                
                predictions.extend(pred.cpu().detach().numpy()) 
            
            # save prediction results
            with open(opt.output_gene_list, "r") as f:
                gene_list = [x.strip() for x in f.readlines()]
                # gene_list = sorted(gene_list)
            sample_list = input.index.tolist()
            prediction_df = pd.DataFrame(predictions, index=sample_list, columns=gene_list)

            # make negative values to zeros
            prediction_df[prediction_df<0]=0
            print("prediction")
            print(prediction_df)
            print(prediction_df.shape)
            # prediction_df = prediction_df.iloc[:5, :]
            save_df(prediction_df, filename=prediction_folder+opt.y_pred_output_filename, shuffle=opt.shuffle)


        else: ## evaluation == True
            if opt.eval_input_dataset_name in dataset_dict:
                data_fileA = dataset_dict[opt.eval_input_dataset_name]
            else:
                data_fileA = opt.eval_input_dataset_name
            
            if opt.eval_output_dataset_name in dataset_dict:
                data_fileB = dataset_dict[opt.eval_output_dataset_name]
            else:
                data_fileB = opt.eval_output_dataset_name
            print(data_fileA)
            print(data_fileB)

            output = pd.read_feather(data_fileB)
            first_col = output.columns.tolist()[0]
            output.set_index(first_col, inplace=True)
            # output = output.sort_index(axis=1)
            if data_fileA.endswith(".f"):
                input = pd.read_feather(data_fileA)
                first_col = input.columns.tolist()[0]
                input.set_index(first_col, inplace=True)
            else:
                input = pd.read_csv(data_fileA, sep="\t", index_col=0)
            print("input")
            print(input)
        
            input_tensor = torch.FloatTensor(input.values)
            output_tensor = torch.FloatTensor(output.values)
            if opt.shuffle == True:
                output_tensor=output_tensor[:,torch.randperm(output_tensor.size()[1])]

            dataloaderA = DataLoader(
                TensorDataset(input_tensor),
                batch_size=opt.batch_size,
                num_workers=opt.n_cpu,
            )
            dataloaderB = DataLoader(
                TensorDataset(output_tensor),
                batch_size=opt.batch_size,
                num_workers=opt.n_cpu,
            )

            # load model
            E.load_state_dict(torch.load(model_folder+"E.pth"))
            E.eval()

            y_true = list()
            predictions = list()
            for i, (batchA, batchB) in enumerate(zip(dataloaderA, dataloaderB)):
                # Set model input
                input_batch = batchA[0].cuda()  # number of samples in category A
                y_true_batch = batchB[0].cuda()
        
                pred = E(input_batch)
                
                y_true.extend(y_true_batch.cpu().detach().numpy()) 
                predictions.extend(pred.cpu().detach().numpy()) 
            scores = get_scores(y_true, predictions)

            print("/".join(scores.keys()))
            # print("\t".join(map(str, [round(item, 4) for item in scores.values()])))
            print("\t".join(map(str, [item for item in scores.values()])))
            print("prediction")
            # save prediction results
            with open(opt.output_gene_list, "r") as f:
                gene_list = [x.strip() for x in f.readlines()]
                # gene_list = sorted(gene_list)
            sample_list = input.index.tolist()
            prediction_df = pd.DataFrame(predictions, index=sample_list, columns=gene_list)
            print("prediction")
            print(prediction_df)

            print("answer")
            print(output)
            # save prediction results
            save(y_true, filename=prediction_folder+opt.y_true_output_filename, shuffle=opt.shuffle)
            save_df(prediction_df, filename=prediction_folder+opt.y_pred_output_filename, shuffle=opt.shuffle)


def train_step(model, criterion, optimizer, input_tensor_train, output_tensor_train):
    model.train()

    # ----------
    #  Training
    # ----------
    dataloaderA = DataLoader(
        TensorDataset(input_tensor_train),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )
    dataloaderB = DataLoader(
        TensorDataset(output_tensor_train),
        batch_size=opt.batch_size,
        num_workers=opt.n_cpu,
    )       

    tr_loss = 0.0
    print("Training...")
    for i, (batchA, batchB) in enumerate(zip(dataloaderA, dataloaderB)):

        # Set model input
        input = batchA[0].cuda()
        y_true = batchB[0].cuda()
        
        # ------------------
        #  Train FCNN
        # ------------------

        optimizer.zero_grad()
        pred = model(input)
        loss_E = criterion(pred, y_true)                
        loss_E.backward()
        optimizer.step()

        # Print log
        sys.stdout.write(f"\rProcess Training Batch: [{i}/{len(dataloaderA)}]")

        # print(f"Process Training Batch: [{i}/{len(dataloaderA)}]", end='\r')  
        # print("print?")
        tr_loss += loss_E.item()
    return tr_loss/len(dataloaderA)
    

def eval_step(model, criterion, input_tensor_test, output_tensor_test, valid_mode):
    
    model.eval()
    dataloaderA = DataLoader(
        TensorDataset(input_tensor_test),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )
    dataloaderB = DataLoader(
        TensorDataset(output_tensor_test),
        batch_size=opt.batch_size,
        num_workers=opt.n_cpu,
    )       
    val_loss = 0.0
    val_labels = list()
    val_pred = list()
    with torch.no_grad():    
        for i, (batchA, batchB) in enumerate(zip(dataloaderA, dataloaderB)):
            # Set model input
            input = batchA[0].cuda()
            y_true = batchB[0].cuda()

                    
            # Predicting
            predicted = model(input)
            val_loss += criterion(predicted, y_true).item()

            if cuda == True:
                predicted = predicted.cpu().numpy()
            else:
                predicted = predicted.numpy()
            
            val_labels.extend(y_true.cpu().numpy())
            val_pred.extend(predicted)
            

            print("Process {} Batch: [{}/{}]".format(valid_mode, i, len(dataloaderA)), end='\r')

    val_loss /= len(dataloaderA)

    scores_dict= get_scores(val_labels, val_pred)         
    scores_dict["val_loss"] = val_loss

    return val_labels, val_pred, scores_dict


if __name__ == "__main__":
    main()