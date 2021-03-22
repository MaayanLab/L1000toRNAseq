import argparse
import os
import numpy as np
import itertools
import datetime
import time
import pandas as pd
import json

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torch.autograd import Variable

from models_transcript import *
from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys

import wandb
parser = argparse.ArgumentParser()
parser.add_argument("--epoch_resume", type=int, default=0,
                    help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200,
                    help="number of epochs of training")
parser.add_argument("--dataset_nameA", type=str,
                    default="L1000", help="name of the dataset")
parser.add_argument("--dataset_nameB", type=str,
                    default="L1000", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--weight_decay", type=float, default=0.00001,
                    help="l2 regularization")


parser.add_argument("--decay_epoch", type=int, default=100,
                    help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--input_dimA", type=int, default=10,
                    help="size of input dimensions")
parser.add_argument("--hidden_dimA", type=int,default=10,
                    help="size of intermediate layer")
parser.add_argument("--output_dimA", type=int,default=10,
                    help="size of latent space")
parser.add_argument("--input_dimB", type=int, default=10,
                    help="size of input dimensions")
parser.add_argument("--hidden_dimB", type=int,default=10,
                    help="size of intermediate layer")
parser.add_argument("--output_dimB", type=int,default=10,
                    help="size of latent space")


parser.add_argument("--num_samples", type=int,default=10,
                    help="number of samples in input dataset")
parser.add_argument("--sample_interval", type=int, default=100,
                    help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=10,
                    help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=1,
                    help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float,
                    default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0,
                    help="identity loss weight")
parser.add_argument("--load_model_index", type=int, default=100,
                    help="load_model_index")
parser.add_argument("--eval_dataset_nameA", type=str, default="GTEx",
                    help="GTEx")
parser.add_argument("--eval_dataset_nameB", type=str, default="GTEx",
                    help="GTEx")
parser.add_argument("--exp_index", type=int, default=0,
                    help="index of experiment")                    
parser.add_argument('--ispredicting',  default=False, help='istest', action='store_true')                  
parser.add_argument("--cell_line", type=str, help="cell_line")  
parser.add_argument("--gamma", type=float, default=0.1,
                    help="step lr ratio")
parser.add_argument('--shuffle',  default=False, help='shuffle', action='store_true') 
parser.add_argument('--evaluation',  default=False, help='evaluation', action='store_true')         


parser.add_argument("--y_true_output_filename", type=str, help="y_true.txt")
parser.add_argument("--y_pred_output_filename", type=str, help="y_pred.txt")  
parser.add_argument("--prediction_folder", type=str, default="./", help="prediction_folder")
                
opt = parser.parse_args()

dataset_dict = {
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

    "L1000": "../data/processed/L1000/L1000_filtered_GSE92742_Broad_LINCS_Level3_INF_mlr12k_n{}x{}.f",
    "ARCHS4": "../data/processed/ARCHS4/human_matrix_v9_filtered_n50000x962_v2.f",
    "ARCHS4_full": "../data/processed/ARCHS4/human_matrix_v9_filtered_n{}x{}_step2_output.f",
    "GTEx_L1000": "../data/processed/GTEx/GSE92743_Broad_GTEx_L1000_Level3_Q2NORM_filtered_n2929x962_v2.f",
    "GTEx_RNAseq_landmark": "../data/processed/GTEx/GSE92743_Broad_GTEx_RNAseq_Log2RPKM_q2norm_filtered_n2929x962_v2.f",
}

# Create sample and checkpoint directories
model_folder = f"saved_models/{opt.exp_index}/"
log_folder = f"../output/{opt.exp_index}/logs/"
# prediction_folder = f"../output/{opt.exp_index}/prediction/"
prediction_folder = opt.prediction_folder
visualization_folder = f"../output/{opt.exp_index}/viz/"
if opt.ispredicting == False:
    wandb.init(project="L1000toRNAseq_step1")
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
        if "input_dimA" in saved_opt:
            opt.input_dimA = saved_opt["input_dimA"]
            opt.input_dimB = saved_opt["input_dimB"]

            opt.hidden_dimA = saved_opt["hidden_dimA"]
            opt.hidden_dimB = saved_opt["hidden_dimB"]
            
            opt.output_dimA = saved_opt["output_dimA"]
            opt.output_dimB = saved_opt["output_dimB"]
        else:
            opt.input_dimA = saved_opt["input_dim"]
            opt.input_dimB = saved_opt["input_dim"]

            opt.hidden_dimA = saved_opt["hidden_dim"]
            opt.hidden_dimB = saved_opt["hidden_dim"]
            
            opt.output_dimA = saved_opt["output_dim"]
            opt.output_dimB = saved_opt["output_dim"]
        opt.num_samples = saved_opt["num_samples"]

print(opt)


def main():
    
    # Losses
    criterion_GAN = torch.nn.MSELoss()
    # criterion_cycle = torch.nn.L1Loss()
    # criterion_identity = torch.nn.L1Loss()
    criterion_cycle = torch.nn.MSELoss()
    criterion_identity = torch.nn.MSELoss()

    cuda = torch.cuda.is_available()
    print("Cuda available", cuda)
    input_shapeA = (opt.num_samples, opt.input_dimA)
    input_shapeB = (opt.num_samples, opt.input_dimB)
    intermediate_dimA = opt.hidden_dimA
    intermediate_dimB = opt.hidden_dimB
    output_dimA = opt.output_dimA
    output_dimB = opt.output_dimB

    # Initialize generator and discriminator
    G_AB = GeneratorResNet(input_shapeA, intermediate_dimA, output_dimA, input_shapeB[1], opt.n_residual_blocks)
    G_BA = GeneratorResNet(input_shapeB, intermediate_dimB, output_dimB, input_shapeA[1], opt.n_residual_blocks)
    D_A = Discriminator(input_shapeA, intermediate_dimA, output_dimA)
    D_B = Discriminator(input_shapeB, intermediate_dimB, output_dimB)

    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    if opt.epoch_resume != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load(model_folder+"G_AB_%d.pth" % (opt.epoch_resume)))
        G_BA.load_state_dict(torch.load(model_folder+"G_BA_%d.pth" % (opt.epoch_resume)))
        D_A.load_state_dict(torch.load(model_folder+"D_A_%d.pth" % (opt.epoch_resume)))
        D_B.load_state_dict(torch.load(model_folder+"D_B_%d.pth" % (opt.epoch_resume)))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay
    )
    optimizer_D_A = torch.optim.Adam(
        D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
    optimizer_D_B = torch.optim.Adam(
        D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

    # Learning rate update schedulers
    # lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer_G, lr_lambda=LambdaLR(
    #         opt.n_epochs, opt.epoch_resume, opt.decay_epoch).step
    # )
    # lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer_D_A, lr_lambda=LambdaLR(
    #         opt.n_epochs, opt.epoch_resume, opt.decay_epoch).step
    # )
    # lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer_D_B, lr_lambda=LambdaLR(
    #         opt.n_epochs, opt.epoch_resume, opt.decay_epoch).step
    # )
    lr_scheduler_G = torch.optim.lr_scheduler.StepLR(
        optimizer_G, step_size=opt.decay_epoch, gamma=opt.gamma
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.StepLR(
        optimizer_D_A, step_size=opt.decay_epoch, gamma=opt.gamma
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.StepLR(
        optimizer_D_B, step_size=opt.decay_epoch, gamma=opt.gamma
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Training data loader
    if opt.ispredicting == False:

        if opt.dataset_nameA == "pseudomonas":
            data_fileA = os.path.join("../data/%s" % opt.dataset_nameA +
                                 "/train/A/all-pseudomonas-gene-normalized.zip")
            data_fileB = os.path.join("../data/%s" % opt.dataset_nameB +
                                 "/train/B/all-pseudomonas-gene-normalized.zip")
            l1000 = pd.read_table(data_fileA, index_col=0, header=0).T

            rnaseq = pd.read_table(data_fileB, index_col=0, header=0).T
        else:
            
            data_fileA = dataset_dict[opt.dataset_nameA].format(opt.num_samples, opt.input_dimA)
            data_fileB = dataset_dict[opt.dataset_nameB].format(opt.num_samples, opt.input_dimB)

            print(data_fileA)
            print(data_fileB)

            l1000 = pd.read_feather(data_fileA)
            first_col = l1000.columns.tolist()[0]
            l1000.set_index(first_col, inplace=True)
            

            rnaseq = pd.read_feather(data_fileB)
            first_col = rnaseq.columns.tolist()[0]
            rnaseq.set_index(first_col, inplace=True)

        # sort by column names
        l1000 = l1000.sort_index(axis=1)
        rnaseq = rnaseq.sort_index(axis=1)
        print(l1000)
        print(rnaseq)
        l1000_tensor = torch.FloatTensor(l1000.values)
        rnaseq_tensor = torch.FloatTensor(rnaseq.values)
        print(rnaseq.shape)


        dataloaderA = DataLoader(
            TensorDataset(l1000_tensor),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
        )
        dataloaderB = DataLoader(
            TensorDataset(rnaseq_tensor),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
        )

        # ----------
        #  Training
        # ----------


        loss_G_perEpoch = list()
        loss_D_perEpoch = list()

        prev_time = time.time()

        for epoch in range(opt.epoch_resume, opt.n_epochs):
            for i, (batchA, batchB) in enumerate(zip(dataloaderA, dataloaderB)):

                # Set model input
                if cuda:
                    real_A = batchA[0].cuda()  # number of samples in category A
                    real_B = batchB[0].cuda()
                else:
                    real_A = batchA[0]
                    real_B = batchB[0]

                # Adversarial ground truths
                validA = Variable(
                    Tensor(np.ones((real_A.size(0), 1))), requires_grad=False)
                fakeA = Variable(
                    Tensor(np.zeros((real_A.size(0), 1))), requires_grad=False)

                validB = Variable(
                    Tensor(np.ones((real_B.size(0), 1))), requires_grad=False)
                fakeB = Variable(
                    Tensor(np.zeros((real_B.size(0), 1))), requires_grad=False)

                # ------------------
                #  Train Generators
                # ------------------

                G_AB.train()
                G_BA.train()

                optimizer_G.zero_grad()
                
                # Identity loss
                loss_id_A = criterion_identity(G_BA(real_A), real_A)
                loss_id_B = criterion_identity(G_AB(real_B), real_B)

                loss_identity = (loss_id_A + loss_id_B) / 2
                
                # GAN loss                
                fake_B = G_AB(real_A)
                loss_GAN_AB = criterion_GAN(D_B(fake_B), validA)
                fake_A = G_BA(real_B)
                loss_GAN_BA = criterion_GAN(D_A(fake_A), validA)

                loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

                # Cycle loss
                recov_A = G_BA(fake_B)
                loss_cycle_A = criterion_cycle(recov_A, real_A)
                recov_B = G_AB(fake_A)
                loss_cycle_B = criterion_cycle(recov_B, real_B)

                loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

                # Total loss
                loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

                loss_G.backward()
                optimizer_G.step()

                # -----------------------
                #  Train Discriminator A
                # -----------------------

                optimizer_D_A.zero_grad()

                # Real loss
                loss_real = criterion_GAN(D_A(real_A), validA)
                # Fake loss (on batch of previously generated samples)
                fake_A_ = fake_A_buffer.push_and_pop(fake_A)
                loss_fake = criterion_GAN(D_A(fake_A_.detach()), fakeA)
                # Total loss
                loss_D_A = (loss_real + loss_fake) / 2

                loss_D_A.backward()
                optimizer_D_A.step()

                # -----------------------
                #  Train Discriminator B
                # -----------------------

                optimizer_D_B.zero_grad()

                # Real loss
                loss_real = criterion_GAN(D_B(real_B), validB)

                # Fake loss (on batch of previously generated samples)
                fake_B_ = fake_B_buffer.push_and_pop(fake_B)
                loss_fake = criterion_GAN(D_B(fake_B_.detach()), fakeB)

                # Total loss
                loss_D_B = (loss_real + loss_fake) / 2

                loss_D_B.backward()
                optimizer_D_B.step()

                loss_D = (loss_D_A + loss_D_B) / 2

                # --------------
                #  Log Progress
                # --------------

                # Determine approximate time left
                batches_done = epoch * len(dataloaderA) + i
                batches_left = opt.n_epochs * len(dataloaderA) - batches_done
                time_left = datetime.timedelta(
                    seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                    % (
                        epoch+1,
                        opt.n_epochs,
                        i,
                        len(dataloaderA),
                        loss_D.item(),
                        loss_G.item(),
                        loss_GAN.item(),
                        loss_cycle.item(),
                        loss_identity.item(),
                        time_left,
                    )
                )
            wandb.log({"D loss": loss_D.item(), "G Loss": loss_G.item(), "GAN loss": loss_GAN.item()})

            # Save loss_G and loss_D per epoch
            loss_G_perEpoch.append(loss_G.item())
            loss_D_perEpoch.append(loss_D.item())
            

            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

            if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == opt.checkpoint_interval-1:
                # Save model checkpoints
                torch.save(G_AB.state_dict(), model_folder+"G_AB_%d.pth" %
                        (epoch))
                torch.save(G_BA.state_dict(), model_folder+"G_BA_%d.pth" %
                        (epoch))
                torch.save(D_A.state_dict(), model_folder+"D_A_%d.pth" %
                        (epoch))
                torch.save(D_B.state_dict(), model_folder+"D_B_%d.pth" %
                        (epoch))

                # benchmark evaluation
                

        # Write loss arrays to file
        G_loss_file = os.path.join(log_folder+"G_loss.txt")
        D_loss_file = os.path.join(log_folder+"D_loss.txt")
        with open(G_loss_file, "w") as f:
            f.write("\n".join(map(str, loss_G_perEpoch)))

        with open(D_loss_file, "w") as f:
            f.write("\n".join(map(str, loss_D_perEpoch)))


    else: #opt.ispredicting == True:
        # ----------
        #  Evaluating
        # ----------
        # load datasets
        if opt.evaluation == True:
            if opt.cell_line is not None and opt.cell_line != "None" and opt.cell_line != "":
                folder = f"../data/processed/{opt.eval_dataset_nameA}/"
                filenames = os.listdir(folder)
                for filename in filenames:
                    if "v2" in filename and opt.cell_line in filename:
                        if "L1000" in filename:
                            data_fileA = folder+filename
                        else:
                            data_fileB = folder+filename
            else:
                if opt.eval_dataset_nameA in dataset_dict:
                    data_fileA = dataset_dict[opt.eval_dataset_nameA]
                    data_fileB = dataset_dict[opt.eval_dataset_nameB]
                else:
                    data_fileA = opt.eval_dataset_nameA
                    data_fileB = opt.eval_dataset_nameB
            print(data_fileA)
            print(data_fileB)
            

            l1000 = pd.read_feather(data_fileA)
            first_col = l1000.columns.tolist()[0]
            l1000.set_index(first_col, inplace=True)
            l1000 = l1000.sort_index(axis=1)

            l1000_samplenames = l1000.index.tolist()
            l1000_featurenames = l1000.columns.tolist()


            rnaseq = pd.read_feather(data_fileB)
            first_col = rnaseq.columns.tolist()[0]
            rnaseq.set_index(first_col, inplace=True)
            rnaseq = rnaseq.sort_index(axis=1)
            
            l1000_tensor = torch.FloatTensor(l1000.values)
            rnaseq_tensor = torch.FloatTensor(rnaseq.values)
            if opt.shuffle == True:
                rnaseq_tensor=rnaseq_tensor[:,torch.randperm(rnaseq_tensor.size()[1])]

            dataloaderA = DataLoader(
                TensorDataset(l1000_tensor),
                batch_size=opt.batch_size,
                num_workers=opt.n_cpu,
            )
            dataloaderB = DataLoader(
                TensorDataset(rnaseq_tensor),
                batch_size=opt.batch_size,
                num_workers=opt.n_cpu,
            )

            # load model
            G_AB.load_state_dict(torch.load(model_folder+"G_AB_%d.pth" %
                            (opt.load_model_index)))
            # G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" %
            #                 (opt.dataset_nameA, opt.load_model_index)))
            # D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" %
            #                 (opt.dataset_nameA, opt.load_model_index)))
            # D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" %
            #                 (opt.dataset_nameA, opt.load_model_index)))
            G_AB.eval()



            real_A_total = list()
            fake_B_total = list() # prediction of A
            real_B_total = list()
            prev_time = time.time()

            for i, (batchA, batchB) in enumerate(zip(dataloaderA, dataloaderB)):
                # Set model input
                if cuda:
                    real_A = batchA[0].cuda()  # number of samples in category A
                    real_B = batchB[0].cuda()
                else:
                    real_A = batchA[0]
                    real_B = batchB[0]
        
                fake_B = G_AB(real_A)
                fake_B = torch.reshape(fake_B, (real_B.shape[0], real_B.shape[1]))
                
                real_A_total.extend(real_A.cpu().detach().numpy()) 
                fake_B_total.extend(fake_B.cpu().detach().numpy()) 
                real_B_total.extend(real_B.cpu().detach().numpy())
            scores = get_scores(real_B_total, fake_B_total)
            # scores = get_scores(normalization(pd.DataFrame(real_B_total).T, z_normalization=True).T.values, normalization(pd.DataFrame(fake_B_total).T, z_normalization=True).T.values)
            

            print("/".join(scores.keys()))
            # print("\t".join(map(str, [round(item, 4) for item in scores.values()])))
            print("\t".join(map(str, [item for item in scores.values()])))

            real_B_total_df = pd.DataFrame(real_B_total, index=l1000_samplenames, columns=l1000_featurenames)
            fake_B_total_df = pd.DataFrame(fake_B_total, index=l1000_samplenames, columns=l1000_featurenames)
            real_A_total_df = pd.DataFrame(real_A_total, index=l1000_samplenames, columns=l1000_featurenames)

            # save prediction results
            save_df(real_B_total_df, filename=prediction_folder+opt.y_true_output_filename, shuffle=opt.shuffle)
            save_df(fake_B_total_df, filename=prediction_folder+opt.y_pred_output_filename, shuffle=opt.shuffle)
            save_df(real_A_total_df, filename=prediction_folder+"y_input.txt", shuffle=opt.shuffle)


        else: # evaluation == False
            if opt.eval_dataset_nameA in dataset_dict:
                data_fileA = dataset_dict[opt.eval_dataset_nameA]
            else:
                data_fileA = opt.eval_dataset_nameA

            print(data_fileA)
            

            l1000 = pd.read_feather(data_fileA) # sample x feature
            first_col = l1000.columns.tolist()[0]
            l1000.set_index(first_col, inplace=True)
            l1000 = l1000.sort_index(axis=1)
            l1000_samplenames = l1000.index.tolist()
            l1000_featurenames = l1000.columns.tolist()
            print("input")
            print(l1000)
            
            l1000_tensor = torch.FloatTensor(l1000.values)
            
            dataloaderA = DataLoader(
                TensorDataset(l1000_tensor),
                batch_size=opt.batch_size,
                num_workers=opt.n_cpu,
            )
            
            # load model
            G_AB.load_state_dict(torch.load(model_folder+"G_AB_%d.pth" %
                            (opt.load_model_index)))
            G_AB.eval()

            real_A_total = list()
            fake_B_total = list() # prediction of A
            
            for i, (batchA) in enumerate(dataloaderA):
                # Set model input
                if cuda:
                    real_A = batchA[0].cuda()  # number of samples in category A
                else:
                    real_A = batchA[0]
        
                fake_B = G_AB(real_A)
                fake_B = torch.reshape(fake_B, (real_A.shape[0], real_A.shape[1]))
                
                real_A_total.extend(real_A.cpu().detach().numpy()) 
                fake_B_total.extend(fake_B.cpu().detach().numpy()) 

            fake_B_total_df = pd.DataFrame(fake_B_total, index=l1000_samplenames, columns=l1000_featurenames)
            print("output")
            print(fake_B_total_df)
            save_df(fake_B_total_df, filename=prediction_folder+opt.y_pred_output_filename, shuffle=opt.shuffle)

if __name__ == "__main__":
    main()