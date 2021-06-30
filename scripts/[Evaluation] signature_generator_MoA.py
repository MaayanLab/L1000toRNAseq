import pandas as pd
import os
import multiprocessing
import get_chdir_signatures
import re
import logging
import time
from maayanlab_bioinformatics.dge.characteristic_direction import characteristic_direction


# init logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('log.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# init Parameters
folder = "../data/LINCS_CFDE/GSE92742_full_batches"
pred_folder = "../data/LINCS_CFDE/L1000_GSE92742_prediction_results_step2_batch_divided_38_35"
gene_filename = folder+"/GSE92742_genes.txt"
inst_filename = "../data/L1000/GSE92742_Broad_LINCS_inst_info.txt"
pert_filename = "../data/L1000/GSE92742_Broad_LINCS_pert_info.txt"
l1000_folder = "../data/LINCS_CFDE/GSE92742_full_batches"
touchstone_filename = "../data/L1000/touchstone.txt"

pert_ids = list()
MoAs = [
#     'Zinc fingers, C2H2-type',
#  'CD molecules',
#  'Dopamine receptor antagonist',
#  'GPCR / Class A : Orphans',
#  'RING-type (C3HC4) zinc fingers',
#  'RNA binding motif (RRM) containing',
#  'EF-hand domain containing',
#  'Endogenous ligands',
#  'Adrenergic receptor antagonist',
#  'Basic helix-loop-helix proteins',
#  'Cyclooxygenase inhibitor',
#  'WD repeat domain containing',
#  'Acetylcholine receptor antagonist',
#  'Serotonin receptor antagonist',
#  'Glucocorticoid receptor agonist',
 'Adrenergic receptor agonist',
 'Mitochondrial respiratory chain complex / Complex I',
 'EGFR inhibitor',
 'basic leucine zipper proteins',
 'Zinc fingers, PHD-type',
 'Histamine receptor antagonist',
 'Ankyrin repeat domain containing',
 'Serotonin receptor agonist',
 'Glutamate receptor antagonist',
 'BTB/POZ domain containing',
 'SH2 domain containing',
#  'HDAC inhibitor',
 'Homeoboxes / ANTP class : HOXL subclass',
 'Phosphodiesterase inhibitor',
 'Bacterial cell wall synthesis inhibitor',
 'Calcium channel blocker',
 'RNA polymerase subunits',
 'Immunoglobulin superfamily / C1-set domain containing',
 'Dopamine receptor agonist',
 'ETS Transcription Factors',
 'Pleckstrin homology (PH) domain containing',
 'S ribosomal proteins',
 'Tumour necrosis factor (TNF) receptor family',
 'Proteasome (prosome, macropain) subunits',
 'Estrogen receptor agonist',
 'Interleukins and interleukin receptors',
#  'ATPase inhibitor',
 'Aldehyde dehydrogenases',
#  'CDK inhibitor',
 'Sodium channel blocker',
 'Adhesion Class GPCRs',
 'Forkhead boxes',
 'General transcription factors',
 'Ubiquitin-conjugating enzymes E2',
 'Histone deacetylases']

def get_signature(output_folder, inst_info, batchfile, batch_pert_id, gene_list, common_gene_list):

    if os.path.exists(f"{pred_folder}/{batchfile}") == False:
        return None
    
    batch = batchfile.replace(".f", "")
    sig_df = inst_info[inst_info['batch'] == batch]
    
    if os.path.exists(f"{output_folder}/predicted_rnaseq_signature_{batch}_{batch_pert_id}.csv") == True:
        return None

    if sig_df.shape[0] > 0 and batch_pert_id in sig_df["pert_id"].unique():
        batch_data_df = get_chdir_signatures.load_feather(f"{pred_folder}/{batchfile}", None)
        batch_data_df = batch_data_df[common_gene_list].T

        batch_data_df_l1000 = get_chdir_signatures.load_feather(f"{l1000_folder}/{batchfile}", gene_list)
        batch_data_df_l1000 = batch_data_df_l1000.loc[common_gene_list, :]

        
        logger.info(batch_pert_id+"in"+batchfile) 
        # single pert_id is case
        control_inst_id = [x for x in sig_df.loc[sig_df["pert_id"]!=batch_pert_id, "inst_id"].tolist() if x in batch_data_df.columns]
        case_inst_id = [x for x in sig_df.loc[sig_df["pert_id"]==batch_pert_id, "inst_id"].tolist() if x in batch_data_df.columns]
        # print(batch_data_df)
        # print(control_inst_id)
        # print([x for x in control_inst_id if x not in batch_data_df.columns])
        control_gex_data_df = batch_data_df.loc[:, control_inst_id]
        case_gex_data_df = batch_data_df.loc[:, case_inst_id]
        # print(control_gex_data_df.shape, case_gex_data_df.shape)
        signature = characteristic_direction(control_gex_data_df, case_gex_data_df, calculate_sig=False, sort=False)
        
        control_gex_data_df_l1000 = batch_data_df_l1000.loc[:, control_inst_id]
        case_gex_data_df_l1000 = batch_data_df_l1000.loc[:, case_inst_id]
        # print(control_gex_data_df_l1000.shape, case_gex_data_df_l1000.shape)
        signature_l1000 = characteristic_direction(control_gex_data_df_l1000, case_gex_data_df_l1000, calculate_sig=False, sort=False)

        # save signature_df 
        signature.to_csv(f"{output_folder}/predicted_rnaseq_signature_{batch}_{batch_pert_id}.csv")
        signature_l1000.to_csv(f"{output_folder}/l1000_signature_{batch}_{batch_pert_id}.csv")
        logger.info(f"Saved!{output_folder}/predicted_rnaseq_signature_{batch}_{batch_pert_id}.csv")
        logger.info(f"Saved!{output_folder}/l1000_signature_{batch}_{batch_pert_id}.csv")
            
    

def main():
    # load gene index
    with open(gene_filename, "r") as f:
        gene_list = [x.strip() for x in f.readlines()]
    # get common genes
    prediction_gene_filename = "../output_step2/35/logs/prediction_gene_list.txt"
    with open(prediction_gene_filename, "r") as f:
        prediction_gene_list = [x.strip() for x in f.readlines()]

    common_gene_list = list(set(gene_list).intersection(prediction_gene_list))


    # load inst info
    inst_info = pd.read_csv(inst_filename, sep="\t")
    # generate experiment column
    inst_info["batch"] = inst_info["rna_plate"].replace(to_replace='_X[0-9]', value='', regex=True)

    # load touchstone to get pert_ids of MoA
    touchstone = pd.read_csv(touchstone_filename, sep="\t")
    touchstone = touchstone.dropna()
    touchstone = touchstone[touchstone["MoA"]!="-666"]
    
    # merge touchstone info and inst_info
    # touchstone.columns = touchstone.columns.map({"ID":"pert_id"})
    touchstone_inst_info = pd.merge(inst_info, touchstone, left_on="pert_id", right_on="ID", how="left")
    touchstone_inst_info = touchstone_inst_info.dropna()
    
    for MoA in MoAs:
        # create output folder
        output_folder = f"../data/L1000_signatures/{MoA}/"
        try:
            os.mkdir(output_folder)
        except:
            pass
        print(f"Output folder: {output_folder}")

        moa_inst_info = touchstone_inst_info[touchstone_inst_info["MoA"]==MoA]
        batchlist = moa_inst_info["batch"].unique()
        i = 0
        for batch in batchlist:
            batch_pert_ids = moa_inst_info.loc[moa_inst_info["batch"]==batch, "pert_id"].unique().tolist()
            for batch_pert_id in batch_pert_ids:
                # print(MoA)
                print(i, "out of", moa_inst_info.shape[0], MoA, batch_pert_id+" in "+batch)
                get_signature(output_folder, inst_info, batch+".f", batch_pert_id, gene_list, common_gene_list)
                i += 1
            #     break
            # break
if __name__ == "__main__":
    main()