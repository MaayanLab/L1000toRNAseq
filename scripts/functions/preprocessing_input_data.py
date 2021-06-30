import argparse
import cmapPy.pandasGEXpress.parse_gctx as parse_gctx
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument('--input_filename',  default="../data/GTEx/GSE92743_Broad_GTEx_L1000_Level3_Q2NORM_n3176x12320.gctx", help='input_filename', type=str)          
parser.add_argument('--gene_info_filename',  default="../data/GTEx/GSE92743_Broad_GTEx_gene_info.txt", help='gene_info_filename', type=str)          
parser.add_argument('--output_filename',  default="../data/processed/GTEx/example.f", help='output_filename', type=str)        
parser.add_argument('--gene_names',  default="../data/processed/overlap_landmark_gene_file.txt", help='gene_names', type=str)        

opt = parser.parse_args()
def main():
    # load
    # load gene names
    with open(opt.gene_names, "r") as f:
        gene_names = [x.strip() for x in f.readlines()]
        gene_names = sorted(gene_names)
    
    print('Loading L1000 data.....', opt.input_filename)
    if opt.input_filename.endswith(".gctx"):
        # load gctx
        l1000_data = parse_gctx.parse(opt.input_filename, convert_neg_666=True).data_df
    elif opt.input_filename.endswith(".f"):
        l1000_data = pd.read_feather(opt.input_filename)
        first_col = l1000_data.columns.tolist()[0]
        l1000_data = l1000_data.set_index(first_col)

    print(l1000_data.head())
    print(l1000_data.shape)
    
    gene_info = pd.read_csv(opt.gene_info_filename, sep="\t")
    # create a probe_id to gene name dictionary 
    gene_dict = dict(zip([str(x) for x in gene_info['pr_gene_id']], gene_info['pr_gene_symbol']))

    # label rows with gene names 
    try:
        l1000_data.index = [gene_dict[x] for x in l1000_data.index.values]
    except:
        pass

    # select common lanemark genes
    l1000_data_subset = l1000_data.loc[ gene_names, :].T
    l1000_data_subset = l1000_data_subset.sort_index(axis=1)

    # save
    l1000_data_subset.reset_index().to_feather(opt.output_filename)
    print("Saved!", opt.output_filename)
if __name__ == "__main__":
    main()