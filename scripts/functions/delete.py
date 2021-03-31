import os
import shutil
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp_index", type=int, default=0,
                    help="index of experiment")                    
parser.add_argument('--step2',  default=False, help='is step2', action='store_true')                  
                  
opt = parser.parse_args()

# os.rmdir()

if opt.step2 == True:
    try:
        shutil.rmtree(f"../output_step2/{opt.exp_index}")
        shutil.rmtree(f"./saved_models_step2/{opt.exp_index}")
    except:
        pass
else:
    try:
        shutil.rmtree(f"../output/{opt.exp_index}")
        shutil.rmtree(f"./saved_models/{opt.exp_index}")
    except:
        pass