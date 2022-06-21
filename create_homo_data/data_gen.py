import os
from Process_data import savedata
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--data_root', type=str, default='/root/public/DJ/medical/syn_colon', help='root path for original data')
parser.add_argument(
    '--is_video', type=bool, default=False, help='is input format video')
# parser.add_argument(
#     '--save_dir', type=str, default='./', help='where to save the processed data')
parser.add_argument(
    '--link_dir', type=str, default='/root/code/LoFTR-MedicalData/data', help='where to link the processed data')
parser.add_argument(
    '--int_dir', type=str, default=None, help='where to save the intrinsics data')

if __name__ == '__main__':
    args = parser.parse_args()
    savedata(args.data_root)
    pass




