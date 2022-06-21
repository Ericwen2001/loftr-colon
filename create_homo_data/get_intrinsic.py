import numpy as np
import os
path = 'C:/Users/PanTianbo/Desktop/val_dataset'

list = os.listdir(path)
print(list)

np.savez('C:/Users/PanTianbo/Desktop/Val_homo.npz',list=list)
np.savez('C:/Users/PanTianbo/Desktop/val_intrinsics.npz',list=list)