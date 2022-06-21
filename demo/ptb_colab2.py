import torch
import cv2
import numpy as np
import matplotlib.cm as cm
import torch.cuda.profiler as profiler
import pyprof
import imageio
import random

from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg
#from torch.profiler import profile, record_function, ProfilerActivity

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()

img0_pth = "train_processed/00170.jpg/img.png"
img1_pth = "train_processed/00171.jpg/img.png"
image_pair = [img0_pth, img1_pth]
pyprof.init()
# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()
img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
#pdb.set_trace()
img0_raw = cv2.resize(img0_raw, (640, 480))
img1_raw = cv2.resize(img1_raw, (640, 480))

img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}

# Inference with LoFTR and get prediction
with torch.no_grad():
    paths = [("LoFTR", "loftr_coarse", "layers", "0", "attention")]
    path = ('LoFTR', 'loftr_coarse', 'layers', '0', 'attention')
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()
    #print(prof.display(show_events=False))

num = len(mkpts0)
index = [random.randint(0, num - 1) for _ in range(100)]
mkpts0_t = np.array(list(zip(mkpts0[index]))).reshape(100,2)
mkpts1_t = np.array(list(zip(mkpts1[index]))).reshape(100,2)
mconf_t = np.array(list(zip(mconf[index]))).reshape(100,1)

color = cm.jet(mconf_t, alpha=0.7)
text = [
    'LoFTR',
    'Matches: {}'.format(len(mkpts0)),
]


#测试将两帧图片用特征点对齐
M, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
img0_warp = cv2.warpPerspective(img0_raw,M,(640, 480))
image_list= (img0_warp, img1_raw)
gif_name = 'homo.gif'
create_gif(image_list,gif_name)


#fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1, text)

# A high-res PDF will also be downloaded automatically.
make_matching_figure(img0_raw, img1_raw, mkpts0_t, mkpts1_t, color, mkpts0_t, mkpts1_t, text, path="LoFTR-colab-demo.pdf")
#files.download("LoFTR-colab-demo.pdf")