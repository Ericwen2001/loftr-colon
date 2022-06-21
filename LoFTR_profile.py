import torch
import cv2
import numpy as np
import matplotlib.cm as cm
import torch.cuda.profiler as profiler
import pyprof
import pdb
import torchprof
import io

from copy import deepcopy
from matplotlib import pyplot as plt
from highcharts import Highchart
from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg
#from torch.profiler import profile, record_function, ProfilerActivity

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()

img0_pth = "assets/ptb_testdata/1.jpg" #set your image path here
img1_pth = "assets/ptb_testdata/2.jpg"
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
    with torchprof.Profile(matcher, use_cuda=True, profile_memory=True) as prof:
        matcher(batch)
    print(prof)
    result = ""
    result += prof.display()
    trace, event_lists_dict = prof.raw()
    run_time = 1 #set the time threshold(ms) to filter the layers running longer than threshold
    result += "\n\nThe layers cost longer than %dms: \n" % run_time
    for i in range(0,len(trace)):
        if len(event_lists_dict[trace[i].path]):
            if event_lists_dict[trace[i].path][0].self_cpu_time_total>run_time*1000:
                print(trace[i].path)
                result += "%s\t\t\tCost time in CPU:%fms\n" % (str(trace[i].path),(event_lists_dict[trace[i].path][0].self_cpu_time_total/1000))
                #result += "%s\n" % str(event_lists_dict[trace[i].path][0]) #show the details of these costy modules
        else:
            i+=1
    #pdb.set_trace()
    excel = open("../LoFTR/profile.xlsx", "w").write(result)



    #draw a pie
    run_time_pie = 0.5 #set the pie-chart time threshold(ms)
    H = Highchart(width=850,height=500)
    Pai_dic = {}
    data = []  # 各个值，影响各个扇形的面积
    for i in range(0, len(trace)):
        if len(event_lists_dict[trace[i].path]):
            if event_lists_dict[trace[i].path][0].self_cpu_time_total > run_time_pie * 1000:
                Pai_dic['id'] = i
                Pai_dic['name'] = trace[i].path
                Pai_dic['y'] = event_lists_dict[trace[i].path][0].self_cpu_time_total
                Pai_dic['colors'] = 'Highcharts.getOptions().colors[%d]' % i
                data.append(Pai_dic)
                data = deepcopy(data)
    #pdb.set_trace()
    options = {
        'title':{
            'text': 'Profile'
        },
        'tooltip':{
            'pointFormat': '{series.name}: <b>{point.percentage:.1f}%</b>}'
        },
        'plotOptions':{
            'pie': {
                'allowPointSelect': True,  # 允许某个区块选择后弹出来
                'cursor': 'pointer',
            }
        }
    }
    H.set_dict_options(options)
    H.add_data_set(data, 'pie', "Cost:")
    H.save_file('LoFTR_Profile_pie')


    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()
    #print(prof.display(show_events=False))
color = cm.jet(mconf, alpha=0.7)
text = [
    'LoFTR',
    'Matches: {}'.format(len(mkpts0)),
]


fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1, text)

# A high-res PDF will also be downloaded automatically.
make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1, text, path="LoFTR-colab-demo.pdf")
#files.download("LoFTR-colab-demo.pdf")