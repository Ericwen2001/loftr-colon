front_matter = """
------------------------------------------------------------------------
Online demo for [LoFTR](https://zju3dv.github.io/loftr/).

This demo is heavily inspired by [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork/).
We thank the authors for their execellent work.
------------------------------------------------------------------------
"""
import pdb
import os
import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.cm as cm

os.sys.path.append("../")  # Add the project directory
from src.loftr import LoFTR, default_cfg
from src.config.default import get_cfg_defaults

try:
    from demo.utils import (AverageTimer, VideoStreamer,
                            make_matching_plot_fast, make_matching_plot, frame2tensor)
except:
    raise ImportError("This demo requires utils.py from SuperGlue, please use run_demo.sh to start this script.")

def abc(x):
    a = keyboard.KeyboardEvent('down', 28, 'q')
    #按键事件a为按下enter键，第二个参数如果不知道每个按键的值就随便写，
    #如果想知道按键的值可以用hook绑定所有事件后，输出x.scan_code即可
    if x.event_type == 'down' and x.name == a.name:
        return True
    #当监听的事件为enter键，且是按下的时候

def on_press(key):
    if key == Key.esc:
        print(f"你按下了esc，监听结束")
        return False

def img_intensify(img_tensor):
    a=2
    b=0
    img_tensor =img_tensor.astype(float)*a+b
    np.maximum(img_tensor, 0)
    np.minimum(img_tensor,255)
    return img_tensor


torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LoFTR online demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weight', type=str, help="Path to the checkpoint.")
    parser.add_argument(
        '--input', type=str, default='/data/pantianbo/ptb_LoFTR/demo/FIBONEDO04.MP4',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')
    parser.add_argument(
        '--output_matches_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')
    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--cut', type=int, nargs='+',default=None,
        help='cut the image, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--save_video', action='store_true',
        help='Save output (with match visualizations) to a video.')
    parser.add_argument(
        '--save_input', action='store_true',
        help='Save the input images to a video (for gathering repeatable input source).')
    parser.add_argument(
        '--skip_frames', type=int, default=1, 
        help="Skip frames from webcam input.")
    parser.add_argument(
        '--top_k', type=int, default=2000, help="The max vis_range (please refer to the code).")
    parser.add_argument(
        '--bottom_k', type=int, default=0, help="The min vis_range (please refer to the code).")

    opt = parser.parse_args()
    print(front_matter)
    parser.print_help()
    
    
    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        raise RuntimeError("GPU is required to run this demo.")



    # Configure I/O
    if not opt.save_video:
        print('Writing video to loftr-matches.mp4...')
        writer = cv2.VideoWriter('loftr-matches.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (703*2 + 10, 703))
    if opt.save_input:
        print('Writing video to demo-input.mp4...')
        input_writer = cv2.VideoWriter('demo-input.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))

    vs = VideoStreamer(opt.input, opt.resize, opt.cut, opt.skip,
                       opt.image_glob, opt.max_length)
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    frame_id = 0  
    last_image_id = 0
    #frame = torch.from_numpy(frame)
    #torch.clamp(frame, 0, 255)
    #frame = img_intensify(frame)
    #frame = frame.cpu().numpy()
    frame_tensor = frame2tensor(frame, device)
    last_data = {'image0': frame_tensor}
    last_frame = frame
    #pdb.set_trace()

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    if opt.output_matches_dir is not None:
        print('==> Will write Matching pairs to {}'.format(opt.output_matches_dir))
        Path(opt.output_matches_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if  opt.no_display:
        window_name = 'LoFTR Matches'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, (640*2, 480))
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the reference image (left)\n'
          '\td/f: move the range of the matches (ranked by confidence) to visualize\n'
          '\tc/v: increase/decrease the length of the visualization range (i.e., total number of matches) to show\n'
          '\tq: quit')

    timer = AverageTimer()
    vis_range = [opt.bottom_k, opt.top_k]

    while True:
        if frame_id % 500==0:
            x = input('请输入（按s暂停  按c继续看后面100帧）：')
            if x == 's':
                break
        frame_id += 1
        frame, ret = vs.next_frame()
        if frame_id % opt.skip_frames != 0:
            # print("Skipping frame.")
            continue
        if opt.save_input:
            inp = np.stack([frame] * 3, -1)
            inp_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            input_writer.write(inp_rgb)
        if not ret:
            print('Finished demo_loftr.py')
            break
        timer.update('data')
        stem0, stem1 = last_image_id, vs.i - 1
        #frame = torch.from_numpy(frame)
        #torch.clamp(frame, 0, 255)
        #frame = img_intensify(frame)
        #frame = frame.cpu().numpy()
        frame_tensor = frame2tensor(frame, device)
        last_data = {**last_data, 'image1': frame_tensor}
        #pdb.set_trace()
        #pdb.set_trace()
        # Normalize confidence.


        timer.update('forward')
        
     
        # Save high quality png, optionally with dynamic alpha support (unreleased yet).
        # save_path = 'demo_vid/{:06}'.format(frame_id)
        # make_matching_plot(
        #     last_frame, frame, mkpts0, mkpts1, mkpts0, mkpts1, color, text,
        #     path=save_path, show_keypoints=opt.show_keypoints, small_text=small_text)
        frame_num = 'frame_{:06}'.format(stem0)
        croped_path = str(Path('/data/pantianbo/ptb_LoFTR/demo/Croped_Image', frame_num + '.png'))
        cv2.imwrite(croped_path, last_frame)
        print('input frame :',frame_num)
        last_data['image0'] = frame_tensor
        last_frame = frame
        last_image_id = (vs.i - 1)
        frame_id_left = frame_id

        timer.update('viz')
        timer.print()
    croped_path = str(Path('/data/pantianbo/LoFTR/demo/Croped_Image', 'frame_{:06}'.format(stem1) + '.png'))
    cv2.imwrite(croped_path, frame)
    writer.release()
    cv2.destroyAllWindows()
    vs.cleanup()
