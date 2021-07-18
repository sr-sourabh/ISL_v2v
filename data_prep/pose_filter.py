import numpy as np
import argparse
import os
import glob
from renderpose import *
import cv2
import time
import pickle
import sys
sys.path.append(os.getcwd())
print(sys.path)
import util.hand_utils as hand_utils
from scipy import signal

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--keypoints", type=str, help="keypoint directory")
parser.add_argument("--frame_dir", type=str, help='frame directory')
parser.add_argument("--save_dir", type=str, help='save directory')
parser.add_argument("--display", help='display the output', action="store_true")

args = parser.parse_args()

img_path = os.path.join(args.frame_dir, "*.png")
imgs = glob.glob(img_path)
imgs.sort()

keypoint_path = os.path.join(args.keypoints, "*.json")
keypoints = glob.glob(keypoint_path)
keypoints.sort()

posepts_list = []
facepts_list = []
r_handpts_list = []
l_handpts_list = []

_r_hpts_x = [np.zeros((len(keypoints)))] * 21
_r_hpts_y = [np.zeros((len(keypoints)))] * 21
_l_hpts_x = [np.zeros((len(keypoints)))] * 21
_l_hpts_y = [np.zeros((len(keypoints)))] * 21

for f in range(len(keypoints)):
    posepts, facepts, r_handpts, l_handpts = readkeypointsfile_json(keypoints[f])
    posepts_list.append(posepts)
    facepts_list.append(facepts)
    r_handpts_list.append(r_handpts)
    l_handpts_list.append(l_handpts)
    # for p in range(21):
    #     _r_hpts_x[p][f] = r_handpts[p*3]
    #     _r_hpts_y[p][f] = r_handpts[p*3+1]
    #     _l_hpts_x[p][f] = l_handpts[p*3]
    #     _l_hpts_y[p][f] = l_handpts[p*3+1]
    

# for p in range(21):
#     _r_hpts_x_new = signal.savgol_filter(_r_hpts_x, 13, 2)
#     _r_hpts_y_new = signal.savgol_filter(_r_hpts_y, 13, 2)
#     _l_hpts_x_new = signal.savgol_filter(_l_hpts_x, 13, 2)
#     _l_hpts_y_new = signal.savgol_filter(_l_hpts_y, 13, 2)

# for f in range(len(keypoints)):
#     for p in range(21):
#        r_handpts_list[f][p*0] =  _r_hpts_x_new[p][f]
#        r_handpts_list[f][p*0+1] =  _r_hpts_y_new[p][f]
#        l_handpts_list[f][p*0] =  _l_hpts_x_new[p][f]
#        l_handpts_list[f][p*0+1] =  _l_hpts_y_new[p][f]
    
def get_valid_prev_idx(points_lists, k):
    if k > 0:
        if set(points_lists[k-1]) == {0}:
            k = get_valid_prev_idx(points_lists, k-1)
        else:
            return k-1
    return k

def get_valid_next_idx(points_lists, k):
    if k < len(points_lists)-1:
        if set(points_lists[k+1]) == {0}:
            k = get_valid_next_idx(points_lists, k+1)
        else:
            return k+1
    return k


for k in range(len(keypoints)):
    if set(r_handpts_list[k]) == {0} and k > 0 and k < len(r_handpts_list)-1:
        r_prev = get_valid_prev_idx(r_handpts_list,k)
        #r_prev = r_prev if abs(r_prev-k)  < 3 else k
        r_next = get_valid_next_idx(r_handpts_list,k)
        #r_next = r_next if abs(r_next-k)  < 3 else k
        r_handpts_list[k] = [(a + b)/2 for a, b in zip(r_handpts_list[r_prev], r_handpts_list[r_next])]
    if set(l_handpts_list[k]) == {0} and k > 0 and k < len(l_handpts_list)-1:
        l_prev = get_valid_prev_idx(l_handpts_list,k)
        #l_prev = l_prev if abs(l_prev-k)  < 3 else k
        l_next = get_valid_next_idx(l_handpts_list,k)
        #l_next = l_next if abs(l_next-k)  < 3 else k
        l_handpts_list[k] = [(a + b)/2 for a, b in zip(l_handpts_list[l_prev], l_handpts_list[l_next])]

for f in range(len(imgs)):
    posepts, facepts, r_handpts, l_handpts = readkeypointsfile_json(keypoints[f])
    if not posepts or not facepts:
        print("Skipping frame: ", f)
        continue
    _frame = cv2.imread(imgs[f])
    height, width, _ = _frame.shape
    output_frame = np.zeros((height, width, 3), np.uint8)
    output_frame.fill(255)
    output_frame_ = np.zeros((height, width, 3), np.uint8)
    output_frame_.fill(255)
    display_skleton(output_frame, posepts, facepts, r_handpts, l_handpts, [0, 0])
    display_skleton(output_frame_, posepts_list[f], facepts_list[f], r_handpts_list[f], l_handpts_list[f], [0, 0])
    _out = cv2.hconcat([output_frame, output_frame_])
    _out = cv2.putText(_out, str(f), (30,30), cv2.FONT_HERSHEY_SIMPLEX , 0.4, (255, 0, 0) , 1, cv2.LINE_AA) 
    cv2.imshow("Skleton Frame", _out)
    cv2.waitKey(0)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    