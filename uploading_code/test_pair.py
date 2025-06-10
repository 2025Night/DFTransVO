import torch
from copy import deepcopy

import cv2
import matplotlib.cm as cm

from EfficientLoFTR.src.loftr import LoFTR, full_default_cfg, reparameter
# from utils import get_world_points, calc_depth, new_get_world_points
from EfficientLoFTR.src.utils.plotting import make_matching_figure
from est_pose import estimate_pose




_default_cfg = deepcopy(full_default_cfg)


# _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt

matcher = LoFTR(config=_default_cfg)
matcher.load_state_dict(torch.load("weights/df_outdoor.ckpt")['state_dict'])
matcher = reparameter(matcher)
matcher = matcher.eval().cuda()

img0_pth = "LoFTR/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
img1_pth = "LoFTR/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"

img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))
img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))

img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}
batch1 = {'image0': img1, 'image1': img0}



# Inference
with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()

color = cm.jet(mconf)
text = [
    'Matches: {}'.format(len(mkpts0)),
]
fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text, path="matching_figure.png")
pose = estimate_pose(mkpts0, mkpts1)
print(pose)

