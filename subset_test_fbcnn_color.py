import argparse, glob, os
import cv2, torch, numpy as np
from models.network_fbcnn import FBCNN
from utils import utils_image as util

'''
# --------------------------------------------
# PSNR of an FBCNN checkpoint on the first N COCO val2017 images at QF=10,
# computed the same way as the test step in main_train_fbcnn.py
# --------------------------------------------
'''

here = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--model', default=os.path.join(here, '../../checkpoints/fbcnn_color.pth'),
                    help='checkpoint to test, e.g. deblocking/FBCNN-Color-finetune/models/25000_G.pth')
parser.add_argument('--n_images', type=int, default=50, help='must match the size of val_small')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

net = FBCNN(act_mode='R')
net.load_state_dict(torch.load(args.model, map_location='cpu'), strict=True)
net.eval().to(device)

# Same files as val_small: the first n_images of val2017, sorted
val_dir = os.path.join(here, '../../data/coco/coco_minitrain_10k/images/val2017')
paths = sorted(glob.glob(os.path.join(val_dir, '*.jpg')))[:args.n_images]
assert paths, 'No images found in {:s}'.format(val_dir)

psnrs = []
for p in paths:
    H = util.imread_uint(p, 3)
    _, enc = cv2.imencode('.jpg', cv2.cvtColor(H, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 10])
    L = cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        E, _ = net(util.uint2tensor4(L).to(device))
    psnrs.append(util.calculate_psnr(util.tensor2uint(E), H))

print('{:s} on {:s}: val PSNR @ QF10 = {:.2f} dB ({:d} images)'.format(
    os.path.basename(args.model), str(device), np.mean(psnrs), len(psnrs)))
