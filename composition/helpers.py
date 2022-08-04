import torch
import cv2
from composition.vMFMM import *

def getVmfKernels(dict_dir, device):
	vc = np.load(dict_dir, allow_pickle=True)
	vc = vc[:, :, np.newaxis, np.newaxis]
	vc = torch.from_numpy(vc).type(torch.FloatTensor)
	if device:
		vc = vc.to(device)
	return vc

def myresize(img, dim, tp):
	H, W = img.shape[0:2]
	if tp == 'short':
		if H <= W:
			ratio = dim / float(H)
		else:
			ratio = dim / float(W)

	elif tp == 'long':
		if H <= W:
			ratio = dim / float(W)
		else:
			ratio = dim / float(H)

	return cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
