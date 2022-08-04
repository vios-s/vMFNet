import torch
from tqdm import tqdm
import torch.nn.functional as F
from metrics.dice_loss import dice_coeff

def eval_vmfnet(model, loader, device, layer):
    """Evaluation without the densecrf with the dice coefficient"""
    model.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0
    tot_lv = 0
    tot_myo = 0
    tot_rv = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for imgs, true_masks in loader:
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                rec, pre_seg, features, kernels, L_visuals = model(imgs, layer=layer)

            pred = (pre_seg > 0.5).float()
            tot += dice_coeff(pred[:, 0:3, :, :], true_masks[:, 0:3, :, :], device).item()
            tot_lv += dice_coeff(pred[:, 0, :, :], true_masks[:, 0, :, :], device).item()
            tot_myo += dice_coeff(pred[:, 1, :, :], true_masks[:, 1, :, :], device).item()
            tot_rv += dice_coeff(pred[:, 2, :, :], true_masks[:, 2, :, :], device).item()
            pbar.update()
    model.train()
    return tot / n_val, tot_lv / n_val, tot_myo / n_val, tot_rv / n_val
