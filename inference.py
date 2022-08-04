import torch
import argparse
import torch.nn.functional as F
import statistics
import utils
from loaders.mms_dataloader_dg_aug_test import get_dg_data_loaders
import models
from metrics.dice_loss import dice_coeff
from metrics.hausdorff import hausdorff_distance
from torch.utils.tensorboard import SummaryWriter

# python inference.py -bs 1 -c cp_vmfnet_2_tvA/ -enc /home/s1575424/xiao/Year3/comp_decoder/CompCSD/cp_unet_100_tvA/UNet.pth -t A -g 0


def get_args():
    usage_text = (
        "vMFNet Pytorch Implementation"
        "Usage:  python train.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('-e','--epochs', type= int, default=50, help='Number of epochs')
    parser.add_argument('-bs','--batch_size', type= int, default=4, help='Number of inputs per batch')
    parser.add_argument('-c', '--cp', type=str, default='checkpoints', help='The name of the checkpoints.')
    parser.add_argument('-t', '--tv', type=str, default='D', help='The name of the checkpoints.')
    parser.add_argument('-w', '--wc', type=str, default='SDNet_LR00002_nB_FT', help='The name of the checkpoints.')
    parser.add_argument('-n','--name', type=str, default='default_name', help='The name of this train/test. Used when storing information.')
    parser.add_argument('-enc', '--encoder_dir', type=str, default='cp_unet_100_tvA/', help='The name of the pretrained encoder checkpoints.')
    parser.add_argument('-mn','--model_name', type=str, default='compcsd', help='Name of the model architecture to be used for training/testing.')
    parser.add_argument('-lr','--learning_rate', type=float, default='0.0001', help='The learning rate for model training')
    parser.add_argument('-wi','--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--save_path', type=str, default='checkpoints', help= 'Path to save model checkpoints')
    #hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    parser.add_argument('--num_workers' ,type= int, default = 0, help='Number of workers to use for dataload')

    return parser.parse_args()


args = get_args()
device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

torch.manual_seed(14)
if device.type == 'cuda':
    torch.cuda.manual_seed(14)

batch_size = args.batch_size

dir_checkpoint = args.cp
test_vendor = args.tv
wc = args.wc
model_name = args.model_name
enc_dir = args.encoder_dir

layer = 8
vc_num = 12 # kernel numbers

# Model selection and initialization
model_params = {
    'image_channels': 1,
    'layer': layer,
    'vc_numbers': vc_num,
    'num_classes': 3,
    'anatomy_out_channels': 4,
    'z_length': 8,
    'vMF_kappa': 30
}

model = models.get_model(args.model_name, model_params)
num_params = utils.count_parameters(model)
# print(model)
print('Model Parameters: ', num_params)
model.to(device)
model.load_encoder_weights(enc_dir, device)
if layer == 6:
    kernels_save_dir = '/home/s1575424/xiao/Year3/comp_decoder/CompCSD/' + test_vendor + '8_12kernels/'
elif layer == 7:
    kernels_save_dir = '/home/s1575424/xiao/Year3/comp_decoder/CompCSD/' + test_vendor + '4_12kernels/'
elif layer == 8:
    kernels_save_dir = '/home/s1575424/xiao/Year3/comp_decoder/CompCSD/' + test_vendor + '2_12kernels/'
else:
    kernels_save_dir = '/home/s1575424/xiao/Year3/comp_decoder/CompCSD/' + test_vendor + '_12kernels/'
init_path = kernels_save_dir + 'init/'
kernel_save_name = 'dictionary_12.pickle'
dict_dir = init_path + 'dictionary/' + kernel_save_name
model.load_vmf_kernels(dict_dir)
model.load_state_dict(torch.load(dir_checkpoint+'CP_epoch.pth', map_location=device))

_, _, _, _, test_loader, test_dataset = get_dg_data_loaders(
    args.batch_size, test_vendor=test_vendor, image_size=224)

step = 0
tot = []
tot_sub = []
tot_hsd = []
tot_sub_hsd = []
flag = '000'
# i = 0
for imgs, true_masks, path_img in test_loader:
    model.eval()
    imgs = imgs.to(device=device, dtype=torch.float32)
    mask_type = torch.float32
    true_masks = true_masks.to(device=device, dtype=mask_type)
    print(flag)
    if path_img[0][-10: -7] != flag:
        # if i > 10:
        #     break
        # i += 1
        flag = path_img[0][-10: -7]
        tot.append(sum(tot_sub)/len(tot_sub))
        tot_sub = []
        tot_hsd.append(sum(tot_sub_hsd)/len(tot_sub_hsd))
        tot_sub_hsd = []
    with torch.no_grad():
        rec, pre_seg, features, kernels, L_visuals = model(imgs,layer=layer)

    pred = (pre_seg > 0.5).float()
    dice = dice_coeff(pred[:, 0:3, :, :], true_masks[:, 0:3, :, :], device).item()
    hsd = hausdorff_distance(pred[:, 0:3, :, :], true_masks[:, 0:3, :, :])
    tot_sub.append(dice)
    tot_sub_hsd.append(hsd)
    # print(step)
    step += 1

print(tot)

print(sum(tot)/len(tot))
print(statistics.stdev(tot))

print(tot_hsd)

print(sum(tot_hsd)/len(tot_hsd))
print(statistics.stdev(tot_hsd))
