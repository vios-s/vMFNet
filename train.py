import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
import logging
from metrics.focal_loss import FocalLoss
from torch.utils.data import DataLoader, random_split
import utils
from eval import eval_vmfnet
from loaders.mms_dataloader_dg_aug import get_dg_data_loaders
import models
from composition.losses import ClusterLoss
import losses
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument('-w', '--wc', type=str, default='vmfnet_12_p100_tvA', help='The name of the checkpoints.')
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

# python train.py -e 200 -bs 4 -c cp_vmfnet_100_tvA/ -enc /home/s1575424/xiao/Year3/comp_decoder/CompCSD/cp_unet_100_tvA/UNet.pth -t A -w vmfnet_12_p100_tvA -g 0
# k_un = 1
# k1 = 40
# k2 = 4

# python train.py -e 1200 -bs 4 -c cp_vmfnet_5_tvA/ -enc /home/s1575424/xiao/Year3/comp_decoder/CompCSD/cp_unet_100_tvA/UNet.pth -t A -w vmfnet_12_p5_tvA -g 0
k_un = 1
k1 = 400
k2 = 40

# python train.py -e 2000 -bs 4 -c cp_vmfnet_2_tvA/ -enc /home/s1575424/xiao/Year3/comp_decoder/CompCSD/cp_unet_100_tvA/UNet.pth -t A -w vmfnet_12_p2_tvA -g 0
# k_un = 1
# k1 = 600
# k2 = 80



lr_patience = 4
layer = 8
vc_num = 12 # kernel numbers

def latent_norm(a):
    n_batch, n_channel, _, _ = a.size()
    for batch in range(n_batch):
        for channel in range(n_channel):
            a_min = a[batch,channel,:,:].min()
            a_max = a[batch, channel, :, :].max()
            a[batch,channel,:,:] -= a_min
            a[batch, channel, :, :] /= a_max - a_min
    return a

def train_net(args):
    best_dice = 0
    best_lv = 0
    best_myo = 0
    best_rv = 0
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    dir_checkpoint = args.cp
    test_vendor = args.tv
    wc = args.wc
    enc_dir = args.encoder_dir

    #Model selection and initialization
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
    models.initialize_weights(model, args.weight_init)
    model.to(device)
    #################################################### load pre-trained encoder and vMF kernels
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
    dict_dir = init_path + 'dictionary/'+kernel_save_name
    model.load_vmf_kernels(dict_dir)
    # models.initialize_weights(model, args.weight_init)
    ####################################################

    train_labeled_loader, train_labeled_dataset, train_unlabeled_loader, train_unlabeled_dataset, test_loader, test_dataset = get_dg_data_loaders(args.batch_size, test_vendor=test_vendor, image_size=224)

    n_train = len(train_labeled_dataset)
    train_loader = train_labeled_loader
    val_loader = train_labeled_loader

    print(n_train)
    print(len(train_unlabeled_dataset))

    #metrics initialization
    l2_distance = nn.MSELoss().to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    l1_distance = nn.L1Loss().to(device)
    focal = FocalLoss()
    cluster_loss = ClusterLoss()

    #optimizer initialization
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=lr_patience)

    writer = SummaryWriter(comment=wc)

    global_step = 0
    un_step = 0

    for epoch in range(epochs):
        model.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            un_itr = iter(train_unlabeled_loader)
            for imgs, true_masks in train_loader:
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                ce_mask = true_masks.clone().to(device=device, dtype=torch.long)
                true_masks = true_masks.to(device=device, dtype=mask_type)
                rec, pre_seg, features, kernels, L_visuals = model(imgs, layer=layer)

                dice_loss_lv = losses.dice_loss(pre_seg[:,0,:,:], true_masks[:,0,:,:])
                dice_loss_myo = losses.dice_loss(pre_seg[:,1,:,:], true_masks[:,1,:,:])
                dice_loss_rv = losses.dice_loss(pre_seg[:,2,:,:], true_masks[:,2,:,:])
                dice_loss_bg = losses.dice_loss(pre_seg[:, 3, :, :], true_masks[:, 3, :, :])
                loss_dice = dice_loss_lv + dice_loss_myo + dice_loss_rv + dice_loss_bg

                ce_target = ce_mask[:, 3, :, :]*0 + ce_mask[:, 0, :, :]*1 + ce_mask[:, 1, :, :]*2 + ce_mask[:, 2, :, :]*3

                seg_pred_swap = torch.cat((pre_seg[:,3,:,:].unsqueeze(1), pre_seg[:,:3,:,:]), dim=1)

                loss_focal = focal(seg_pred_swap, ce_target)

                reco_loss = l1_distance(rec, imgs)
                clu_loss = cluster_loss(features.detach(), kernels)

                batch_loss = 1*reco_loss + 1*clu_loss  + loss_dice + loss_focal

                pbar.set_postfix(**{'loss (batch)': batch_loss.item()})

                optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                writer.add_scalar('loss/batch_loss', batch_loss.item(), global_step)
                writer.add_scalar('loss/reco_loss', reco_loss.item(), global_step)
                writer.add_scalar('loss/loss_focal', loss_focal.item(), global_step)
                writer.add_scalar('loss/loss_dice', loss_dice.item(), global_step)
                writer.add_scalar('loss/loss_dice_lv', dice_loss_lv.item(), global_step)
                writer.add_scalar('loss/loss_dice_myo', dice_loss_myo.item(), global_step)
                writer.add_scalar('loss/loss_dice_rv', dice_loss_rv.item(), global_step)
                writer.add_scalar('loss/loss_dice_bg', dice_loss_bg.item(), global_step)
                writer.add_scalar('loss/cluster_loss', clu_loss.item(), global_step)

                if global_step % ((n_train//batch_size) // 2) == 0:
                    writer.add_images('images/train', imgs, global_step)
                    writer.add_images('images/train_reco', rec, global_step)
                    writer.add_images('images/train_true', true_masks[:, 0:3, :, :], global_step)
                    writer.add_images('images/train_pred', pre_seg[:, 0:3, :, :] > 0.5, global_step)
                    writer.add_images('L_visuals/L_1', L_visuals[:,0,:,:].unsqueeze(1), global_step)
                    writer.add_images('L_visuals/L_2', L_visuals[:,1,:,:].unsqueeze(1), global_step)
                    writer.add_images('L_visuals/L_3', L_visuals[:,2,:,:].unsqueeze(1), global_step)
                    writer.add_images('L_visuals/L_4', L_visuals[:,3,:,:].unsqueeze(1), global_step)
                    writer.add_images('L_visuals/L_5', L_visuals[:,4,:,:].unsqueeze(1), global_step)
                    writer.add_images('L_visuals/L_6', L_visuals[:,5,:,:].unsqueeze(1), global_step)
                    writer.add_images('L_visuals/L_7', L_visuals[:,6,:,:].unsqueeze(1), global_step)
                    writer.add_images('L_visuals/L_8', L_visuals[:,7,:,:].unsqueeze(1), global_step)
                    writer.add_images('L_visuals/L_9', L_visuals[:,8,:,:].unsqueeze(1), global_step)
                    writer.add_images('L_visuals/L_10', L_visuals[:,9,:,:].unsqueeze(1), global_step)
                    writer.add_images('L_visuals/L_11', L_visuals[:,10,:,:].unsqueeze(1), global_step)
                    writer.add_images('L_visuals/L_12', L_visuals[:,11,:,:].unsqueeze(1), global_step)

                for i in range(k_un):
                    un_imgs = next(un_itr)
                    un_imgs = un_imgs.to(device=device, dtype=torch.float32)

                    rec, pre_seg, features, kernels, L_visuals = model(un_imgs, layer=layer)

                    un_reco_loss = l1_distance(rec, un_imgs)
                    un_clu_loss = cluster_loss(features.detach(), kernels)

                    un_batch_loss = 1*un_reco_loss + 1*un_clu_loss

                    optimizer.zero_grad()
                    un_batch_loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), 0.1)
                    optimizer.step()

                    writer.add_scalar('Loss_un/un_reco_loss', un_reco_loss.item(), un_step)
                    writer.add_scalar('Loss_un/un_clu_loss', un_clu_loss.item(), un_step)
                    writer.add_scalar('Loss_un/un_batch_loss', un_batch_loss.item(), un_step)

                    un_step += 1

                    if global_step % (len(train_labeled_dataset) // (2 * batch_size)) == 0:
                        writer.add_images('unlabelled/train_un_img', un_imgs, global_step)
                        writer.add_images('unlabelled/train_un_mask', pre_seg[:, 0:3, :, :] > 0.5, global_step)


                pbar.update(imgs.shape[0])

                global_step += 1

            if optimizer.param_groups[0]['lr'] <= 2e-8:
                print('Converge')
            if (epoch + 1) > k1 and (epoch + 1) % k2 == 0:
                val_score, val_lv, val_myo, val_rv = eval_vmfnet(model, val_loader, device, layer)
                scheduler.step(val_score)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                logging.info('Validation Dice Coeff: {}'.format(val_score))
                logging.info('Validation LV Dice Coeff: {}'.format(val_lv))
                logging.info('Validation MYO Dice Coeff: {}'.format(val_myo))
                logging.info('Validation RV Dice Coeff: {}'.format(val_rv))

                writer.add_scalar('Dice/val', val_score, epoch)
                writer.add_scalar('Dice/val_lv', val_lv, epoch)
                writer.add_scalar('Dice/val_myo', val_myo, epoch)
                writer.add_scalar('Dice/val_rv', val_rv, epoch)

                initial_itr = 0
                for imgs, true_masks in test_loader:
                    if initial_itr == 5:
                        model.eval()
                        imgs = imgs.to(device=device, dtype=torch.float32)
                        with torch.no_grad():
                            rec, pre_seg, features, kernels, L_visuals  = model(imgs, layer=layer)

                        mask_type = torch.float32
                        true_masks = true_masks.to(device=device, dtype=mask_type)
                        writer.add_images('Test_images/test', imgs, epoch)
                        writer.add_images('Test_images/test_reco', rec, epoch)
                        writer.add_images('Test_images/test_true', true_masks[:, 0:3, :, :], epoch)
                        writer.add_images('Test_images/test_pred', pre_seg[:, 0:3, :, :] > 0.5, epoch)
                        model.train()
                        break
                    else:
                        pass
                    initial_itr += 1
                test_score, test_lv, test_myo, test_rv = eval_vmfnet(model, test_loader, device, layer)

                if best_dice < test_score:
                    best_dice = test_score
                    best_lv = test_lv
                    best_myo = test_myo
                    best_rv = test_rv
                    print("Epoch checkpoint")
                    try:
                        os.mkdir(dir_checkpoint)
                        logging.info('Created checkpoint directory')
                    except OSError:
                        pass
                    torch.save(model.state_dict(),
                               dir_checkpoint + 'CP_epoch.pth')
                    logging.info('Checkpoint saved !')
                else:
                    pass
                logging.info('Best Dice Coeff: {}'.format(best_dice))
                logging.info('Best LV Dice Coeff: {}'.format(best_lv))
                logging.info('Best MYO Dice Coeff: {}'.format(best_myo))
                logging.info('Best RV Dice Coeff: {}'.format(best_rv))
                writer.add_scalar('Dice/test', test_score, epoch)
                writer.add_scalar('Dice/test_lv', test_lv, epoch)
                writer.add_scalar('Dice/test_myo', test_myo, epoch)
                writer.add_scalar('Dice/test_rv', test_rv, epoch)
    writer.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    torch.manual_seed(14)
    if device.type == 'cuda':
        torch.cuda.manual_seed(14)

    train_net(args)
