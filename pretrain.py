import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader, random_split
import utils
from loaders.mms_dataloader_dg_aug import get_dg_data_loaders
import models
from torch.utils.tensorboard import SummaryWriter

def get_args():
    usage_text = (
        "UNet Pytorch Implementation"
        "Usage:  python pretrain.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('-e','--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('-bs','--batch_size', type=int, default=4, help='Number of inputs per batch')
    parser.add_argument('-c', '--cp', type=str, default='checkpoints', help='The name of the checkpoints.')
    parser.add_argument('-t', '--tv', type=str, default='A', help='The name of the checkpoints.')
    parser.add_argument('-w', '--wc', type=str, default='UNet', help='The name of the checkpoints.')
    parser.add_argument('-mn', '--model_name', type=str, default='unet', help='Name of the model architecture to be used for training/testing.')
    parser.add_argument('-lr','--learning_rate', type=float, default='0.0001', help='The learning rate for model training')
    parser.add_argument('-wi','--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--save_path', type=str, default='checkpoints', help= 'Path to save model checkpoints')
    #hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    parser.add_argument('--num_workers' ,type= int, default = 0, help='Number of workers to use for dataload')

    return parser.parse_args()

# python pretrain.py -e 50 -bs 4 -c cp_unet_100_tvA/ -t A -w UNet_tvA -g 0

def train_net(args):
    epochs = args.epochs
    batch_size = args.batch_size
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    dir_checkpoint = args.cp
    test_vendor = args.tv
    wc = args.wc

    #Model selection and initialization
    model_params = {
        'num_classes': 1,
    }
    model = models.get_model(args.model_name, model_params)
    num_params = utils.count_parameters(model)
    print('Model Parameters: ', num_params)
    models.initialize_weights(model, args.weight_init)
    model.to(device)

    train_labeled_loader, train_labeled_dataset, train_unlabeled_loader, train_unlabeled_dataset, test_loader, test_dataset = get_dg_data_loaders(args.batch_size, test_vendor=test_vendor, image_size=224)

    n_val = int(len(train_labeled_dataset) * 0.1)
    n_train = len(train_labeled_dataset) - n_val

    train, val = random_split(train_labeled_dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=True)


    print(len(train))
    print(len(val))
    print(len(train_unlabeled_dataset))

    #metrics initialization
    l1_distance = nn.L1Loss().to(device)

    #optimizer initialization
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # need to use a more useful lr_scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    writer = SummaryWriter(comment=wc)

    global_step = 0

    for epoch in range(epochs):
        model.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for imgs, true_masks in train_loader:
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                true_masks = true_masks.to(device=device, dtype=mask_type)

                out = model(imgs)
                reco = out[0]

                reco_loss = l1_distance(reco, imgs)

                batch_loss = reco_loss
                writer.add_scalar('Loss/reco_loss', reco_loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': batch_loss.item()})

                optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])

                if global_step % (len(train_labeled_dataset) // (2 * batch_size)) == 0:
                    writer.add_images('images/train', imgs, global_step)
                    writer.add_images('images/train_reco', reco, global_step)
                global_step += 1
        if optimizer.param_groups[0]['lr']<=2e-8:
            print('Converge')

        scheduler.step()
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

    # save checkpoint
    print("Epoch checkpoint")
    try:
        os.mkdir(dir_checkpoint)
        logging.info('Created checkpoint directory')
    except OSError:
        pass
    torch.save(model.state_dict(),
               dir_checkpoint + 'UNet.pth')
    logging.info(f'Checkpoint {epoch + 1} saved !')

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