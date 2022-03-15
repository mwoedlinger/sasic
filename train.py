__all__ = []
__author__ = "Matthias Wödlinger"


from sasic import *
from torchvision import transforms
from datetime import datetime
from time import time
from tqdm import tqdm
from pathlib import Path
import argparse
from pprint import pprint
import torch
from torch import optim
from torch.utils.data import DataLoader

##########################################################################################################################
# Experiment parameters
c = Dict(
    DEBUG = False,
    PROJECT_NAME = 'cvpr2022',
    ENTITY = 'aistream',
    EPOCHS = 300,
    LMDA = None,
    LR = None,
    LR_DROP = None,
    BATCH_SIZE = 1,
    EPS = 1e-9,
    EVAL_EPOCHS = 10,

    TRAIN = Dict(
                name='cityscapes#train',
                transform=[transforms.RandomCrop((256, 256))],
                kwargs={'debug': False}),
    EVAL = Dict(
                name='cityscapes#eval', 
                # transform=[MinimalCrop()], 
                transform=[transforms.CenterCrop((1024, 1024))], 
                kwargs={'debug': False})
)
##########################################################################################################################



def loss_func(bpp, mse):
    return (bpp + c.LMDA * mse) / (1 + c.LMDA)

def train(exp_path, device, resume=False):
    """
    Training loop function.
    """
    train_set = StereoImageDataset(c.TRAIN.name, transform=c.TRAIN.transform, **c.TRAIN.kwargs)
    train_loader = DataLoader(train_set, batch_size=c.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
                              
    eval_set = StereoImageDataset(c.EVAL.name, transform=c.EVAL.transform, **c.EVAL.kwargs)
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = StereoEncoderDecoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=c.LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    if resume:
        print(f'Resume from {resume}')
        start_epoch = load_model(model, optimizer, resume)
    else:
        start_epoch = 0

    last_time = time()
    for epoch in range(start_epoch, c.EPOCHS):
        print(f'\ntrain epoch {epoch}/{c.EPOCHS}')

        model.train()
        scalars = {k: [] for k in ('loss', 'bpp', 'mse', 'psnr', 'lr')}

        last_time = time()
        for sample in tqdm(train_loader):
            left = sample['left'].to(device)
            right = sample['right'].to(device)

            ################
            # Training Phase
            ################

            optimizer.zero_grad()

            output = model(left, right, training=True)
            pred_left, pred_right, rate_left, rate_right = output.pred_left, output.pred_right, output.rate_left, output.rate_right

            # Compute MSE
            mse_left = calc_mse(left, pred_left)
            mse_right = calc_mse(right, pred_right)
            mse = (mse_left + mse_right)/2

            # Compute PSNR
            psnr_left = calc_psnr(mse_left, eps=c.EPS)
            psnr_right = calc_psnr(mse_right, eps=c.EPS)
            psnr = (psnr_left + psnr_right)/2

            # Compute BPP
            bpp_y_left = calc_bpp(rate_left.y, left)
            bpp_z_left = calc_bpp(rate_left.z, left)
            bpp_y_right = calc_bpp(rate_right.y, right)
            bpp_z_right = calc_bpp(rate_right.z, right)
            bpp = (bpp_y_left + bpp_z_left + bpp_y_right + bpp_z_right)/2

            # Computer RD-Loss
            loss = loss_func(bpp, mse)

            # Backward - optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.50)
            optimizer.step()

            # Log scalars

            scalars['loss'].append(loss.item())
            scalars['bpp'].append(bpp.item())
            scalars['mse'].append(mse.item())
            scalars['psnr'].append(psnr.item())
            scalars['lr'].append(optimizer.param_groups[0]['lr'])

        print(f'  ## Finished epoch in time: {time()-last_time}.')
        last_time = time()

        log = [f'  ## epoch {epoch} averages:']
        for scalar in scalars:
            log.append(f'  {scalar:10}: {torch.tensor(scalars[scalar]).mean().item():.4}')
        log_str = '\n' + '\n'.join(log)

        with open(exp_path / 'log.txt', 'a') as f:
            f.write(log_str)
            print(log_str)

        
        if epoch % c.EVAL_EPOCHS == 0:
            print(f'########### EVALUATION BEGIN ################')
            with torch.no_grad():
                eval_model(eval_loader, model, device, exp_path)
            save_model(model, optimizer, epoch, exp_path)
            print(f'############ EVALUATION END #################')

        if epoch % int(c.LR_DROP/len(train_set)) == 0 and epoch > 0:
            scheduler.step()

def save_model(model, optimizer, epoch, exp_path):
    state_dict_model = model.state_dict()
    state_dict_training = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict()
    }

    print(f'save model in {exp_path}')
    torch.save(state_dict_model, str(exp_path / 'model.pt'))
    torch.save(state_dict_training, str(exp_path / 'training_state.pt'))


def load_model(model, optimizer, exp_path):
    checkpoint_model = torch.load(str(exp_path / 'model.pt'))
    checkpoint_training = torch.load(str(exp_path / 'training_state.pt'))

    model.load_state_dict(checkpoint_model)
    optimizer.load_state_dict(checkpoint_training['optimizer'])

    return checkpoint_training['epoch']

def eval_model(eval_loader, model, device, exp_path):
    print(f'  Start evaluation on {c.EVAL.name} dataset.')
    model.eval()
    scalars = {k: [] for k in ('loss', 'bpp', 'bpp_left', 'bpp_right', 'mse', 'mse_left', 'mse_right', 'psnr')}

    for sample in tqdm(eval_loader):
        left = sample['left'].to(device)
        right = sample['right'].to(device)

        ################
        # Training Phase
        ################

        output = model(left, right, training=False)
        pred_left, pred_right, rate_left, rate_right = output.pred_left, output.pred_right, output.rate_left, output.rate_right

        # Compute MSE
        mse_left = calc_mse(left, pred_left)
        mse_right = calc_mse(right, pred_right)
        mse = (mse_left + mse_right)/2

        # Compute PSNR
        psnr_left = calc_psnr(mse_left, eps=c.EPS)
        psnr_right = calc_psnr(mse_right, eps=c.EPS)
        psnr = (psnr_left + psnr_right)/2

        # Compute BPP
        bpp_y_left = calc_bpp(rate_left.y, left)
        bpp_z_left = calc_bpp(rate_left.z, left)
        bpp_y_right = calc_bpp(rate_right.y, right)
        bpp_z_right = calc_bpp(rate_right.z, right)
        bpp = (bpp_y_left + bpp_z_left + bpp_y_right + bpp_z_right)/2

        # Computer RD-Loss
        loss = loss_func(bpp, mse)

        # Log scalars
        scalars['loss'].append(loss.item())
        scalars['bpp'].append(bpp.item())
        scalars['bpp_left'].append(bpp_y_left.item() + bpp_z_left.item())
        scalars['bpp_right'].append(bpp_y_right.item() + bpp_z_right.item())
        scalars['mse'].append(mse.item())
        scalars['mse_left'].append(mse_left.item())
        scalars['mse_right'].append(mse_right.item())
        scalars['psnr'].append(psnr.item())

    log = [f'    ## eval averages:']
    for scalar in scalars:
        log.append(f'    {scalar:10}: {torch.tensor(scalars[scalar]).mean().item():.4}')
    log_str = '\n' + '\n'.join(log)

    with open(exp_path / 'log.txt', 'a') as f:
        f.write(log_str)
        print(log_str)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Order of arguments: exp_name gpu_idx restore_path')
    parser.add_argument('argv', nargs='*', help='exp_name gpu_idx resume')
    parser.add_argument('--lmda', default=0.01, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr_drop', default=1500000, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--train', default='instereo2k', 
                        help=f'name of training dataset. Options (includes stero image datasets): {", ".join(list(data_zoo_stereo.keys()))}')
    parser.add_argument('--generic', action='store_true', 
                        help='If the program is called with the --generic flag set, the experiment name will be appended with lambda, epochs and training dataset name.')
    args = parser.parse_args()

    c.EPOCHS = args.epochs
    c.LMDA = args.lmda
    c.LR = args.lr
    c.BATCH_SIZE = args.batch_size
    c.LR_DROP = args.lr_drop
    c.TRAIN.name = args.train+'#train'
    print('CONFIG:')
    pprint(c)
    print('\n')

    if args.generic:
        exp_name = args.argv[0] + f'_{c.LMDA}_{c.EPOCHS}_{c.TRAIN.name}'
    else:
        exp_name = args.argv[0]
    if len(args.argv) > 2:
        resume = Path(args.argv[2])
    else:
        resume = False
    device = torch.device("".join(["cuda:", args.argv[1]]))

    dt = str(datetime.now())
    run_name = str(exp_name) + '-' + dt[2:10].replace('-', '') + '-' + dt[11:19].replace(':', '')
    exp_path = Path(f'./experiments/{run_name}')
    exp_path.mkdir(exist_ok=False)

    train(exp_path, device, resume)