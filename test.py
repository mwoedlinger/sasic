__all__ = []
__author__ = "Matthias Wödlinger"


from sasic import *
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import argparse
from pprint import pprint
import torch
from torch.utils.data import DataLoader

##########################################################################################################################
# Experiment parameters
c = Dict(
    EPS = 1e-9,

    TEST = Dict(
                name='cityscapes#test', 
                # transform=[MinimalCrop()], 
                transform=[transforms.CenterCrop((512, 512))], 
                kwargs={'debug': True})
)
##########################################################################################################################



def load_model(model, exp_path):
    checkpoint_model = torch.load(str(exp_path / 'model.pt'))
    model.load_state_dict(checkpoint_model)

def test(device, resume):
                              
    test_set = StereoImageDataset(c.TEST.name, transform=c.TEST.transform, **c.TEST.kwargs)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = StereoEncoderDecoder().to(device)
    load_model(model, resume)

    print(f'  Start test on {c.TEST.name} dataset.')
    model.eval()
    scalars = {k: [] for k in ('bpp', 'bpp_left', 'bpp_right', 'mse', 'mse_left', 'mse_right', 'psnr')}

    for sample in tqdm(test_loader):
        left = sample['left'].to(device)
        right = sample['right'].to(device)

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

        # Log scalars
        scalars['bpp'].append(bpp.item())
        scalars['bpp_left'].append(bpp_y_left.item() + bpp_z_left.item())
        scalars['bpp_right'].append(bpp_y_right.item() + bpp_z_right.item())
        scalars['mse'].append(mse.item())
        scalars['mse_left'].append(mse_left.item())
        scalars['mse_right'].append(mse_right.item())
        scalars['psnr'].append(psnr.item())

    log = [f'## {c.TEST.name} test averages:']
    for scalar in scalars:
        log.append(f'    {scalar:10}: {torch.tensor(scalars[scalar]).mean().item():.4}')
    log_str = '\n' + '\n'.join(log)

    with open(resume / 'test.txt', 'a') as f:
        f.write(log_str)
        print(log_str + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Order of arguments: gpu_idx restore_path. Restore_path should point to a directory containing the model weights "model.pt"')
    parser.add_argument('argv', nargs='*', help='gpu_idx resume')
    parser.add_argument('--test', default='cityscapes', 
                        help=f'name of test dataset. Options (includes stero image datasets): {", ".join(list(data_zoo_stereo.keys()))}')
    args = parser.parse_args()

    c.TEST.name = args.test+'#test'
    print('CONFIG:')
    pprint(c)
    print('\n')

    resume = Path(args.argv[1])
    device = torch.device("".join(["cuda:", args.argv[0]]))

    test(device, resume)