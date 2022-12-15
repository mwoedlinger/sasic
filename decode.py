import argparse
import torch
import torch.nn as nn
import torchac
import pickle
from torchvision import transforms
import PIL

from sasic.utils import *
from sasic.model import StereoEncoderDecoder


def cdf(x, loc, scale):
    return 0.5 - 0.5 * (x - loc).sign() * torch.expm1(-(x - loc).abs() / scale)


class Decoder:
    """Decode image with learned image compression model
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Args:
            model (nn.Module): torch model on device
            device (torch.device): torch.device
        """
        super().__init__()

        self.device = device

        self.hd_left = model.model_left.hyper_decoder
        self.hd_right = model.model_right.hyper_decoder

        self.zl_loc = model.model_left.z_loc
        self.zl_scale = model.model_left.z_scale
        self.zr_loc = model.model_right.z_loc
        self.zr_scale = model.model_right.z_scale

        self.sam1 = model.sam1
        self.sam2 = model.sam2
        self.sam3 = model.sam3

        self.decoder_left1 = model.decoder_left1
        self.decoder_left2 = model.decoder_left2
        self.decoder_left3 = model.decoder_left3        
        self.decoder_right1 = model.decoder_right1
        self.decoder_right2 = model.decoder_right2
        self.decoder_right3 = model.decoder_right3

    def _dec(self, byte_stream: bytes, loc: torch.tensor, scale: torch.tensor, L: int) -> torch.tensor:
        """decode a bytes object with the arithemtic encoder

        Args:
            byte_stream (bytes): encoded bytes object
            loc (torch.tensor): loc entropy parameter
            scale (torch.tensor): scale entropy parameter

        Returns:
            torch.tensor: quantised latent (or hyperlatent (or hyperhyperlatent (or ...)))
        """
        Lp = 2*L + 1

        lin_tens = torch.linspace(-L - 0.5, L + 0.5, Lp+1)
        lin_tens = lin_tens.reshape((1, 1, 1, 1, len(lin_tens))).expand(
            (*loc.shape, len(lin_tens)))

        scale_resh = scale.reshape(
            [*scale.shape, 1]).expand([*scale.shape, Lp+1])
        cdf_tens = cdf(lin_tens, loc=torch.zeros(
            scale_resh.shape), scale=scale_resh)

        sym = torchac.decode_float_cdf(cdf_tens, byte_stream)
        # If you compute lq = loc + sym - L you will get a small error due to numerical stability issues.
        # Computing lq = sym - L + loc instead fixes this, however we did the computation in double precision,
        # just in case ...
        # Neglecting these numerical errors leads to errors for the predicted latent entropy parameters
        # which in turn leads to completely wrong decoded latent because the arithmetic decoder is very+
        # sensitive to noise.
        lq = sym.double() - L + loc.double()

        return lq.float()

    def decode_tensor(self, filename: str) -> torch.tensor:
        """decode a compressed image file as a torch.tensor with the model m

        Args:
            filename (str): filename of encoded image

        Returns:
            torch.tensor: decoded torch.tensor
        """
        with open(filename, 'rb') as f:
            byte_dict = pickle.load(f)

        yl_bytes = byte_dict['yl']
        zl_bytes = byte_dict['zl']
        yr_bytes = byte_dict['yr']
        zr_bytes = byte_dict['zr']
        shift = byte_dict['shift']
        z_shape = byte_dict['z_shape']
        L = byte_dict['L']

        zl_loc = self.zl_loc.expand(z_shape)
        zr_loc = self.zr_loc.expand(z_shape)
        zl_scale = self.zl_scale.expand(z_shape)
        zr_scale = self.zr_scale.expand(z_shape) 
        
        zl_quant = self._dec(zl_bytes, zl_loc.cpu(), zl_scale.cpu(), L).to(self.device)
        yl_probs = self.hd_left(zl_quant)
        yl_loc, yl_scale = torch.chunk(yl_probs, 2, dim=1)
        yl_quant = self._dec(yl_bytes, yl_loc.cpu(), yl_scale.cpu(), L).to(self.device)
        
        y_right_from_left = left_to_right(yl_quant, shift)

        zr_quant = self._dec(zr_bytes, zr_loc.cpu(), zr_scale.cpu(), L).to(self.device)
        zr_upscaled = nn.functional.interpolate(zr_quant, size=yl_quant.shape[-2:], mode='nearest')
        hd_right_in = torch.cat([y_right_from_left, zr_upscaled], dim=1)
        yr_probs = self.hd_right(hd_right_in)
        yr_loc, yr_scale = torch.chunk(yr_probs, 2, dim=1)     

        yr_quant_res = self._dec(yr_bytes, yr_loc.cpu(), yr_scale.cpu(), L).to(self.device)
        yr_quant = yr_quant_res + y_right_from_left

        yl_quant = yl_quant.to(device)
        yr_quant = yr_quant.to(device)

        # decode left and right
        l_left, l_right = self.sam1(yl_quant, yr_quant)
        l_left = self.decoder_left1(l_left)
        l_right = self.decoder_right1(l_right)

        l_left, l_right = self.sam2(l_left, l_right)
        l_left = self.decoder_left2(l_left)
        l_right = self.decoder_right2(l_right)

        l_left, l_right = self.sam3(l_left, l_right)
        x_hat_left = self.decoder_left3(l_left)
        x_hat_right = self.decoder_right3(l_right)

        return torch.clamp(x_hat_left, 0, 1), torch.clamp(x_hat_right, 0, 1)

    def decode_PIL_Image(self, filename: str) -> PIL.Image:
        """decode a compressed image file as a PIL.Image with the model m

        Args:
            filename (str): filename of encoded image

        Returns:
            PIL.Image: decoded PIL.Image
        """
        xl_hat, xr_hat = self.decode_tensor(filename)
        left = transforms.ToPILImage()(xl_hat[0])
        right = transforms.ToPILImage()(xr_hat[0])

        return left, right


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_filename', type=str,
                        help='Compressed image file')
    parser.add_argument('--output_left', type=str, help='output filename left', default='left')
    parser.add_argument('--output_right', type=str, help='output filename left', default='right')
    parser.add_argument(
        '--model', type=str, help='A trained pytorch compression model.', default='model.pt')
    parser.add_argument("--gpu", action='store_true', help="Use gpu?")
    args = parser.parse_args()

    if args.gpu:
        device = torch.device(f'cuda:{get_free_gpu()}')
    else:
        device = torch.device('cpu')
    print(f'  ## Using device: {device}')

    model = StereoEncoderDecoder().to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    dec = Decoder(model=model, device=device)

    print(f'  ## Decode image {args.image_filename}')
    dec_image_left, dec_image_right = dec.decode_PIL_Image(args.image_filename)

    print(f'  ## Save image as {args.output_left}, {args.output_right}')
    dec_image_left.save(args.output_left+'.png', 'PNG')
    dec_image_right.save(args.output_right+'.png', 'PNG')