import argparse
import torch
import torch.nn as nn
import torchac
import pickle
from torchvision.transforms import ToTensor, CenterCrop, ToPILImage
import PIL

from sasic.utils import *
from sasic.model import StereoEncoderDecoder


def cdf(x, loc, scale):
    return 0.5 - 0.5 * (x - loc).sign() * torch.expm1(-(x - loc).abs() / scale)


class Encoder:
    """Encode image with learned image compression model
    """

    def __init__(self, model: nn.Module, device: torch.device, L: int = 10):
        """
        Args:
            model (nn.Module): torch model on device
            device (torch.device): torch.device
            L (int, optional): Vocabulary size parameter: Half of the bin number for quantisation. 
                               Example: L = 2 means that only 5 different symbols are possible for every pixel. 
                               Defaults to 10.
        """
        super().__init__()
        model.eval()
        self.device = device
        self.L = L
        self.model = model

    def _enc(self, lq: torch.tensor, loc: torch.tensor, scale: torch.tensor) -> bytes:
        """encode a quantised latent with the arithemtic encoder

        Args:
            lq (torch.tensor): quantised latent (or hyperlatent (or hyperhyperlatent (or ...)))
            loc (torch.tensor): loc entropy parameter
            scale (torch.tensor): scale entropy parameter

        Returns:
            bytes: encoded bytes object
        """
        Lp = 2*self.L + 1

        # Get cdf (See FAQ on https://github.com/fab-jul/torchac):
        # For lq with shape [B, C, H, W] the cdf needs to have shape [B, C, H, W, L+1].
        # The final dimension gives the cdf of the corresponding symbols.
        # The symbols must be positive integers starting with zero, the last dim of the cdf
        # tensor then contains the probabilty for a symbol to be less or equal that value - 0.5.
        # For example cdf[0,0,0,0,4] is the probability that sym[0,0,0,0] is smaller or equal
        # than 3.5 so cdf[0,0,0,0,5] - cdf[0,0,0,0,4] gives the probability for sym[0,0,0,0] to be
        # exactly 4.

        lin_tens = torch.linspace(-self.L - 0.5, self.L + 0.5, Lp+1)
        lin_tens = lin_tens.reshape((1, 1, 1, 1, len(lin_tens))).expand(
            (*lq.shape, len(lin_tens)))

        scale_resh = scale.reshape(
            [*scale.shape, 1]).expand([*scale.shape, Lp+1])
        cdf_tens = cdf(lin_tens, loc=torch.zeros(
            scale_resh.shape), scale=scale_resh)

        sym = lq - loc + self.L

        # L determines the maximun value for symbols, so we clamp any symbol that is larger.
        sym = torch.clamp(sym, 0, Lp-2).short()

        return torchac.encode_float_cdf(cdf_tens, sym, needs_normalization=True)

    def encode_tensor(self, left: torch.tensor, right: torch.tensor, filename: str) -> dict:
        """encode a tensor with the model m.

        Encoding process:
            1) Get the latent tensor "l" through applying the autoencoder encoder.
            2) Apply the hyperprior encoder on the latent tensor to obtain the hyperlatent "hl".
            3) Obtain the latent entropy parameters "l_loc" and "l_scale" by applying the 
               hyperprior decoder on the quantised hyperlatent.
            4) Quantise the latent.
            5) Encode hyperlatent and latent with the arithmetic encoder using the parameters from step 3
               for the latent and the entropy paramters stored in the model for the hyperlatent.
            6) Combine the encoded latent and hyperlatent with the vocabulary size "L" and
               the shape of the hyperlatent and save.

        Args:
            x (torch.tensor): tensor that will be decoded. Expects tensor to have the shape [B, C, H, W].
            filename (str): filename of the file where the encoded bytestream is saved.

        Returns:
            dict: a dictionary containing the encoded latent and hyperlatent as well as the hyperlatent shape
                  as well as the vocabulary parameter L.
        """
        xl = left.to(self.device)
        xr = right.to(self.device)

        with torch.no_grad():
            self.model.eval()
            out = self.model(xl, xr, False)

        shift = out.shift 

        # Encode y
        ol_y, or_y = out.latents_left.y, out.latents_right.y
        yl_quantised, yl_loc, yl_scale = ol_y.value_hat, ol_y.loc, ol_y.scale
        yr_quantised, yr_loc, yr_scale = or_y.value_hat, or_y.loc, or_y.scale
        
        yl_bytes = self._enc(yl_quantised.cpu(), yl_loc.cpu(), yl_scale.cpu())
        yr_bytes = self._enc(yr_quantised.cpu(), yr_loc.cpu(), yr_scale.cpu())

        # Encode z
        ol_z, or_z = out.latents_left.z, out.latents_right.z
        zl_quantised, zl_loc, zl_scale = ol_z.value_hat, ol_z.loc, ol_z.scale
        zr_quantised, zr_loc, zr_scale = or_z.value_hat, or_z.loc, or_z.scale
        
        zl_bytes = self._enc(zl_quantised.cpu(), zl_loc.cpu(), zl_scale.cpu())
        zr_bytes = self._enc(zr_quantised.cpu(), zr_loc.cpu(), zr_scale.cpu())

        byte_dict = {
            'yl': yl_bytes,
            'zl': zl_bytes,
            'yr': yr_bytes,
            'zr': zr_bytes,
            'shift': shift,
            'z_shape': zl_quantised.shape,
            'L': self.L
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(byte_dict, f)

        return byte_dict

    def encode_PIL_Image(self, left: PIL.Image, right: PIL.Image, filename: str) -> dict:
        """encode a PIL image stereo pair with the model m

        Args:
            left/right (PIL.Image): images that will be compressed
            filename (str): filename of the file where the encoded bytestream is saved

        Returns:
            dict: a dictionary containing the encoded latent and hyperlatent as well as the hyperlatent shape
                  as well as the vocabulary parameter L.
        """
        left_tensor = ToTensor()(left).unsqueeze(0)
        right_tensor =  ToTensor()(right).unsqueeze(0)

        return self.encode_tensor(left_tensor, right_tensor, filename)

    def encode_file(self, left: str, right: str, output_filename: str) -> dict:
        left_image = PIL.Image.open(left)
        right_image = PIL.Image.open(right)

        return self.encode_PIL_Image(left_image, right_image, output_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', type=str, required=True)
    parser.add_argument('--right', type=str, required=True)
    parser.add_argument('--output_filename', type=str, default='out.sasic', help='output filename')
    parser.add_argument(
        '--model', type=str, help='A trained pytorch compression model.', required=True)
    parser.add_argument('--L', type=int, help='Vocabulary size.', default=50)
    parser.add_argument("--gpu", action='store_true', help="Use gpu?")
    args = parser.parse_args()

    if args.gpu:
        device = torch.device(f'cuda:{get_free_gpu()}')
    else:
        device = torch.device('cpu')
    print(f'  ## Using device: {device}')

    model = StereoEncoderDecoder().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    enc = Encoder(model=model, device=device, L=args.L)
    print(
        f'  ## Encode images {args.left} and {args.right} and save as {args.output_filename}')
    enc.encode_file(args.left, args.right, args.output_filename)