import torch
from torch.autograd import Function
import torch
import torch.nn.functional as F
from skimage import morphology
import numpy as np

def calc_bpp(rate, image):
    H, W = image.shape[-2:]
    return rate / (H * W)

def calc_mse(target, pred):
    mse = torch.mean((target - pred) ** 2, dim=(-1, -2, -3)) * 255.0 ** 2
    if mse.shape[0] == 1:
        mse = mse[0]
    return mse

def calc_psnr(mse, eps):
    mse = F.threshold(mse, eps, eps)
    psnr = 10. * torch.log10(255. ** 2 / mse)
    return psnr

class LaplaceCDF(torch.autograd.Function):
    """
    CDF of the Laplacian distribution.
    """

    @staticmethod
    def forward(ctx, x):
        s = torch.sign(x)
        expm1 = torch.expm1(-x.abs())
        ctx.save_for_backward(expm1)
        return 0.5 - 0.5 * s * expm1

    @staticmethod
    def backward(ctx, grad_output):
        expm1, = ctx.saved_tensors
        return 0.5 * grad_output * (expm1 + 1)

def _standard_cumulative_laplace(input):
    """
    CDF of the Laplacian distribution.
    """
    return LaplaceCDF.apply(input)

def laplace_cdf(input):
    """ 
	Computes CDF of standard Laplace distribution
    """
    return _standard_cumulative_laplace(input)

class LowerBound(Function):
    """ Applies a lower bounded threshold function on to the inputs
        ensuring all scalars in the input >= bound.
        
        Gradients are propagated for values below the bound (as opposed to
        the built in PyTorch operations such as threshold and clamp)
    """

    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size(), device=inputs.device, dtype=inputs.dtype) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None

class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

# class RoundWithSTE(Function):
#     """ Rounds each element of the input tensor to nearest integer value.
#     """
#     @staticmethod
#     def forward(ctx, x):
#         return torch.round(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output

def quantise_with_noise(x):
	""" Quantise input with uniform noise [-0.5, 0.5]
	"""
	noise = torch.zeros_like(x, device=x.device).uniform_(-0.5, 0.5)
	noisy_x = x + noise
	return noisy_x

def quantise_with_ste(x, mean):
	""" Quantise input with rounding with straight through estimator (STE).
	With the STE, the gradients of the rounding operation is taken as unity.
	"""
	x = x - mean
	# x = RoundWithSTE.apply(x)
	x = x + (torch.round(x) - x).detach()
	x = x + mean
	return x

def quantise_for_entropy_and_decoder(x, mean, training):
	""" 
	Performs quantisation of the latent `x` for the entropy calculation and decoder.


	Returns:
		torch.Tensor: x_entropy, if training==True: noise quantised tensor, else STE
		torch.Tensor: x_decoder, STE quantised tensor
	"""
	if training:
		# in training it's split such that entropy uses noise, decoder uses STE
		x_entropy = quantise_with_noise(x)
		x_ste = quantise_with_ste(x, mean)
		return x_entropy, x_ste 
	else:
		# in validation both uses STE
		x_ste = quantise_with_ste(x, mean)
		x_entropy, x_decoder = x_ste, x_ste
		return x_entropy, x_decoder

def calc_rate(y_q, mean, scale, sigma_lower_bound=0.1, likelihood_lower_bound=1e-9, offset=0.5, per_channel=False):
	"""
	Rate loss estimation of quantised latent variables using the provided CDF function (default = Laplacian CDF)
	Computation is performed per batch (across, channels, height, width), i.e. return shape is [BATCH]
	"""
	scale = LowerBound.apply(scale, sigma_lower_bound)
	y_q0 = y_q - mean
	y_q0 = y_q0.abs()
	upper = laplace_cdf(( offset - y_q0) / scale)
	lower = laplace_cdf((-offset - y_q0) / scale)
	likelihood = upper - lower
	likelihood = LowerBound.apply(likelihood, likelihood_lower_bound)

	if per_channel:
		total_bits = -torch.sum(torch.log2(likelihood), dim=(-1, -2))
	else:
		total_bits = -torch.sum(torch.log2(likelihood), dim=(-1, -2, -3))
	return total_bits

def right_to_left(x_right, shift):
	out = torch.zeros(x_right.shape).to(x_right.device)
	
	for c, s in enumerate(shift):
		if s > 0:
			out[:,c,:,s:] = x_right[:,c,:,:-s]
		else:
			out[:,c,:,:] = x_right[:,c,:,:]
	return out

def left_to_right(x_left, shift):
	out = torch.zeros(x_left.shape).to(x_left.device)
	
	for c, s in enumerate(shift):
		if s > 0:
			out[:,c,:,:-s:] = x_left[:,c,:,s:]
		else:
			out[:,c,:,:] = x_left[:,c,:,:]
	return out

class GetShift:
	
	def __init__(self, max_shift=100):
		def mse(pred, target, dim=[2,3]):
			return torch.mean((pred-target)**2, dim=dim)
		
		self.criterion = mse
		self.max_shift = max_shift
		
	def __call__(self, x_left, x_right):
		assert x_left.shape[0] == 1, 'Batch size must be 1'
		c = x_left.shape[1]
		min_loss = [1e10]*c
		best_shift = [0]*c
		
		for s in range(0, self.max_shift):
			if s == 0:
				loss = self.criterion(x_left, x_right)
			else:
				loss = self.criterion(x_left[:,:,:,s:], x_right[:,:,:,:-s])
			
			for n in range(c):
				if loss[0,n] < min_loss[n]:
					min_loss[n] = loss[0,n]
					best_shift[n] = s
									
		return best_shift  

def morphologic_process(mask):
	device = mask.device
	b,_,_,_ = mask.shape

	mask = ~mask
	mask_np = mask.cpu().numpy().astype(bool)
	mask_np = morphology.remove_small_objects(mask_np, 20, 2)
	mask_np = morphology.remove_small_holes(mask_np, 10, 2)

	for idx in range(b):
		buffer = np.pad(mask_np[idx,0,:,:],((3,3),(3,3)),'constant')
		buffer = morphology.binary_closing(buffer, morphology.disk(3))
		mask_np[idx,0,:,:] = buffer[3:-3,3:-3]

	mask_np = 1-mask_np
	mask_np = mask_np.astype(float)

	return torch.from_numpy(mask_np).float().to(device)