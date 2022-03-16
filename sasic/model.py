import torch
import torch.nn as nn
from .utils import *
from .layers import *


class RB(nn.Module):
	def __init__(self, channels):
		super(RB, self).__init__()

		self.body = nn.Sequential(
			nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
		)
	def forward(self, x):
		return self.body(x) + x


class SAM(nn.Module):
	def __init__(self, channels):
		super(SAM, self).__init__()

		self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
		self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
		self.rb = RB(channels)
		self.softmax = nn.Softmax(-1)
		self.bottleneck = nn.Conv2d(channels * 2+1, channels, 1, 1, 0, bias=True)

	def forward(self, x_left, x_right):
		b, c, h, w = x_left.shape
		buffer_left = self.rb(x_left)
		buffer_right = self.rb(x_right)

		### M_{right_to_left}
		Q = self.b1(buffer_left).permute(0, 2, 3, 1)  # B * H * W * C
		S = self.b2(buffer_right).permute(0, 2, 1, 3)  # B * H * C * W
		score = torch.bmm(Q.contiguous().view(-1, w, c),
						  S.contiguous().view(-1, c, w))  # (B*H) * W * W
		M_right_to_left = self.softmax(score)

		score_T = score.permute(0,2,1)
		M_left_to_right = self.softmax(score_T)

		# valid mask
		V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1
		V_left_to_right = V_left_to_right.view(b, 1, h, w)  # B * 1 * H * W
		V_left_to_right = morphologic_process(V_left_to_right)
		V_right_to_left = torch.sum(M_right_to_left.detach(), 1) > 0.1
		V_right_to_left = V_right_to_left.view(b, 1, h, w)  # B * 1 * H * W
		V_right_to_left = morphologic_process(V_right_to_left)

		buffer_R = x_right.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
		buffer_l = torch.bmm(M_right_to_left, buffer_R).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W

		buffer_L = x_left.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
		buffer_r = torch.bmm(M_left_to_right, buffer_L).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W

		out_L = self.bottleneck(torch.cat((buffer_l, x_left, V_left_to_right), 1))
		out_R = self.bottleneck(torch.cat((buffer_r, x_right, V_right_to_left), 1))

		return out_L, out_R


class EntropyModelRight(nn.Module):
	"""
	Defines the layers for the encoder-decoder model. Initialisation of the
	convolutional layers is done according to He et al. (2015)
	(source: https://arxiv.org/pdf/1502.01852.pdf)
	"""

	def __init__(self, N=192, M=12):
		super(EntropyModelRight, self).__init__()
		
		# Define HyperEncoder convolution and activation layers
		self.hyper_encoder = nn.Sequential(
			ConvLayer( M, N, 3, 1), # N 64 64
			nn.PReLU(N, init=0.2),
			ConvLayer(N, N, 3, 2), # N 32 32
			nn.PReLU(N, init=0.2),
			ConvLayer(N, N, 3, 2), # N 16 16
			nn.PReLU(N, init=0.2),
			ConvLayer(N,  M, 3, 1) #  M 16 16
		)

		# Define HyperDecoder convolution and activation layers
		self.hyper_decoder = nn.Sequential(
			ConvLayer(2*M, N, 3, 1),    # N 16 16
			nn.PReLU( N, init=0.2),
			ConvLayer(N, N, 3, 1),          # N 32 32
			nn.PReLU(N, init=0.2),
			ConvLayer(N, N, 3, 1),          # N 64 64
			nn.PReLU(N, init=0.2),
			ConvLayer(N,  2*M, 3, 1)   #  24 64 64
		)

		# Define entropy parameters for hyperlatents
		self.z_loc = nn.Parameter(torch.empty((1, M, 1, 1)))
		self.z_loc.data.normal_(0.0, 1.0)

		self.z_scale = nn.Parameter(torch.empty((1, M, 1, 1)))
		self.z_scale.data.uniform_(1.0, 1.5)

	def forward(self, y, y_other_hat, training):

		# HyperEncoder: y --> z
		z = self.hyper_encoder(y)

		# Quantisation and rate loss of z --> z_hat for decoder and rate
		z_hat_for_entropy, z_hat_for_decoder = quantise_for_entropy_and_decoder(z, 
																				mean=self.z_loc,
																				training=training)

		# Rate loss of z_hat
		rz_loss = calc_rate(z_hat_for_entropy, self.z_loc, self.z_scale)

		# HyperDecoder: z_hat --> entropy parameters
		z_upscaled = nn.functional.interpolate(z_hat_for_decoder, size=y_other_hat.shape[-2:], mode='nearest')

		dec_in = torch.cat([y_other_hat, z_upscaled], dim=1)
		entropy_p = self.hyper_decoder(dec_in)

		# Break up entropy parameters into locs and scales
		y_loc, y_scale = torch.chunk(entropy_p, 2, dim=1)

		# Quantisation of y --> y_hat for decoder and rate
		y_hat_for_entropy, y_hat_for_decoder = quantise_for_entropy_and_decoder(y, 
																				mean=y_loc, 
																				training=training)

		# Rate loss of y_hat
		ry_loss = calc_rate(y_hat_for_entropy, y_loc, y_scale)
		
		ry_rate_per_channel = calc_rate(y_hat_for_entropy, y_loc, y_scale, per_channel=True)
		rz_rate_per_channel = calc_rate(z_hat_for_entropy, self.z_loc, self.z_scale, per_channel=True)

		# Prepare the model output
		y_latent = Dict(value=y, value_hat=y_hat_for_decoder, loc=y_loc, scale=y_scale)
		z_latent = Dict(value=z, value_hat=z_hat_for_decoder, loc=self.z_loc, scale=self.z_scale)
		latents = Dict(y=y_latent, z=z_latent)
		rate = Dict(y=ry_loss, y_per_channel=ry_rate_per_channel, z=rz_loss, z_per_channel=rz_rate_per_channel)

		output = Dict(y_hat=y_hat_for_decoder, rate=rate, latents=latents)

		return output


class EntropyModelLeft(nn.Module):
	"""
	Defines the layers for the encoder-decoder model. Initialisation of the
	convolutional layers is done according to He et al. (2015)
	(source: https://arxiv.org/pdf/1502.01852.pdf)
	"""

	def __init__(self, N=192, M=12):
		super(EntropyModelLeft, self).__init__()
		
		# Define HyperEncoder convolution and activation layers
		self.hyper_encoder = nn.Sequential(
			ConvLayer( M, N, 3, 1), # N 64 64
			nn.PReLU(N, init=0.2),
			ConvLayer(N, N, 3, 2), # N 32 32
			nn.PReLU(N, init=0.2),
			ConvLayer(N, N, 3, 2), # N 16 16
			nn.PReLU(N, init=0.2),
			ConvLayer(N,  M, 3, 1) #  M 16 16
		)

		# Define HyperDecoder convolution and activation layers
		self.hyper_decoder = nn.Sequential(
			ConvLayer(         M, N, 3, 1),    # N 16 16
			nn.PReLU(N, init=0.2),
			UpsampleConvLayer(N, N, 3, 1, 2), # N 32 32
			nn.PReLU(N, init=0.2),
			UpsampleConvLayer(N, N, 3, 1, 2), # N 64 64
			nn.PReLU(N, init=0.2),
			ConvLayer(        N,  2*M, 3, 1)    #  24 64 64
		)

		# Define entropy parameters for hyperlatents
		self.z_loc = nn.Parameter(torch.empty((1, M, 1, 1)))
		self.z_loc.data.normal_(0.0, 1.0)

		self.z_scale = nn.Parameter(torch.empty((1, M, 1, 1)))
		self.z_scale.data.uniform_(1.0, 1.5)

	def forward(self, y, training):

		# HyperEncoder: y --> z
		z = self.hyper_encoder(y)

		# Quantisation and rate loss of z --> z_hat for decoder and rate
		z_hat_for_entropy, z_hat_for_decoder = quantise_for_entropy_and_decoder(z, 
																				mean=self.z_loc,
																				training=training)

		# Rate loss of z_hat
		rz_loss = calc_rate(z_hat_for_entropy, self.z_loc, self.z_scale)

		# HyperDecoder: z_hat --> entropy parameters
		entropy_p = self.hyper_decoder(z_hat_for_decoder)

		# Break up entropy parameters into locs and scales
		y_loc, y_scale = torch.chunk(entropy_p, 2, dim=1)

		# Quantisation of y --> y_hat for decoder and rate
		y_hat_for_entropy, y_hat_for_decoder = quantise_for_entropy_and_decoder(y, 
																				mean=y_loc, 
																				training=training)

		# Rate loss of y_hat
		ry_loss = calc_rate(y_hat_for_entropy, y_loc, y_scale)
		
		ry_rate_per_channel = calc_rate(y_hat_for_entropy, y_loc, y_scale, per_channel=True)
		rz_rate_per_channel = calc_rate(z_hat_for_entropy, self.z_loc, self.z_scale, per_channel=True)

		# Prepare the model output
		y_latent = Dict(value=y, value_hat=y_hat_for_decoder, loc=y_loc, scale=y_scale)
		z_latent = Dict(value=z, value_hat=z_hat_for_decoder, loc=self.z_loc, scale=self.z_scale)
		latents = Dict(y=y_latent, z=z_latent)
		rate = Dict(y=ry_loss, y_per_channel=ry_rate_per_channel, z=rz_loss, z_per_channel=rz_rate_per_channel)

		output = Dict(y_hat=y_hat_for_decoder, rate=rate, latents=latents)

		return output


class StereoEncoderDecoder(nn.Module):
	"""
	Defines the layers for the encoder-decoder model. Initialisation of the
	convolutional layers is done according to He et al. (2015)
	(source: https://arxiv.org/pdf/1502.01852.pdf)
	"""

	def __init__(self, in_channels=3, N=192, M=12):
		super(StereoEncoderDecoder, self).__init__()

		self.get_shift = GetShift(max_shift=64)
		self.model_left = EntropyModelLeft()
		self.model_right = EntropyModelRight()

		# Define Encoder convolution and activation layers
		self.encoder = nn.Sequential(
			ConvLayer(  in_channels, N, 3, 1), # N 256 256
			nn.PReLU(N, init=0.2),
			ConvLayer(N, N, 3, 2), # N 128 128,
			nn.PReLU(N, init=0.2),
			ConvLayer(N, N, 3, 2), # N  64  64
			nn.PReLU(N, init=0.2),
			ConvLayer(N,  M, 3, 1) #  M  64  64
		)

		# Define Decoder convolution and activation layers
		self.sam1 = SAM(M)
		self.sam2 = SAM(N)
		self.sam3 = SAM(N)

		self.decoder_left1 = nn.Sequential(
			ConvLayer(         M, N, 3, 1),    # N  64  64
			nn.PReLU(N, init=0.2))
		self.decoder_left2 = nn.Sequential(
			UpsampleConvLayer(N, N, 3, 1, 2), # N 128 128
			nn.PReLU(N, init=0.2))
		self.decoder_left3 = nn.Sequential(
			UpsampleConvLayer(N, N, 3, 1, 2), # N 256 256
			nn.PReLU(N, init=0.2),
			ConvLayer(        N,   3, 3, 1),    #   3 256 256
		)
		self.decoder_right1 = nn.Sequential(
			ConvLayer(         M, N, 3, 1),    # N  64  64
			nn.PReLU(N, init=0.2))
		self.decoder_right2 = nn.Sequential(
			UpsampleConvLayer(N, N, 3, 1, 2), # N 128 128
			nn.PReLU(N, init=0.2))
		self.decoder_right3 = nn.Sequential(
			UpsampleConvLayer(N, N, 3, 1, 2), # N 256 256
			nn.PReLU(N, init=0.2),
			ConvLayer(        N,   3, 3, 1),    #   3 256 256
		)


	def forward(self, x_left, x_right, training):

		# Left
		y_left = self.encoder(x_left)
		out_left = self.model_left(y_left, training) # left image is compressed as always
		y_left_hat = out_left.y_hat

		# Right
		y_right = self.encoder(x_right)
		shift = self.get_shift(y_left, y_right)                     # compute shift for left -> right
		y_right_from_left = left_to_right(out_left.y_hat, shift)    # warp latent left -> right
		y_right_residual = y_right - y_right_from_left              # compute latent residual
		out_right = self.model_right(y_right_residual, y_right_from_left, training)    # apply model to residual
		y_right_hat = out_right.y_hat + y_right_from_left           # Add missing information

		# decode left and right
		l_left, l_right = self.sam1(y_left_hat, y_right_hat)
		l_left = self.decoder_left1(l_left)
		l_right = self.decoder_right1(l_right)

		l_left, l_right = self.sam2(l_left, l_right)
		l_left = self.decoder_left2(l_left)
		l_right = self.decoder_right2(l_right)

		l_left, l_right = self.sam3(l_left, l_right)
		x_hat_left = self.decoder_left3(l_left)
		x_hat_right = self.decoder_right3(l_right)

		if not training:
			warp_dict = Dict(y_right_from_left=y_right_from_left, shift=shift)
			output = Dict(pred_left=x_hat_left, rate_left=out_left.rate, latents_left=out_left.latents, warp=warp_dict,
						pred_right=x_hat_right, rate_right=out_right.rate, latents_right=out_right.latents)
		else:
			output = Dict(pred_left=x_hat_left, rate_left=out_left.rate, latents_left=out_left.latents,
						pred_right=x_hat_right, rate_right=out_right.rate, latents_right=out_right.latents)

		return output
