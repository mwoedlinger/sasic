import torch
import torch.nn.functional as F

class ConvLayer(torch.nn.Module):
    """ Conv layer where padding is computed to 
    keep size same is stride = 2. Reflective padding
    used to avoid boundary artifacts. 
    """
    
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding_mode='reflect', padding=None, bias=True):
        """
        Args:
            in_ch: number of input channels
            out_ch: number of output channels
            kernel_size: single int with size of kernel 
            stride: stride, use 2 for downsampling by factor of 2
            padding_mode: padding scheme used
            padding: padding to use. If `None` padding maintains resolution
            bias (bool): whether to use bias
        """
        super().__init__() 
        if padding is None:
            padding = kernel_size // 2 
        self.conv2d = torch.nn.Conv2d(in_ch, 
                                      out_ch, 
                                      kernel_size, 
                                      stride, 
                                      padding_mode=padding_mode, 
                                      padding=padding,
                                      bias=bias)
        torch.nn.init.kaiming_normal_(self.conv2d.weight)
        if bias:
            self.conv2d.bias.data.fill_(0.0)
    def forward(self, x):
        return self.conv2d(x)

class UpsampleConvLayer(torch.nn.Module):
    """ This method of upsampling alleviates checkerboard patterns,
    introduced by conv2dTranspose
    http://distill.pub/2016/deconv-checkerboard/
    """
    
    def __init__(self, in_ch, out_ch, kernel_size, stride, scale_factor, upsample_mode='nearest', padding_mode='reflect', padding=None):
        """
        Args:
            in_ch: number of input channels
            out_ch: number of output channels
            kernel_size: single int with size of kernel 
            stride: convolutional stride
            scale_factor: upsample factor
            upsample_mode: mode of upsampling. Defaults to 'nearest'
            padding_mode: mode of padding. Defaults to 'reflect'
            padding: padding to use. If `None` padding maintains resolution
        """
        super().__init__()
        self._scale_factor = scale_factor
        self._upsample_mode = upsample_mode

        if padding is None:
            padding = kernel_size // 2

        self.conv2d = torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding_mode=padding_mode, padding=padding)
        
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, mode=self._upsample_mode, scale_factor=self._scale_factor)
        return self.conv2d(x)
