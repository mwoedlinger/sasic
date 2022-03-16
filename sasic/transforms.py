class CropCityscapesArtefacts:
	"""Crop Cityscapes images to remove artefacts"""

	def __init__(self):
		self.top = 64
		self.left = 128
		self.right = 128
		self.bottom = 256

	def __call__(self, image):
		"""Crops a PIL image.

		Args:
			image (PIL.Image): Cityscapes image (or disparity map)

		Returns:
			PIL.Image: Cropped PIL Image
		"""
		w, h = image.size
		assert w == 2048 and h == 1024, f'Expected (2048, 1024) image but got ({w}, {h}). Maybe the ordering of transforms is wrong?'

		return image.crop((self.left, self.top, w-self.right, h-self.bottom))


class MinimalCrop:
	"""
	Performs the minimal crop such that height and width are both divisible by min_div.
	"""
	
	def __init__(self, min_div=16):
		self.min_div = min_div
		
	def __call__(self, image):
		w, h = image.size
		
		h_new = h - (h % self.min_div)
		w_new = w - (w % self.min_div)
		
		if h_new == 0 and w_new == 0:
			return image
		else:    
			h_diff = h-h_new
			w_diff = w-w_new

			top = int(h_diff/2)
			bottom = h_diff-top
			left = int(w_diff/2)
			right = w_diff-left

			return image.crop((left, top, w-right, h-bottom))
