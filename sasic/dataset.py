import torch
import re
import numpy as np
import chardet 
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image


root_list = [
	Path('/scratch/aistream'),
	Path('/data/databases/aistream')
]
# pick the first root that works
try:
	root = [r for r in root_list if r.is_dir()][0]
except IndexError:
	root = './data'

data_zoo_stereo = {
	'cityscapes': {
		'train': root / Path('cityscapes_stereo/train'),
		'eval': root / Path('cityscapes_stereo/eval'),
		'test': root / Path('cityscapes_stereo/test')
	},
	'instereo2k': {
		'train': root / Path('instereo2k/train'),
		'test': root / Path('instereo2k/test'),
	}
}


def readPFM(file):
	""" read a PFM file and return a torch tensor with shape (h, w) """
	file = open(file, 'rb')

	color = None
	width = None
	height = None
	scale = None
	endian = None

	header = file.readline().rstrip()
	encode_type = chardet.detect(header)  
	header = header.decode(encode_type['encoding'])
	if header == 'PF':
		color = True
	elif header == 'Pf':
		color = False
	else:
		raise Exception('Not a PFM file.')

	dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode(encode_type['encoding']))
	if dim_match:
		width, height = map(int, dim_match.groups())
	else:
		raise Exception('Malformed PFM header.')

	scale = float(file.readline().rstrip().decode(encode_type['encoding']))
	if scale < 0: # little-endian
		endian = '<'
		scale = -scale
	else:
		endian = '>' # big-endian

	data = np.fromfile(file, endian + 'f')
	shape = (height, width, 3) if color else (height, width)

	data = np.reshape(data, shape)
	data = np.flipud(data)
	
	return torch.from_numpy(data.copy())


class StereoImageDataset(torch.utils.data.Dataset):
	"""Dataset class for image compression datasets."""

	def __init__(self, name, path=None, transform=None, debug=False, return_filename=False, **kwargs):
		"""
		Args:
			name (str): name of dataset, template: ds_name#ds_type. No '#' in ds_name or ds_type allowed. ds_type in (train, eval, test).
			path (str): if given the dataset is loaded from path instead of by name.
			transforms (Transform): transforms to apply to image
			debug (bool, optional): If set to true, limits the list of files to 10. Defaults to False.
		"""
		super().__init__()

		if path is None:
			assert len(name.split('#')) == 2, f'invalid name ({name}). correct template: ds_name#ds_type. No "#" in ds_name or ds_type allowed'
			
			ds_name = name.split('#')[0]
			ds_type = name.split('#')[1]
			path = data_zoo_stereo[ds_name][ds_type]
		else:
			# If dataset has been specified and no path, use path manager
			# If path has been specified with or without dataset name, use the path
			path = Path(path)
			if not path.exists:
				raise OSError("The path {} doesn't exist".format(path))
			if not path.is_dir:
				raise OSError("The path {} is not a directory".format(path))
		self.path = path

		self.name = name
		self.return_filename = return_filename
		self.ds_name = name.split('#')[0]
		self.ds_type = name.split('#')[1]
		self.transform = transform or [(lambda x: x)]
		self.transforms = transforms.Compose([*self.transform])
		self.file_dict = self.get_file_dict()
		if debug:
			for k in self.file_dict:
				self.file_dict[k] = self.file_dict[k][:10]

		print(f'Loaded dataset {name} from {path}. Found {len(self.file_dict["left_image"])} files.')

	def __len__(self):
		return len(self.file_dict[list(self.file_dict.keys())[0]])

	# Not very secure, assumes only torch.seed is relevant
	def apply_transforms(self, data_dict: dict):
		seed = torch.seed()

		for k in data_dict:
			torch.manual_seed(seed)

			data_dict[k] = self.transforms(data_dict[k])
			if not torch.is_tensor(data_dict[k]):
				data_dict[k] = transforms.ToTensor()(data_dict[k])

		return data_dict

	def __getitem__(self, idx):
		data_dict = {
			'left': Image.open(self.file_dict['left_image'][idx]),
			'right': Image.open(self.file_dict['right_image'][idx])
		}

		data_dict = self.apply_transforms(data_dict)

		if self.return_filename:
			return data_dict, str(self.file_dict['left_image'][idx])
		else:
			return data_dict, idx

	def get_file_dict(self) -> list:
		"""Get dictionary of all files in folder data_dir."""

		if self.ds_name == 'cityscapes':
			image_list = [file for file in self.path.glob(
				'**/*') if file.is_file() and file.suffix.lower() == '.png']

			# set removes duplicates due to *_disparity.png, *_rightImg8bit.png, *_leftImg8bit.png
			names = list(
				{'_'.join(str(f).split('_')[:-1]) for f in image_list})
			names.sort()

			files = {
				'left_image': [name + '_leftImg8bit.png' for name in names],
				'right_image': [name + '_rightImg8bit.png' for name in names],
				'disparity_image': [name + '_disparity.png' for name in names]
			}
		elif self.ds_name == 'instereo2k':
			folders = [f for f in self.path.iterdir() if f.is_dir()]
			left_images = [f / 'left.png' for f in folders]
			right_images = [f / 'right.png' for f in folders]

			files = {
				'left_image': left_images,
				'right_image': right_images
			}
		else:
			raise NotImplementedError

		return files
