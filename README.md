# SASIC
Official code of our CVPR paper *"SASIC: Stereo Image Compression with Latent Shifts and Stereo Attention"* by Matthias Wödlinger, Jan Kotera, Jan Xu, Robert Sablatnig

## Installation

Install the necessary packages from the `requirements.txt` file with pip:

```pip install -r requirements.txt```

## Data
To use your own dataset set the paths to your dataset in the "data_zoo_stereo" dictionary in sasic/dataset.py and write a corresponding section in the "get_file_dict" method.

## Training
Train a new model with train.py. Example:

```python train.py EXP_NAME GPU_IDX --lr 0.0001 --lr_drop 500000 --epochs 500 --train cityscapes```

The model weights are saved under `experiments/EXP_NAME-HASH` (where HASH is added to prevent collisons for experiments with the same EXP_NAME).

## Testing
Test a model with test.py. Example:

```python test.py GPU_IDX RESUME```

where `RESUME` points to a directory that contains a trained `model.pt` file (in the training example above `RESUME` would be set to `experiments/EXP_NAME-HASH`).

## Citation

If you use this project please consider citing our work

```
@InProceedings{Wodlinger_2022_CVPR,
    author    = {W\"odlinger, Matthias and Kotera, Jan and Xu, Jan and Sablatnig, Robert},
    title     = {SASIC: Stereo Image Compression With Latent Shifts and Stereo Attention},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {661-670}
}
```