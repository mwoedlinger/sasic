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

where `RESUME` points to a directory that contains a trained `model.pt` file (in the training example above `RESUME` would be set to `experiments/EXP_NAME-HASH`). A pre-trained model for the cityscapes dataset and lambda=0.01 is included in `experiments/cityscapes_lambda0.01_500epochs`.

## Encoding/decoding
To save the compressed stereo image pair in a bitstream use the encode.py and decode.py python scripts.
Encoding example:

```python3 encode.py --gpu --left assets/frankfurt_000000_009291_leftImg8bit.png --right assets/frankfurt_000000_009291_rightImg8bit.png --output_filename "frankfurt_000000_009291.sasic" --model experiments/cityscapes_lambda0.01_500epochs/model.pt```

Decoding example:

```python3 decode.py --gpu --model experiments/cityscapes_lambda0.01_500epochs/model.pt --image_filename frankfurt_000000_009291.sasic```



## Examples
![image](./assets/cityscapes_example-01.png "Qualitative comparison for a sample from the Cityscapes dataset")
![image](./assets/instereo_example-02.png "Qualitative comparison for a sample from the InStereo2k dataset")

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