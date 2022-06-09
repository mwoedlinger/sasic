# SASIC
Official code of our CVPR paper *"SASIC: Stereo Image Compression with Latent Shifts and Stereo Attention"* by Matthias Wödlinger, Jan Kotera, Jan Xu, Robert Sablatnig

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

<!-- ## Citation

If you use this project please consider citing our work

```
@article{
    TODO
}
``` -->