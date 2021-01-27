# VPGNet-Pytorch
A Pytorch implementation of VPGNet

- gen-label.py: developed, usage in file.
- test_metric_func.py: only three callable functions, don't have usable interface yet. Can be call from import.

## Filelist and labeled ground truth image
Generate the file list with 5:1:1 (train:test:val):
`path/to/database:~$ python3 gen_label_v4.py`
Generate the labeled images and groundtruth image:
```python
# im parameter: 'i' for original image, 'm' for masked image
path/to/database:~$ python3 gen_image.py im
```
