# VPGNet-Pytorch
A Pytorch implementation of VPGNet

## THIS PROJECT IS STILL UNDER CONSTRUCTION
Thanks for star and issues. We are currently working on this, you can keep watching for the latest updates. THANKS!

## Update Log
Recent updates would be uploaded here in time line.

### Filelist and labeled ground truth image
Generate the file list with 5:1:1 (train:test:val):

`path/to/database:~$ python3 gen_label_v4.py`

Generate the labeled images and groundtruth image:

```python
# im parameter: 'i' for original image, 'm' for masked image
path/to/database:~$ python3 gen_image.py im
```
