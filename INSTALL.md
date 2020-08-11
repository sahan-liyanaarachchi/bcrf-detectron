
### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.3
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- OpenCV: `pip install opencv-contrib-python` needed for cityscapes data preparation
- pycocotools: `pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
- panopticapi: `pip install git+https://github.com/cocodataset/panopticapi.git`
- cityscapesScripts: `python -m pip install cityscapesscripts`

### Build Detectron2

After having the above dependencies and gcc & g++ ≥ 5, run:
```
# clone the bcrf-detectron repo and build detectron2 from source:
git clone https://github.com/facebookresearch/detectron2.git
cd bcrf-detectron2 && python -m pip install -e .

# Or if you are on macOS
# CC=clang CXX=clang++ python -m pip install -e .
```

To __rebuild__ detectron2 that's built from a local clone, use `rm -rf build/ **/*.so` to clean the
old build first. You often need to rebuild detectron2 after reinstalling PyTorch.

### Build BCRF

After building detectron2 run the following:
```
# Go to bcrf-detectron/crf
python setup.py build_ext --inplace

# Go to bcrf-detectron/pytorch_permuto/pytorch_permuto
python setup.py install
```
Add the path to "bcrf-detectron/pytorch_permuto" the python path



