CITYSCAPES_PATH="/home/sahanchamara99/data/cityscapes"

mkdir annotations
python pan_city2coco.py $CITYSCAPES_PATH
python ins_city2coco.py --datadir $CITYSCAPES_PATH --outdir annotations
mkdir train
mkdir val
cp $CITYSCAPES_PATH/leftImg8bit/train/*/*.png train/
cp $CITYSCAPES_PATH/leftImg8bit/val/*/*.png val/
python prepare_panoptic_fpn.py

