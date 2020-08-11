CITYSCAPES_PATH=$1

mkdir $CITYSCAPES_PATH/annotations
python pan_city2coco.py $CITYSCAPES_PATH
python ins_city2coco.py --datadir $CITYSCAPES_PATH --outdir $CITYSCAPES_PATH/annotations
mkdir $CITYSCAPES_PATH/train
mkdir $CITYSCAPES_PATH/val
cp $CITYSCAPES_PATH/leftImg8bit/train/*/*.png $CITYSCAPES_PATH/train/
cp $CITYSCAPES_PATH/leftImg8bit/val/*/*.png $CITYSCAPES_PATH/val/
python prepare_panoptic_fpn.py $CITYSCAPES_PATH

