
### Training & Evaluation in Command Line

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](datasets/README.md)

The path to the "bcrf-detectron/pytorch_permuto" must be added to the python path to commence training and evaluation.
 
The two configs files for training and evaluation are;
* configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml  for COCO
* configs/Cityscapes-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml  for Cityscapes

The format of the config files is the same as detectron2.

To start training the model starting from the detectron2 initiation, download the detectron2 pretrained model from [here](https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl)
```
python tools/train_net.py --num-gpus 4 \
	--config-file configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml SOLVER.BASE_LR 0.0025 SOLVER.IMS_PER_BATCH 4
    MODEL.WEIGHTS /path/to/detectron2_pretrained/model
```
SOLVER.IMS_PER_BATCH must be equal to the number of gpus used

To start training training from another initiation change the MODELS.WEIGHTS parameter to path of the pretrained_model.
During training the checkpoints and logs will be saved to bcrf-detectron/ouputs folder.

To resume training from the previous check point,use 
```
python tools/train_net.py --num-gpus 1 \
	--config-file configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml
    --resume  SOLVER.BASE_LR 0.0025 SOLVER.IMS_PER_BATCH 1 
```
Note: To do this you must have at a checkpoint from the previous run in the bcrf-detectron/ouputs folder.

To evaluate a model's performance, use
```
./train_net.py --num-gpus 1\
	--config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
	--eval-only SOLVER.IMS_PER_BATCH 1 MODEL.WEIGHTS /path/to/checkpoint_file
```
For more options, see `./train_net.py -h`.

To speed up training and to conserve memory we downsample images before feeding to the bcrf model. The downsampling factor can be changed by adding `MODEL.BCRF_HEAD.DOWN_FACTOR <factor>` argument.
By default we have set,
* COCO: `MODEL.BCRF_HEAD.DOWN_FACTOR 2`
* CITYSCAPES `MODEL.BCRF_HEAD.DOWN_FACTOR 4`

Other useful config parameters
```
SOLVER.CHECKPOINT_PERIOD 1000  # checkpoint saving period
TEST.EVAL_PERIOD 500 # evaluation period
```
 All the default config parameters can be found in bcrf-detectron/detectron2/confing/defaults.py
 The default config parameters are modified by the config file and the arguments passed to the train_net.py script.

### Inference with Pre-trained Models

1. Download a pretrained model from [here](https://storage.cloud.google.com/bcrf-checkpoints/coco/model_0002999.pth), say coco_pretrained.pth
2. Ensure the path to the "bcrf-detectron/pytorch_permuto" folder is added to the python path
3. Add all the images that needs a panoptic segmentation to one folder say,"image_folder".
4. Run the following,
```
cd demo/
python pan_vis.py --config-file ../configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml \
  --input /path/to/"input_folder" \
  --output /path/to/save/pan/segmentation
  [--other-options]
  --opts MODEL.WEIGHTS path/to/pretrained_model
```
