export PYTHONPATH="$PYTHONPATH:/home/sahanchamara99/bcrf-detectron/pytorch_permuto"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
python tools/train_net.py \
    --num-gpus 4 \
	--config-file configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml SOLVER.BASE_LR 0.0025 SOLVER.IMS_PER_BATCH 4
