export PYTHONPATH="$PYTHONPATH:/home/sahanchamara99/bcrf-detectron/pytorch_permuto"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
python -m pdb tools/train_net.py \
    --resume --num-gpus 1 \
	--config-file configs/Cityscapes-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml SOLVER.BASE_LR 0.0025 SOLVER.IMS_PER_BATCH 1 SOLVER.CHECKPOINT_PERIOD 1000 
