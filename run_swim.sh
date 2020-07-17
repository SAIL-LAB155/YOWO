python train.py --dataset swim_drown \
		--data_cfg cfg/swim.data \
		--cfg_file cfg/swim.cfg \
		--n_classes 3 \
		--backbone_3d resnext101 \
		--backbone_2d darknet19 \
		--backbone_2d_weights ./weights/yolo.weights \

