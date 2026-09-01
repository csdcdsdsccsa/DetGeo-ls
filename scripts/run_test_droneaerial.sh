python train.py --detector_head yolo --val --pretrain saved_models/model_droneaerial_yolo_bs8_model_best.pth.tar --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --savename test_model_droneaerial_yolo --gpu 0 --batch_size 8 --num_workers 8 --print_freq 50

python train.py --detector_head yolo --test --pretrain saved_models/model_droneaerial_yolo_bs8_model_best.pth.tar --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --savename test_model_droneaerial_yolo --gpu 0 --batch_size 8 --num_workers 8 --print_freq 50
