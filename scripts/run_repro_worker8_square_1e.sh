PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" train.py --gpu 0 --num_workers 8 --max_epoch 1 --lr 1e-4 --batch_size 8 --emb_size 512 --img_size 1024 --data_root data --data_name CVOGL_DroneAerial --savename repro_worker8_square_1e_seed13 --seed 13 --beta 1.0 --print_freq 50 > logs/repro_worker8_square_1e_seed13.log 2>&1
