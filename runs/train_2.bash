flag="--debug 0
	  --log_name fgnet_senet_lr1e4
	  --info_path ./datasets/FGNET_resized/infoNoAgeProgression.csv
	  --image_folder_path ./datasets/FGNET_resized/images/
	  --dataset fgnet
	  --batch_size 32
	  --optim Adam
	  --model senet
	  --lr 0.0001
	  --path_best ./snap/fgnet/fgnet_test_NA_senet
	  --epochs 200
	  "

python scripts/train.py $flag
