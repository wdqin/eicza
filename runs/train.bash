flag="--debug 1
	  --log_name awe_squeezenet11_lr5e5
	  --info_path ./datasets/awe_resized/info.csv
	  --image_folder_path ./datasets/awe_resized/
	  --dataset awe
	  --batch_size 32
	  --optim Adam
	  --model squeezenet11
	  --lr 0.00005
	  --path_best ./snap/awe/awe_test_ImageNet_squeezenet11
	  --epochs 200
	  --path_model_loaded ImageNet
	  --load_dataset_path ./datasets/bin/awe_resized.bin
	  "

python scripts/train.py $flag
