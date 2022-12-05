flag="--debug 1
	  --log_name icz_senet_lr5e5
	  --info_path ./datasets/infantCohortZambia/info.csv
	  --image_folder_path ./datasets/infantCohortZambia/jpgs/
	  --dataset icz
	  --batch_size 32
	  --optim Adam
	  --model senet
	  --lr 0.00005
	  --path_best ./snap/icz/icz_test
	  --epochs 200
	  "

python scripts/train.py $flag
