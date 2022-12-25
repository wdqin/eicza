flag="--debug 0
	  --log_name awe_hog
	  --info_path ./datasets/awe_resized/info.csv
	  --image_folder_path ./datasets/awe_resized/images/
	  --dataset awe
	  --model hog
	  --path_best ./snap/awe/awe_test_hog
	  --load_dataset_path ./datasets/bin/awe_resized.bin
	  "

python scripts/eval.py $flag
