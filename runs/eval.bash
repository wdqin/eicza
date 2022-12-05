flag="--debug 0
	  --info_path ./datasets/infantCohortZambia/info.csv
	  --image_folder_path ./datasets/infantCohortZambia/jpgs/
	  --dataset icz
	  --batch_size 8
	  --model senet
	  --eval_split val+test
	  --path_model_eval ./snap/icz/icz_test_top1
	  "

python scripts/eval.py $flag