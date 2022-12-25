flag="--debug 0
	  --info_path ./datasets/infantCohortZambia/info.csv
	  --image_folder_path ./datasets/infantCohortZambia/jpgs/
	  --dataset icz
	  --model hog
	  --eval_split val+test
	  --path_model_loaded ./snap/icz/icz_test_hog.pkl
	  "

python scripts/eval.py $flag