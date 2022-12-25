flag="--debug 0
	  --info_path ./datasets/FGNET_resized/infoNoAgeProgression.csv
	  --image_folder_path ./datasets/FGNET_resized/images/
	  --dataset fgnet
	  --batch_size 8
	  --model sift
	  --eval_split val+test
	  --load_dataset_path ./datasets/bin/fgnet_resized.bin
	  "

python scripts/eval.py $flag