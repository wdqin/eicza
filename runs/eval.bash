flag="--debug 0
	  --info_path ./datasets/FGNET_resized/infoNoAgeProgression.csv
	  --image_folder_path ./datasets/FGNET_resized/images/
	  --dataset fgnet
	  --batch_size 8
	  --model vggface2_resnet50
	  --eval_split val+test
	  --path_model_loaded ./snap/fgnet/fgnet_test_1e4_from_scratch_NA_vggface2_resnet50_top1
	  "

python scripts/eval.py $flag