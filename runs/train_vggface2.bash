flag="--debug 0
	  --log_name fgnet_vggface2_senet50_lr1e4_from_scratch_NA
	  --info_path ./datasets/FGNET_resized/infoNoAgeProgression.csv
	  --image_folder_path ./datasets/FGNET_resized/images/
	  --dataset fgnet
	  --batch_size 32
	  --optim Adam
	  --model vggface2_senet50
	  --lr 0.0001
	  --path_best ./snap/fgnet/fgnet_test_1e4_from_scratch_NA_vggface2_senet50
	  --epochs 200
	  "

python scripts/train.py $flag
