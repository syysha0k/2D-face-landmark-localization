python3 train.py --lr 0.0001 \
	--train_list /home/zhangjianwei/syy/dataset/csv/train-little.csv \
	--val_list /home/zhangjianwei/syy/dataset/csv/train-little.csv  \
	--train_dataset /home/zhangjianwei/syy/dataset/train \
	--val_dataset /home/zhangjianwei/syy/dataset/train \
   	--save_path ./weight/checkpoint_over \
	--batch-size 19 \
	--val-batch-size 19
