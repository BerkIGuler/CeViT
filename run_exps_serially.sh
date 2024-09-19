python3 train.py --max_epoch 1000 \
                 --batch_size 512 \
                 --patience 50 \
                 --test_every_n 10 \
                 --train_set train_large \
                 --val_set val_large \
                 --test_set test \
                 --lr 1e-3 \
                 --cuda 0