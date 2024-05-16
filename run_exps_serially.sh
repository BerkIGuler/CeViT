#python3 train.py --epoch 20 --batch_size 32 --dataset_version v2 --exp_name data_v2_cevit_v1_b32_e20
#python3 train.py --epoch 20 --batch_size 64 --dataset_version v2 --exp_name data_v2_cevit_v1_b64_e20
python3 train.py --epoch 40 --batch_size 128 --dataset_version v2 --exp_name data_v2_cevit_v1_b128_e40
python3 train.py --epoch 40 --batch_size 256 --dataset_version v2 --exp_name data_v2_cevit_v1_b256_e40
#
#python3 train.py --epoch 20 --batch_size 32 --dataset_version v1 --exp_name data_v1_cevit_v1_b32_e20
#python3 train.py --epoch 20 --batch_size 64 --dataset_version v1 --exp_name data_v1_cevit_v1_b64_e20
python3 train.py --epoch 40 --batch_size 128 --dataset_version v1 --exp_name data_v1_cevit_v1_b128_e40
python3 train.py --epoch 40 --batch_size 256 --dataset_version v1 --exp_name data_v1_cevit_v1_b256_e40
