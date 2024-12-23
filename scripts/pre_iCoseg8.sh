export CUDA_VISIBLE_DEVICES=0
# python sd-dino/extract_sd_raw+dino.py --path=data/iCoseg8
# python sinder/extract_dino.py --dataset iCoseg8 --checkpoint pretrained/model.pt --visual_size 512
python3 A2S-v2/test.py cornet --gpus=2 --weight=pretrained/cornet_rgb_duts_tr.pth  --vals=ce --save --data_path=data --datasets=iCoseg8 --save_path=data