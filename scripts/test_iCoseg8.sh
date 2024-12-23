
export CUDA_VISIBLE_DEVICES=0
python main.py --save_root 'work_dirs/' --datasets iCoseg8 --gpu 0 --batch 7 --co_batch 1000 \
    --save_prefix SINDER \
    --using_sd False \
    --sd_pca True \
    --sd_pca_dims 128 256 384 \
    --using_vit True \
    --vit_category dinov2 \
    --vit_pca False \
    --topk 32 \
    --sift SOD \
    --dist cos \
    --global_sup True \
    --sift_sup True \
    --kiter 20 \
    --cluster kmeans \
    --anchor_point_num 64 \
    --positive_point_num 6 \
    --negative_point_num 0 \
    --seg_name ASAM2 \
    --seg_type vit_t \
    --seg_config configs/sam2/sam2_hiera_t.yaml



--seg_name HQ-SAM2.1 \
--seg_type vit_l \
--seg_config configs/sam2.1/sam2.1_hq_hiera_l.yaml
