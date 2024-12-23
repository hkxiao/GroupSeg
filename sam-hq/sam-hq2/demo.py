import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
# Baseline SAM2.1
# checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Ours HQ-SAM 2
checkpoint = "./checkpoints/sam2.1_hq_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
sam2 = build_sam2(model_cfg, checkpoint)
print(sam2.device)

# predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#     predictor.set_image(<your_image>)
#     masks, _, _ = predictor.predict(<input_prompts>, multimask_output=False)