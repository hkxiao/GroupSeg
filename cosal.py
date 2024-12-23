import torch
from torch.nn import functional as F
 
def resize(input, target_size=(224, 224)):
    return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)

class Cosal_Module():
    def __init__(self):
        pass

    def __call__(self, feats, SISMs=None, args=None):
        N, C, H, W = feats.shape
        NFs = F.normalize(feats, dim=1)  # [N, C, H, W]
         
        if SISMs == None: SISMs = torch.ones_like(feats).cuda()
        SISMs = resize(SISMs, [H, W]) # # [N, 1, H, W], SISMs are the saliency map
        SISMs_thd = SISMs.clone()
        SISMs_thd[SISMs_thd<=0.5] = 0
        SISMs_thd[SISMs_thd>0.5] = 1
        
        # Co_attention_maps are utilized to filter more background noise.
        def get_co_maps(co_proxy, feats):
            correlation_maps = F.conv2d(feats, weight=co_proxy)  # [N C H W] [N C 1 1] -> [N, N, H, W]

            # Normalize correlation maps.
            correlation_maps = F.normalize(correlation_maps.reshape(N, N, H*W), dim=2)  #[N, N, HW]
            co_attention_maps = torch.sum(correlation_maps , dim=1)  # shape=[N, HW]

            # Max-min normalize co-attention maps.
            min_value = torch.min(co_attention_maps, dim=1, keepdim=True)[0]
            max_value = torch.max(co_attention_maps, dim=1, keepdim=True)[0]
            co_attention_maps = (co_attention_maps - min_value) / (max_value - min_value + 1e-12)  # [N, HW]
            co_attention_maps = co_attention_maps.view(N, 1, H, W)  # [N, 1, H, W]
            return co_attention_maps

        # Find the center feat.

        # Calculate the correlation map
        if args.dist == 'cos':
            center = F.normalize((NFs * SISMs_thd).mean(dim=3).mean(dim=2), dim=1).view(N, C, 1, 1)  #[N, C, 1, 1]
            r_center = F.normalize((NFs * SISMs_thd).mean(dim=3).mean(dim=2).mean(dim=0), dim=0).view(1, C) #[1 C]
            feats_2d = NFs.reshape(N, C, H*W).permute(0, 2, 1).reshape(N*H*W, C) #[NHW C]
            correlation = torch.matmul(feats_2d, r_center.permute(1, 0)) #[NHW C] @ [1 C] -> [NHW 1]
            ranged_index = torch.argsort(correlation, dim=0, descending=True).repeat(1, C) #[NHW 1] expand-> [NHW c] 
            co_representation = torch.gather(feats_2d, dim=0, index=ranged_index)[:args.topk, :].view(args.topk, C, 1, 1) # [NHW c] -> [topk C 1 1]
            co_attention_maps = F.conv2d(NFs, co_representation)  #  [N, C, H, W]  [topk C 1 1] -> [N topk H W]

        elif args.dist == 'l2':
            distance = torch.cdist(feats_2d, r_center) #[NHW 1]
            ranged_index = torch.argsort(distance, dim=0).repeat(1, C) #[NHW 1] expand-> [NHW c] 
            co_representation = torch.gather(feats_2d, dim=0, index=ranged_index)[:args.topk, :] #[topk C]
            co_attention_maps = torch.cdist(feats_2d, co_representation).view(N, H, W, args. topk).permute(0, 3, 1, 2) #[NHW c] #[topk C]->[NHW topk]->[N topk H W]

            # Max-min normalize co-attention maps.
            min_value = torch.min(co_attention_maps, dim=1, keepdim=True)[0]
            max_value = torch.max(co_attention_maps, dim=1, keepdim=True)[0]
            co_attention_maps = (co_attention_maps - min_value) / (max_value - min_value + 1e-12)
            
        if args.global_sup: #为什么这里对cosal2015和cosod3k有用？
            glabal_attention_maps = get_co_maps(center, NFs)  # [N, 1, H, W]
            co_attention_maps = co_attention_maps * glabal_attention_maps
        if args.sift_sup: co_attention_maps = co_attention_maps * SISMs_thd
        
        return co_attention_maps
