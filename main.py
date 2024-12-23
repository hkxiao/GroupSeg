import os
import torch
import cv2
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
import argparse
from process_feat import process_feat, clip_feat
from cosal import Cosal_Module
from utils.kmeans_pytorch import kmeans, kmeans_pp
from utils.utils import mkdir
from utils.get_segmentation_model import get_model
import random
from torch.cuda.amp import autocast

def compute_iou(preds, target): #N 1 H W
    def mask_iou(pred_label,label):
        '''
        calculate mask iou for pred_label and gt_label
        '''

        pred_label = (pred_label>0.5)[0].int()
        label = (label>0.5)[0].int()

        intersection = ((label * pred_label) > 0).sum()
        union = ((label + pred_label) > 0).sum()
        return intersection / union
    
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + mask_iou(postprocess_preds[i],target[i])
    return iou.item() / len(preds)

def str_to_bool(value):
    if value.lower() in ('yes', 'true', 't', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise ValueError("Invalid boolean value: '{}'".format(value))

def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def show_point(img, color, coord):
    radius = 8  # 点的半径
    thickness = -1  # 填充点的厚度，-1 表示填充
    cv2.circle(img, coord, radius, color, thickness)

def show_correlation(correlation_maps,save_path,name_list,tag=''):
    N,k,H,W = correlation_maps.shape            
    correlation_maps = torch.mean(correlation_maps, dim=1).flatten(-2)
    min_value = torch.min(correlation_maps, dim=1, keepdim=True)[0]
    max_value = torch.max(correlation_maps, dim=1, keepdim=True)[0]
    correlation_maps = (correlation_maps - min_value) / (max_value - min_value)  # shape=[N, HW]
    correlation_maps = correlation_maps.view(N,1,H, W) 
    correlation_maps[correlation_maps>0.5]=1
    correlation_maps[correlation_maps<0.5]=0
    correlation_maps = F.interpolate(correlation_maps,size=(256,256),mode='bilinear',align_corners=False) * 255
    
    for correlation_map,name in zip(correlation_maps,name_list):
        cv2.imwrite(os.path.join(save_path,name[:-4]+tag),correlation_map[0].cpu().numpy())
        
    return correlation_maps
      
def save(examples, outputs, update=False, tag=''):    
    for example,output in zip(examples,outputs):
        masks, low_res_logits, iou_predictions = output['masks'],output['low_res_logits'], output['iou_predictions']
        # masks = torch.nn.functional.interpolate(masks*255.0, size=(224,224), mode='bilinear',align_corners=True)
        masks = masks*255.0
        masks_np = masks[0,0,...].cpu().numpy()
        cv2.imwrite(example['save_dir']+'/'+example['name']+tag+'.png', masks_np) 
        img_np = example['image'].permute(1,2,0).cpu().numpy() 
        if 'point_coords' in example.keys():
            for point_coord,point_label in zip(example['point_coords'][0],example['point_labels'][0]):
                color = (255,0,0) if point_label.item() == 1 else (0,0,255)
                if args.only_show_positive and color == (0,0,255):
                    continue
                show_point(img_np, color, tuple(point_coord.to(torch.int).tolist()))
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(example['save_dir'], example['name'] + '_prompt.jpg'), img_np) 
        if update:
            masks = torch.nn.functional.interpolate(masks, size=(256,256), mode='bilinear',align_corners=True)
            example['mask_inputs'] = masks
        
def get_point(correlation_map, anchor_point_num, point_num=3): # N k H W
    if point_num == 0: return torch.empty([correlation_map.shape[0],0,2]).cuda()
    
    correlation_max = torch.max(correlation_map, dim=1)[0] # N H W
    ranged_index = torch.argsort(torch.flatten(correlation_max, -2), 1, descending=True) #N HW
    coords = torch.stack([ranged_index[:,:anchor_point_num]%60,ranged_index[:,:anchor_point_num]/60],-1) #N 32 2
    centers = []
    for k in range(coords.shape[0]):
        if args.cluster == 'kmeans': center = kmeans(coords[k],K=point_num, max_iters=args.kiter) #2 2
        if args.cluster == 'kmeans++': center = kmeans_pp(coords[k],K=point_num, max_iters=args.kiter) #2 2
        centers.append(center)
    max_centers = torch.stack(centers,dim=0) #N k 2
    
    return max_centers

# @torch.cuda.amp.autocast()
@torch.no_grad
def solve():
    seg_model = get_model(model=args.seg_name, model_type=args.seg_type, model_config=args.seg_config).cuda()
    Cosal = Cosal_Module()
    examples = []
    
    for i, dataset in enumerate(tqdm(args.datasets)):
        dataset_path = os.path.join(args.data_root, dataset)       
        groups = sorted(os.listdir(dataset_path+'/img'), reverse=False)
        mkdir(os.path.join(args.save_path, dataset))
        
        for j, group in enumerate(tqdm(groups)):
            print(group)
            # group = 'sealion'
            # if j==1: break
            save_group_path = os.path.join(args.save_path,dataset,group)
            mkdir(save_group_path)
            img_path = os.path.join(dataset_path, 'img', group)
            img_files = sorted(os.listdir(img_path))
            img_files = [img_file for img_file in img_files if img_file.endswith('jpg')]
            #feat_path = os.path.join(dataset_path, 'sd_raw+dino_feat', group)
            feat_path = os.path.join(dataset_path, 'sinder', group)
            feat_files = sorted(os.listdir(feat_path))
            feat_files = [feat_file for feat_file in feat_files if feat_file.endswith('pth')]
            gt_path = os.path.join(dataset_path, 'gt', group)
            gt_files = sorted(os.listdir(gt_path))
            gt_files = [gt_file for gt_file in gt_files if gt_file.endswith('png')] + [gt_file for gt_file in gt_files if gt_file.endswith('bmp')]
            features, sifts = [], []
            
            for k, file in enumerate(tqdm(feat_files)):
                # all_feat = torch.load(feat_path+'/'+file, map_location='cuda')
                # sd_feat, dino_feat = all_feat['sd_feat'], all_feat['dino_feat']
                # feature = process_feat(sd_feat,dino_feat, sd_target_dim=args.sd_pca_dims, dino_target_dim=args.vit_pca_dim, \
                #     dino_pca=args.vit_pca, using_sd=args.using_sd, using_dino=args.using_vit)

                # feature = feature.flatten(-2).permute(0,2,1).unsqueeze(0) # (1,1,H*W,C)
                # feature = clip_feat(feature, img_path = img_path+ '/' + file.replace('pth', 'jpg')) #[H W C]
                # feature = feature.permute(2,0,1).unsqueeze(0) #[1 C H W]
                # feature = F.interpolate(feature, size=(60,60), mode='bilinear', align_corners=False)

                try:
                    feature = torch.load(feat_path+'/'+file, map_location='cuda')
                except:
                    print(feat_path+'/'+file)
                    raise NameError
                feature = F.interpolate(feature.permute(2,0,1).unsqueeze(0), size=(60,60), mode='bilinear', align_corners=False)
                
                features.append(feature)

                if args.sift.lower() == 'sod':
                    sift = cv2.imread(os.path.join(args.data_root,dataset,'sod', group,file[:-4]+'.png')).astype(np.float32)
                    sift = cv2.cvtColor(sift, cv2.COLOR_BGR2GRAY)
                    sift = cv2.resize(sift, (60,60))
                    sift_torch = torch.from_numpy(sift).cuda() 
                    sifts.append(sift_torch.unsqueeze(0).unsqueeze(0)/255.0)
                
            features, sifts = torch.cat(features), torch.cat(sifts)
            # import pdb; pdb.set_trace()
            correlation, correlation_b = torch.empty(0,args.topk,60,60).cuda(), torch.empty(0,args.topk,60,60).cuda()
            
            for z in range((features.shape[0]+args.co_batch)//args.co_batch):
                co_slice = slice(z*args.co_batch, min(features.shape[0],z*args.co_batch+args.co_batch))
                # print(co_slice)
                tmp_correlation, tmp_correlation_b = Cosal(features[co_slice,...], sifts[co_slice,...], args), Cosal(features[co_slice,...], (1-sifts[co_slice,...]), args)    # N k H W
                correlation = torch.cat([correlation, tmp_correlation])
                correlation_b = torch.cat([correlation_b, tmp_correlation_b])
                
            sifts = show_correlation(correlation, save_group_path, img_files, tag='_co_'+str(k)+'.jpg')
            sifts_b = show_correlation(correlation_b, save_group_path, img_files, tag='_co_'+str(k)+'_b_.jpg')    
            
            p_coords, n_coords = get_point(correlation, args.anchor_point_num, args.positive_point_num), get_point(correlation_b, args.anchor_point_num, args.negative_point_num)
            p_coords, n_coords = p_coords*1024/60, n_coords*1024/60
            
            preds, preds_refine = torch.empty(0,1,1024,1024).cuda(), torch.empty(0,1,1024,1024).cuda()
            for k, file in enumerate(img_files):
                img = cv2.imread(img_path+'/'+file).astype(np.float32)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (1024,1024))
                img_torch = torch.from_numpy(img).permute(2,0,1).cuda()
                
                example = {}
                example['image'] = img_torch
                example['point_coords'] = torch.cat([p_coords[k],n_coords[k]]).unsqueeze(0)
                example['point_labels'] = torch.cat([torch.ones(p_coords[k].shape[0]),torch.zeros(n_coords[k].shape[0])]).unsqueeze(0).cuda().to(torch.float32)                
                example['original_size'] = (1024, 1024)
                example['save_dir'] = os.path.join(args.save_path,dataset,group) 
                example['name'] = file[:-4]
                
                examples.append(example)
                
                # if len(examples) != args.batch and (i != len(args.datasets)-1 or j != len(groups)-1 or k != len(img_files)-1):
                #     continue

                if len(examples) != args.batch and k != len(img_files)-1:
                    continue
                                           
                outputs = seg_model(examples, multimask_output=False)
                save(examples, outputs, update=True, tag='_naive')
                for output in outputs:
                    preds = torch.concat([preds, output['masks'].clone().to(torch.float32)])
                    
                outputs = seg_model(examples, multimask_output=False)
                save(examples, outputs, update=False, tag='_refine')
                for output in outputs:
                    preds_refine = torch.concat([preds_refine, output['masks'].clone().to(torch.float32)])
                examples = []

            gts = torch.empty(0,1,1024,1024).cuda()
            for k, file in enumerate(gt_files):
                gt = cv2.imread(gt_path+'/'+file, 0).astype(np.float32)
                gt = cv2.resize(gt, (1024,1024))
                gt_torch = torch.from_numpy(gt).cuda()
                gts = torch.concat([gts, gt_torch.unsqueeze(0).unsqueeze(0)])
            # print(preds.shape, gts.shape)
            iou = compute_iou(preds, gts)
            iou_refine = compute_iou(preds_refine, gts)
            print(f'{group} Naive IOU: {iou}')
            print(f'{group} Refine IOU: {iou}')

                
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    # Base
    parser.add_argument("--datasets",type=str, nargs='+',default=['CoCA', 'CoSal2015', 'CoSOD3k'])
    parser.add_argument("--co_batch",type=int,default=80)
    parser.add_argument("--batch",type=int,default=10)
    parser.add_argument("--data_root",type=str,default='data/')
    parser.add_argument("--save_root",type=str,default='work_dirs/')
    parser.add_argument("--save_prefix",type=str,default='')
    parser.add_argument("--gpu",type=int,default=1)
    parser.add_argument("--only_show_positive", action='store_true')
    
    # Stable Diffusion Setting
    parser.add_argument("--using_sd",type=str_to_bool, default='True')
    parser.add_argument("--sd_pca",type=str_to_bool, default='True')
    parser.add_argument("--sd_pca_dims", type=int, nargs='+', default=[128, 256, 384])
    
    # VIT Setting
    parser.add_argument("--using_vit",type=str_to_bool, default='true')
    parser.add_argument("--vit_category", default='dinov2')
    parser.add_argument("--vit_pca",type=str_to_bool, default='False')
    parser.add_argument("--vit_pca_dim", type=int, default=784)
    
    # Co-Representation Proxy Setting
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--sift", type=str, default='SOD')
    
    # Interaction Setting
    parser.add_argument("--dist",type=str,default='l2')
    parser.add_argument("--global_sup", type=str_to_bool, default='True')
    parser.add_argument("--sift_sup", type=str_to_bool, default='True')
    parser.add_argument("--cluster", type=str ,default='kmeans')
    parser.add_argument("--kiter", type=int, default=20)    
    parser.add_argument("--anchor_point_num", type=int, default=2) 
    parser.add_argument("--positive_point_num", type=int, default=2)  
    parser.add_argument("--negative_point_num", type=int, default=2)  
    
    # Promptable Segmentation Model Setting 
    parser.add_argument("--seg_name", type=str, default='sam')
    parser.add_argument("--seg_type", type=str, default='vit_b')
    parser.add_argument("--seg_config", type=str, default='vit_b')
    
    global args
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    fix_randseed(0)
    
    # Save Path
    args.save_path = args.save_root
    if args.save_prefix != "": args.save_path += args.save_prefix + '-'
    if args.using_sd: args.save_path += 'SD-'
    if args.using_sd and args.sd_pca: args.save_path += 'PCA-' + ('_'.join(map(str, args.sd_pca_dims))) + '-'
    if args.using_vit: args.save_path += args.vit_category.upper() + '-'
    if args.using_vit and args.vit_pca: args.save_path += 'PCA-' + str(args.vit_pca_dims) + '-'
    args.save_path += 'Proxy-' + str(args.topk) + '-' + args.sift + '-'
    args.save_path += 'Interaction-' +  str(args.dist) + '-'
    if args.global_sup: args.save_path += 'global-'
    if args.sift_sup: args.save_path += 'sift-'
    args.save_path +=  str(args.anchor_point_num) + '-' + str(args.positive_point_num) + '-' + str(args.negative_point_num) + '-' + args.cluster + '-' + str(args.kiter) + '-'
    args.save_path +=  args.seg_name.upper() + '-' + args.seg_type.upper() 
    
    solve()