import torch
import os
import cv2
import numpy as np
import shutil

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


if __name__ == '__main__':
    gt_root = "data/iCoseg8/gt"
    # pred_root =  "data/CoSOD3k_UFO"
    pred_root = '/data2/xhk/zs-cosod/work_dirs/SINDER-DINOV2-Proxy-32-SOD-Interaction-cos-global-sift-32-2-2-kmeans-20-HQ-SAM2.1-VIT_L/iCoseg8'

    groups = sorted(os.listdir(gt_root), reverse=False)
    
    iou, cnt = 0, 0
    
    if os.path.exists(pred_root+'/iou.txt'):
        os.remove(pred_root+'/iou.txt')
    for group in groups:
        gt_path = os.path.join(gt_root, group)
        gt_files = sorted(os.listdir(gt_path))
        gt_files = [gt_file for gt_file in gt_files if gt_file.endswith('png')]
                
        gts = torch.empty(0,1,1024,1024).cuda()
        for k, file in enumerate(gt_files):
            gt = cv2.imread(gt_path+'/'+file, 0).astype(np.float32)
            gt = cv2.resize(gt, (1024,1024))
            gt_torch = torch.from_numpy(gt).cuda()
            gts = torch.concat([gts, gt_torch.unsqueeze(0).unsqueeze(0)])

        pred_path = os.path.join(pred_root, group)
        pred_files = sorted(os.listdir(pred_path))
        pred_files = [pred_file for pred_file in pred_files if pred_file.endswith('_refine.png')]
        
        if len(gt_files) != len(pred_files):
            print(group, len(gt_files), len(pred_files))
            raise NameError
        
        preds = torch.empty(0,1,1024,1024).cuda()
        for k, file in enumerate(pred_files):
            pred = cv2.imread(pred_path+'/'+file, 0).astype(np.float32)
            pred = cv2.resize(pred, (1024,1024))
            pred_torch = torch.from_numpy(pred).cuda()
            preds = torch.concat([preds, pred_torch.unsqueeze(0).unsqueeze(0)])

        group_iou = compute_iou(preds, gts)
        print(f'{group} IOU: {group_iou}')
        with open(pred_root+'/iou.txt', 'a') as f:
            f.write(f'{group} IOU: {group_iou}\n')
        iou = iou + group_iou
        cnt = cnt + 1
    
    print(f"mIOU: {iou/cnt}")
    with open(pred_root+'/iou.txt', 'a') as f:
        f.write(f'mIOU: {iou/cnt}')
