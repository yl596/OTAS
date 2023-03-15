import numpy as np
import os
from arg_pars import opt
import pickle

folder_name = opt.output_path + 'OTAS/' + opt.dataset+'_'+opt.feature_model
pred_path = folder_name + '/detect_seg'

def bdy_measure(pred_labels,gt_labels):
    threshold = 0.05
    
    bdy_timestamps_det=pred_labels
    num_det = len(bdy_timestamps_det)
    bdy_timestamps_gt=[]
    cur_label=gt_labels[0]
    for idx,next_label in enumerate(gt_labels):
        if cur_label!=next_label:
            cur_label=next_label
            bdy_timestamps_gt.append(idx)
    num_pos = len(bdy_timestamps_gt)
    tp = 0
    new_tp=0
    offset_arr = np.zeros((len(bdy_timestamps_gt), len(bdy_timestamps_det))) 
    for ann1_idx in range(len(bdy_timestamps_gt)):
        for ann2_idx in range(len(bdy_timestamps_det)):
            offset_arr[ann1_idx, ann2_idx] = abs(bdy_timestamps_gt[ann1_idx]-bdy_timestamps_det[ann2_idx])
    new_offset_arr=offset_arr 
    for ann1_idx in range(len(bdy_timestamps_gt)):
        if offset_arr.shape[1] == 0:
            break
        min_idx = np.argmin(offset_arr[ann1_idx, :])

        threshold_1=threshold*len(gt_labels)

        if offset_arr[ann1_idx, min_idx] <= threshold_1:
            tp += 1
            offset_arr = np.delete(offset_arr, min_idx, 1)   
    for ann1_idx in range(len(bdy_timestamps_gt)):
        if new_offset_arr.shape[1] == 0:
            break
        min_idx = np.argmin(new_offset_arr[ann1_idx, :])
        threshold_2=10*opt.ds
        if new_offset_arr[ann1_idx, min_idx] <=threshold_2:
            new_tp += 1
            new_offset_arr = np.delete(new_offset_arr, min_idx, 1) 
    fn = num_pos - tp
    fp = num_det - tp
    if num_pos == 0: 
        rec = 1
    else:
        rec = tp/(tp+fn)
    if (tp+fp) == 0: 
        prec = 0
    else: 
        prec = tp/(tp+fp)
    if (rec+prec) == 0:
        f1 = 0
    else:
        f1 = 2*rec*prec/(rec+prec)  
    return rec,prec,f1,tp,num_pos,num_det,new_tp

tp_all = 0
new_tp_all=0
num_pos_all = 0
num_det_all = 0
sum_rec_video=0
sum_prec_video=0
sum_f1_video=0

if opt.dataset=='BF':
    bdy_lists=os.listdir(pred_path)
    num_video=len(bdy_lists)
    print('total videos %s'%num_video)
    for bdy_file in bdy_lists:
        bdy_path=os.path.join(pred_path,bdy_file)
        bdy_list = pickle.load(open(bdy_path, "rb"))
        bdy_name=bdy_file.split('.')[0]
        cam_name=bdy_name.split('_')[1]
        if cam_name=='stereo':
            cam_name='stereo01'
        p_name=bdy_name.split('_')[0]
        activity_name=bdy_name.split('_')[2]
        gt_file_name=p_name+'_'+cam_name+'_'+p_name+'_'+activity_name
        gt_label_path=os.path.join(opt.gt_path,gt_file_name)
        if not os.path.exists(gt_label_path):
            print("no gt mapping for %s" %gt_file_name)
            continue
        gt_labels=np.genfromtxt(gt_label_path, delimiter="",dtype=str)
        

        rec,prec,f1,tp,num_pos,num_det,new_tp=bdy_measure(bdy_list['bdy_idx_list'],gt_labels)

        tp_all += tp
        new_tp_all+=new_tp
        num_pos_all += num_pos
        num_det_all += num_det
        sum_rec_video+=rec
        sum_f1_video+=f1
        sum_prec_video+=prec

        print('evaluation complete for %s' % gt_file_name)

fn_all = num_pos_all - tp_all
fp_all = num_det_all - tp_all
if num_pos_all == 0:
    rec_all = 1
else:
    rec_all = tp_all/(tp_all+fn_all)
if (tp_all+fp_all) == 0:
    prec_all = 0
else:
    prec_all = tp_all/(tp_all+fp_all)
if (rec_all+prec_all) == 0:
    f1_all = 0
else:
    f1_all = 2*rec_all*prec_all/(rec_all+prec_all)


new_rec_all = new_tp_all/num_pos_all
new_prec_all = new_tp_all/num_det_all
new_f1_all = 2*new_rec_all*new_prec_all/(new_rec_all+new_prec_all)


print('f1@5percent all: %s' % f1_all)
print('recall@5percent all: %s' % rec_all)
print('precision@5percent all: %s' % prec_all)
print('f1@30 all: %s' % new_f1_all)
print('recall@30 all: %s' % new_rec_all)
print('precision@30 all: %s' % new_prec_all)




