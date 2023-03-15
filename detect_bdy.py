import os
import pickle
import time
import numpy as np
from scipy.ndimage import filters
from scipy import signal
from arg_pars import opt   

def lmim_boundary(predErrors, offset,order=15,threshold=51):
    predBoundaries = signal.argrelextrema(np.array(predErrors), np.less_equal, order=int(order))[0].tolist()
    start=predBoundaries[0]
    cur=predBoundaries[0]
    for i in range(1,len(predBoundaries)):
        if predBoundaries[i]-cur<=1:
            cur=predBoundaries[i]
            predBoundaries[i]=start 
        else:
            cur=predBoundaries[i]  
            start=predBoundaries[i]     
    predBoundaries=sorted(list(set(predBoundaries)))

    mean=(np.mean(predErrors[predBoundaries])*threshold+np.mean(predErrors)*(100-threshold))/100

    bdy_idx = []
    error_list=[]
    for bdy in predBoundaries:
        if predErrors[bdy]<=mean:
            bdy_idx.append(bdy*opt.ds+offset)
            error_list.append(predErrors[bdy])
    score_list=[]
    for i in range(len(predErrors)):
        score_list.append(predErrors[i]/(np.mean(predErrors)))

    return bdy_idx,score_list



def main():
    # deal with index/position
    downsample = opt.ds
    if 'tf' in opt.feature_model:
        offset=opt.seq_len
    elif opt.feature_model=='gat':
        offset = opt.seq_len-1
    offset = int(offset*downsample)

    folder_name = opt.output_path + 'OTAS/' + opt.dataset+'_'+opt.feature_model
    output_seg_dir = '/detect_seg'
    output_error_dir = '/mean_error'
    if not os.path.exists(folder_name + output_seg_dir):
        os.makedirs(folder_name + output_seg_dir)
    output_error_paths=os.listdir(folder_name+ output_error_dir)
    
    for error_path in output_error_paths:
        video_id=error_path.split('.')[0]
        output_bdy_path = folder_name + output_seg_dir + '/{}.pkl'.format(str(video_id))
        output_error_path=folder_name+ output_error_dir + '/{}.pkl'.format(str(video_id))
        f=open(output_error_path,'rb')
        mean_errors=pickle.load(f)
        f.close()
        mean_errors=mean_errors.squeeze()
        if opt.combine:
            bbox_file=opt.output_path + 'OTAS/' + opt.dataset+'_tf_object/mean_error/{}.pkl'.format(str(video_id))
            with open(bbox_file,'rb') as f:
                bbox_mean_errors=pickle.load(f)
            gat_file=opt.output_path + 'OTAS/' + opt.dataset+'_gat/mean_error/{}.pkl'.format(str(video_id))
            with open(gat_file,'rb') as f:
                gat_mean_errors=pickle.load(f)
            whole_bdy_idx_list,whole_score_list= lmim_boundary(mean_errors, 
                                                            offset=offset, 
                                                            order=opt.order,
                                                            threshold=51)
            bbox_bdy_idx_list,bbox_score_list= lmim_boundary(bbox_mean_errors, 
                                                            offset=offset, 
                                                            order=opt.order,
                                                            threshold=54)
            gat_bdy_idx_list,gat_score_list= lmim_boundary(gat_mean_errors, 
                                                            offset=offset, 
                                                            order=opt.order,
                                                            threshold=70)
            score_list=[]

            for j in range(len(whole_score_list)):
                score_list.append((opt.beta_1*whole_score_list[j]+opt.beta_2*bbox_score_list[j]+opt.beta_3*gat_score_list[j])/3)

            candidate_bdy_idx_list=list(set(bbox_bdy_idx_list+gat_bdy_idx_list))

            bdy_idx_list=[]
            remove_list=[]

            for bdy_idx in whole_bdy_idx_list:
                bdy_idx=int((bdy_idx-offset)/opt.ds)
                neighbor_list=[]
                for candidate_idx in candidate_bdy_idx_list:
                    candidate_idx=int((candidate_idx-offset)/opt.ds)
                    if abs(candidate_idx-bdy_idx)<opt.theta_n:
                        neighbor_list.append(candidate_idx)
                if len(neighbor_list)>0:#or score_list[bdy_idx]>1.6:
                    for neighbor_idx in neighbor_list:
                        if score_list[neighbor_idx]>score_list[bdy_idx]:#+2
                            bdy_idx=neighbor_idx
                        remove_list.append(neighbor_idx*opt.ds+offset)
                    bdy_idx_list.append(bdy_idx)
            remove_list=list(set(remove_list))
            for r in remove_list:
                candidate_bdy_idx_list.remove(r)

            for candidate_idx in candidate_bdy_idx_list:
                candidate_idx=int((candidate_idx-offset)/opt.ds)
                if score_list[candidate_idx]>3.9:
                    bdy_idx_list.append(candidate_idx)
            error_list=[score_list[i] for i in bdy_idx_list]
            bdy_idx_list=sorted([i*opt.ds+offset for i in bdy_idx_list])
            
        elif len(mean_errors)< 2:
            print('len mean errors <2 for %s'%video_id)
            continue
        else:
            bdy_idx_list,error_list,_ = lmim_boundary(mean_errors, offset=offset)
        bdy_idx_save = {}
        bdy_idx_save['bdy_idx_list'] = bdy_idx_list
        bdy_idx_save['error_list'] = error_list
        print('bdy_idx_list %s' %bdy_idx_list)
        f=open(output_bdy_path,'wb')
        pickle.dump(bdy_idx_save, f)
        f.close()
        print("write boundary %s"%video_id)
    
if __name__ == "__main__":
    main()

