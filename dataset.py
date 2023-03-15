import torch
from torch.utils import data
import pickle
import json
from torchvision import transforms
from transform import ToTensor,Normalize,Scale
import os
import pandas as pd
import numpy as np
from PIL import Image
from arg_pars import opt



def pil_loader(path,points=None):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if opt.feature_model=='tf_object':
                if points:
                    img=img.crop(points)   
                return img.convert('RGB')
            else:
                return img.convert('RGB')


class Breakfast(data.Dataset):
    def __init__(self,
                 seq_len=5,
                 num_seq=2,
                 downsample=3,
                 pred_step=1,
                 mode='train'):

        if opt.feature_model=='tf_object':
            Interpolation=Image.BICUBIC
        else:
            Interpolation=Image.NEAREST

        self.transform = transforms.Compose([
        Scale(size=(opt.img_dim,opt.img_dim),interpolation=Interpolation),
        ToTensor(),
        Normalize()
    ])
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.pred_step = pred_step
        self.mode=mode
        pkl_dir = opt.pkl_folder_name
        
        window_folder = pkl_dir+"/window_lists"
        if not os.path.exists(window_folder): 
            os.makedirs(window_folder)

        if opt.feature_model =='gat':
            f=open(opt.node_info_path,'rb')
            node_info=json.load(f)
            f.close()
            self.idx_map={k:v['idx'] for k, v in node_info.items()}

        print('construct windows of stride 1 from each video ...')
        window_file_list=[]
        loc_list=[]
        self.window_list_path=window_folder+'/window_list_info.pkl'
        self.loc_list_path=window_folder+'/loc_list_info.pkl'

        if os.path.exists(self.window_list_path):       
            print("skip constructing window list... use pre-computed one...")
        else:
            info_file_name=opt.video_info_file
            with open(info_file_name,'rb') as info_file:
                video_info = pickle.load(info_file)
            for vpath,info in video_info.items():
                vlen=info['true_vlen']
                P_name=vpath.split('/')[-3]
                cam_name=vpath.split('/')[-2]
                act_name=vpath.split('/')[-1].split('.')[0].split('_')[1]
                window_info = pd.DataFrame(columns=['vpath', 'vlen', 'window_idx',
                                            'all_window_seq_idx_block',
                                            'current_frame_idx'])
                window_file=window_folder+'/'+P_name+'_'+cam_name+'_'+act_name+'.pkl'
                if vlen-self.num_seq*self.seq_len*self.downsample <= 0: 
                    print(("too short of a vlen: "+str(vlen)))
                    continue
                n_window = int(vlen/self.downsample) - self.num_seq*self.seq_len + 1
                all_window_seq_idx_block = np.zeros((n_window, self.num_seq, self.seq_len))
                for window_idx in range(n_window):
                    start_zeropad = self.num_seq - 1 - self.pred_step
                    window_start_frame_idx = window_idx*self.downsample - start_zeropad*self.seq_len*self.downsample
                    seq_idx = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + window_start_frame_idx
                    tmp = seq_idx + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
                    tmp[tmp<0] = 0
                    tmp[tmp>vlen-1] = vlen-1
                    all_window_seq_idx_block[window_idx] = tmp
                    window_info.loc[window_idx] = [vpath, vlen, window_idx,
                                                        all_window_seq_idx_block[window_idx],
                                                        all_window_seq_idx_block[window_idx][self.num_seq-self.pred_step-1][0]]
                    loc_list.append(window_idx)
                    window_file_list.append(window_file)
                    f=open(window_file, "wb")
                    pickle.dump(window_info,f)
                    f.close()
                print("constuction done for %s: "%window_file)
            f=open(self.window_list_path, "wb")
            pickle.dump(window_file_list, f)
            f.close()
            f=open(self.loc_list_path, "wb")
            pickle.dump(loc_list,f )
            f.close()
        print("Done constructing windows....................")
              
    def __getitem__(self, index):
        f=open(self.window_list_path, "rb")
        window_file_list = pickle.load(f)
        if self.eval:
            window_file_list=window_file_list[:opt.eval_size]
        f.close()
        f=open(self.loc_list_path, "rb")
        loc_list=pickle.load(f)
        if self.eval:
            loc_list=loc_list[:opt.eval_size]
        f.close()
        window_folder=window_file_list[index]
        index=loc_list[index]
        f=open(window_folder, "rb")
        window_info = pickle.load(f) 
        f.close()
        vpath, vlen, window_idx, idx_block, current_frame_idx = window_info.loc[index]
        if idx_block is None: 
            print("wrong vpath: %s" % vpath)  

        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq*self.seq_len)

        v_name=vpath.split('/')[-1].split('.')[0]
        p_name=v_name.split('_')[0]
        act_name='_'.join(v_name.split('_')[1:])
        cam_name=vpath.split('/')[-2]
        dirName='_'.join([p_name,cam_name,act_name])
        vidPath = os.path.join(opt.frame_path,dirName)
        short_act_name=v_name.split('_')[1]
        sdirName='_'.join([p_name,cam_name,short_act_name])
        obPath=os.path.join(opt.object_path,sdirName+'.pkl')

        if opt.feature_model == 'tf':
            seq = [pil_loader(os.path.join(vidPath, 'Frame_%06d.jpg' % (i+1))) for i in idx_block]
            seq = self.transform(seq) 
            (C, H, W) = seq[0].size()
            seq = torch.stack(seq, 0)
            seq = seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        elif opt.feature_model == 'tf_object':
            f=open(obPath,'rb')
            video_dict=pickle.load(f)
            f.close()
            object_list=list(self.idx_map.values())
            V=len(object_list)  
            person_list=[]
            has_person=False
            for idx in idx_block:
                frame_dict=video_dict[idx]
                for object_dict in frame_dict['objects']:
                    c=object_dict['class']
                    if c == 0:
                        b=object_dict['box']
                        person_masked=pil_loader(os.path.join(vidPath, 'Frame_%06d.jpg' % (idx+1)),(b[0]/1067*320,b[1]/800*240,b[2]/1067*320,b[3]/800*240)) 
                        has_person=True
                        continue 
                if not has_person:
                    person_masked=pil_loader(os.path.join(vidPath, 'Frame_%06d.jpg' % (idx+1)))
                    person_masked=person_masked.point(lambda x: x * 0)
                person_list.append(person_masked)
                has_person=False
            person_masks = self.transform(person_list) 
            person_masks = torch.stack(person_masks, 0)
            seq=person_masks
            seq = seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)
        elif opt.feature_model=='gat':
            seq = [pil_loader(os.path.join(vidPath, 'Frame_%06d.jpg' % (i+1)),self.cfg) for i in idx_block]
            seq = self.transform(seq) 
            (C, H, W) = seq[0].size()
            seq = torch.stack(seq, 0)
            seq = seq.view(self.num_seq, self.seq_len, C, H, W)
            f=open(obPath,'rb')
            video_dict=pickle.load(f)
            f.close()
            object_list=list(self.idx_map.values())
            V=len(object_list)
            masks_class_list=[]
            points_list=[]
            masks_list=[]
            for idx in idx_block:
                frame_dict=video_dict[idx]
                object_1h=torch.tensor(np.zeros((V,1)))
                masks_class=torch.tensor([-1 for _ in range(opt.get_num)])
                points=torch.tensor(np.zeros((opt.get_num,4)))
                masks=torch.tensor(np.zeros((opt.get_num,H,W)))
                i=0
                for object_dict in frame_dict['objects']:
                    c=object_dict['class']
                    if c in object_list:
                        c=object_list.index(c)
                        b=object_dict['box']
                        score=object_dict['score']
                        if score>opt.score_threshold:
                            if object_1h[c]==0 and i<opt.get_num:
                                x1=b[0]/1067*opt.img_dim
                                y1=b[1]/800*opt.img_dim
                                x2=b[2]/1067*opt.img_dim
                                y2=b[3]/800*opt.img_dim
                                points[i]=torch.tensor([x1,x2,y1,y2])
                                x1=int(round(x1))
                                y1=int(round(y1))
                                x2=int(round(x2))
                                y2=int(round(y2))
                                masks_class[i]=c
                                masks[i,y1:y2,x1:x2]=1
                                i+=1    
                                object_1h[c]=1   
                masks_list.append(masks)
                masks_class_list.append(masks_class)
                points_list.append(points)
            points_list=torch.stack(points_list,0)
            masks_class_list=torch.stack(masks_class_list,0)
            masks_list=torch.stack(masks_list,0)
            seq=(seq,points_list,masks_class_list,V,masks_list)
        
        videoid = window_folder.split('/')[-1][:-4]
        wlen=len(window_info)
        return seq, videoid, wlen

    def __len__(self):
        f=open(self.window_list_path, "rb")
        window_file_list = pickle.load(f)
        if self.eval:
            window_file_list=window_file_list[:opt.eval_size]
        f.close()
        l=len(window_file_list)
        return l