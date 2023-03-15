import os
import numpy as np
from tqdm import tqdm
import pickle
from model import gat,ED_tf_predictor
from dataset import *
import torch
import torch.nn as nn
import torch.optim as optim
from arg_pars import opt

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu)
    cuda = torch.device('cuda')
    opt.pkl_folder_name = opt.output_path + 'OTAS/' + opt.dataset+'_'+opt.feature_model
    model_dir = '/model'
    model_save_path=opt.pkl_folder_name+model_dir
    

    if opt.feature_model=='tf':
        model = ED_tf_predictor(feature_dim=opt.feature_dim,mode=opt.mode,device=cuda)
        model = nn.DataParallel(model)
        model = model.to(cuda)
        if opt.mode=='val':
            print('\n===========No grad for all layers============')
            for name, param in model.named_parameters():
                param.requires_grad = False
            print('==============================================\n')
            checkpoint = torch.load(model_save_path+'/'+'lr_'+str(opt.training_lr)+"_best.pth.tar")
            model.load_state_dict(checkpoint['state_dict'])

    elif opt.feature_model=='tf_object':
        model = ED_tf_predictor(feature_dim=opt.feature_dim,mode=opt.mode,device=cuda)
        model = nn.DataParallel(model)
        model = model.to(cuda)
        print('\n===========No grad for all layers============')
        for name, param in model.named_parameters():
            param.requires_grad = False
        print('==============================================\n')
        model_save_path=opt.output_path + 'OTAS/' + opt.dataset+'_tf'+model_dir
        checkpoint = torch.load(model_save_path+'/'+'lr_'+str(opt.training_lr)+"_best.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])

    elif opt.feature_model=='gat':
        g_model = gat(feature_dim=opt.feature_dim,mode=opt.mode,num_layers=opt.num_layers,num_heads=opt.num_heads,device=cuda,pretrained=True)
        g_model = nn.DataParallel(g_model)
        g_model = g_model.to(cuda)
        if opt.mode=='val':
            print('\n===========No grad for all layers============')
            for name, param in g_model.named_parameters():
                param.requires_grad = False
            print('==============================================\n')
            checkpoint = torch.load(model_save_path+'/'+'lr_'+str(opt.training_lr)+"_best.pth.tar")
            g_model.load_state_dict(checkpoint['state_dict'],strict=False)
        f=open(opt.node_info_path,'rb')
        node_info=json.load(f)
        f.close()
        edge_info={}
        for k,v in node_info.items():
            n1=list(node_info.keys()).index(k)
            edge_info[n1]=[]
            for r_node in v['related_nodes']:
                n2=list(node_info.keys()).index(r_node)
                edge_info[n1].append(n2)
        model=(g_model,edge_info)
    else: raise ValueError('wrong model!')

    if opt.PA_mode=='val':
        ### load data ###
        val_loader = get_data(opt.mode)
        ### main loop ###
        validate_bdy(val_loader, model)
    elif opt.PA_mode=='train':
        ### load data ###
        train_loader = get_data(opt.mode)
        ### main loop ###
        train_predictor(train_loader, model)

def train_predictor(data_loader, model):
            
    criterion = nn.MSELoss()

    if opt.feature_model=='gat':
        edge_info=model[1]
        model=model[0]

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.training_lr,weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(i) for i in opt.step_size], gamma=opt.step_gamma)

    best_loss=1e10
    for epoch in range(opt.num_epoch):
        print('start epoch %d with lr %s ............................'%(epoch,str(optimizer.param_groups[0]['lr'])))
        for i, (input_seq, _, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
            if opt.feature_model=='tf':
                model.train()
                optimizer.zero_grad()
                input_seq = input_seq.to(torch.device('cuda'))
                pred_features,target_features = model(input_seq) # each output of shape [B, pred_step, D, last_size, last_size]
            elif opt.feature_model=='gat':
                model.train()
                optimizer.zero_grad()
                target_features=input_seq[0].squeeze().float().to(torch.device('cuda'))
                seq = target_features
                points_list=input_seq[1].squeeze()#(B,n,4)
                masks_class_list=input_seq[2].squeeze()#(B,n,1)
                masks_list=input_seq[4].squeeze()#(B,n,H,W)
                V=int(input_seq[3][0])
                masks_list=masks_list.float().to(torch.device('cuda'))
                points_list = points_list.float().to(torch.device('cuda'))
                masks_class_list = masks_class_list.long().to(torch.device('cuda'))
                pred_features=model((seq,masks_list,points_list,masks_class_list,V,edge_info))
            loss = criterion(pred_features, target_features)
            train_loss=loss.cpu().detach().numpy()

            if i%(int(opt.info_interval/opt.batch_size))==0 and i!=0:
                print("epoch %d  batch %d loss %.10f"%(epoch, i, train_loss))

            if i%(int(opt.ckpt_interval/opt.batch_size))==0 and i!=0:
                state = {'epoch': epoch + 1,'state_dict': model.state_dict()}
                torch.save(state, model_save_path+'/'+'lr_'+str(opt.training_lr)+'_epoch_'+str(epoch)+'_'+str(i)+"_checkpoint.pth.tar")

            if i%(int(opt.save_interval/opt.batch_size))==0  or i ==len(data_loader)-1:
                model_dir = '/model'
                model_save_path=opt.pkl_folder_name+model_dir
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                    if loss < best_loss:
                        best_loss = loss
                        torch.save(state, model_save_path+'/'+'lr_'+str(opt.training_lr)+"_best.pth.tar")
                        print("save best model with loss %.10f......................."%loss)

            loss.backward()
            optimizer.step()
            scheduler.step()
        

def validate_bdy(data_loader, model_set):
    
    downsample = opt.ds
    if 'tf' in opt.feature_model:
        offset=opt.seq_len
        model=model_set
        model.eval()
    elif opt.feature_model=='gat':
        offset = opt.seq_len-1
        model_set[0].eval()

    offset = int(offset*downsample)

    err_video=[]
    cur_wlen=0
    output_error_dir = '/mean_error'
            
    criterion = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for i, (input_seq, video_id, wlen) in tqdm(enumerate(data_loader), total=len(data_loader)):

            if opt.feature_model=='gat':
                seq=input_seq[0]
                points_list=input_seq[1]
                masks_class_list=input_seq[2]
                V=int(input_seq[3][0])
                masks_list=input_seq[4]
                pred_=[]
                for i in range(opt.seq_len):
                    seq_1=seq[:,0,i].float().to(torch.device('cuda'))
                    masks_list_1=masks_list[:,i].float().to(torch.device('cuda'))
                    points_list_1 = points_list[:,i].float().to(torch.device('cuda'))
                    masks_class_list_1 = masks_class_list[:,i].long().to(torch.device('cuda'))
                    pred_features=model_set[0]((seq_1,masks_list_1,points_list_1,masks_class_list_1,V,model_set[1]))
                    pred_.append(pred_features)
                pred_=torch.stack(pred_,1)
                pred_=pred_.unsqueeze(-1).unsqueeze(-1)
                target_=[]
                for i in range(opt.seq_len):
                    seq_2=seq[:,1,i].float().to(torch.device('cuda'))
                    masks_list_2=masks_list[:,opt.seq_len+i].float().to(torch.device('cuda'))
                    points_list_2 = points_list[:,opt.seq_len+i].float().to(torch.device('cuda'))
                    masks_class_list_2 = masks_class_list[:,opt.seq_len+i].long().to(torch.device('cuda'))
                    target_features,_=model_set[0]((seq_2,masks_list_2,points_list_2,masks_class_list_2,V,model_set[1]))
                    target_.append(target_features)
                target_=torch.stack(target_,1)
                target_=target_.unsqueeze(-1).unsqueeze(-1)

            elif 'tf' in opt.feature_model:
                input_seq = input_seq.to(torch.device('cuda'))
                pred_features,target_features= model(input_seq) 
                pred_=pred_features.unsqueeze(dim=1).unsqueeze(-1)
                target_=target_features.unsqueeze(dim=1).unsqueeze(-1)

            loss = criterion(pred_, target_)

            for idx in range(len(video_id)):
                err_video.append(loss[idx].cpu().numpy())
                cur_wlen+=1
                if cur_wlen==wlen[idx]:
                    mean_errors = -np.nanmean(np.stack(err_video, axis=0), axis=(1,2,3,4))
                    if len(mean_errors)< 2:
                        print('len mean errors <2 for %s'%video_id[idx])
                    output_error_path=opt.pkl_folder_name+ output_error_dir + '/{}.pkl'.format(str(video_id[idx]))
                    if not os.path.exists(opt.pkl_folder_name+ output_error_dir):
                        os.makedirs(opt.pkl_folder_name+ output_error_dir)
                    f=open(output_error_path,'wb')
                    pickle.dump(mean_errors, f)
                    f.close()
                    err_video=[]
                    cur_wlen=0
                    print("write boundary %s"%video_id[idx])

def get_data(mode):
    print('Loading data for "%s" ...' % mode)
    if opt.dataset == 'BF':
        dataset = Breakfast(  seq_len=opt.seq_len,
                              num_seq=opt.num_seq,
                              downsample=opt.ds,
                              pred_step=opt.pred_step,
                              mode=mode)
    else:
        raise ValueError('dataset not supported')
    if mode=='val':
        shuffle=False
    elif mode=='train':
        shuffle=True
    data_loader = data.DataLoader(dataset,
                                  batch_size=opt.batch_size,
                                  sampler=None,
                                  shuffle=shuffle,
                                  num_workers=opt.num_workers,
                                  pin_memory=True)
    
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


if __name__ == '__main__':
    main()