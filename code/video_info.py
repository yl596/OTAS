from arg_pars import opt
import cv2
import os
import pickle

if not os.path.exists(opt.frame_path):
    os.makedirs(opt.frame_path)

video_info={}

if opt.dataset=='BF':
    p_names=sorted(os.listdir(opt.video_path))
    for p_name in p_names:
        cam_dir=os.path.join(opt.video_path,p_name)
        cam_names=sorted(os.listdir(cam_dir))
        for cam_name in cam_names:
            video_dir=os.path.join(cam_dir,cam_name)
            video_names=sorted(os.listdir(video_dir))
            for video_name in video_names:
                if 'labels' in video_name:
                    continue
                full_act_name='_'.join(video_name.split('.')[0].split('_')[1:])
                act_name=full_act_name.split('_')[0]
                video_path=os.path.join(video_dir,video_name)
                frame_name='_'.join([p_name,cam_name,full_act_name])
                frame_dir=os.path.join(opt.frame_path,frame_name)
                if not os.path.exists(frame_dir):
                    os.makedirs(frame_dir)
                cap=cv2.VideoCapture(video_path)
                lost = []
                n=0
                vlen=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for i in range(vlen):
                    ret, frame = cap.read()
                    if ret:
                        n += 1
                        cv2.imwrite(os.path.join(frame_dir, 'Frame_%06d.jpg' % n),frame)
                    else:
                        lost.append(i)
                video_info[video_path]={'lose':lost,'true_vlen':vlen-len(lost)}
                print("Done for: %s"%video_path)
pickle.dump(video_info,open(opt.video_info_file,'wb'))