import os,sys
import torch
import time
import multiprocessing
import numpy as np
from tqdm import tqdm
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.dpsr import DPSR

data_path = '/mnt/raphael/ModelNet10/' # path for ShapeNet from ONet
multiprocess = False
njobs = 5
save_psr_field = True
resolution = 128
zero_level = 0.0
num_points = 100000
padding = 1.2

dpsr = DPSR(res=(resolution, resolution, resolution), sig=0)

def process_one(obj):


    s = obj.split('/')
    c = s[0]
    id = s[1]
    outpath = os.path.join(data_path,c,"sap",id)

    if(os.path.isfile(os.path.join(outpath, 'psr.npz'))):
        return

    os.makedirs(outpath, exist_ok=True)

    gt_path = os.path.join(data_path, c,"eval",id, 'pointcloud.npz')
    data = np.load(gt_path)
    points = data['points']
    normals = data['normals']

    # normalize the point to [0, 1)
    points = points / padding + 0.5
    # to scale back during inference, we should:
    #! p = (p - 0.5) * padding
    
    # if save_pointcloud:
    #     outdir = os.path.join(out_path_cur_obj, 'pointcloud.npz')
    #     # np.savez(outdir, points=points, normals=normals)
    #     np.savez(outdir, points=data['points'], normals=data['normals'])
    #     # return
    
    if save_psr_field:
        psr_gt = dpsr(torch.from_numpy(points.astype(np.float32))[None], 
                      torch.from_numpy(normals.astype(np.float32))[None]).squeeze().cpu().numpy().astype(np.float16)

        outdir = os.path.join(outpath, 'psr.npz')
        np.savez(outdir, psr=psr_gt)

    a=5
    

def main(c):

    for split in ['train','test']:
        print('---------------------------------------')
        print('Processing {} {}'.format(c, split))
        print('---------------------------------------')
        fname = os.path.join(data_path, c, split+'.lst')
        with open(fname, 'r') as f:
            obj_list = f.read().splitlines() 
        
        obj_list = [c+'/'+s for s in obj_list]

        if multiprocess:
            # multiprocessing.set_start_method('spawn', force=True)
            pool = multiprocessing.Pool(njobs)
            try:
                for _ in tqdm(pool.imap_unordered(process_one, obj_list), total=len(obj_list)):
                    pass
                # pool.map_async(process_one, obj_list).get()
            except KeyboardInterrupt:
                # Allow ^C to interrupt from any thread.
                exit()
            pool.close()
        else:
            for obj in tqdm(obj_list):
                process_one(obj)
        
        print('Done Processing {} {}!'.format(c, split))
        
                
if __name__ == "__main__":

    classes = os.listdir(data_path)

    t_start = time.time()
    for c in classes:
        main(c)
    
    t_end = time.time()
    print('Total processing time: ', t_end - t_start)
