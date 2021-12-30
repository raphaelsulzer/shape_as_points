import os
import torch
# os.environ["NCCL_P2P_DISABLE"]="1"

from torch.utils.data import Dataset, DataLoader
import numpy as np; np.set_printoptions(precision=4)
import shutil, argparse, time, os
import pandas as pd
from collections import defaultdict
from src import config
from src.utils import mc_from_psr, export_mesh, export_pointcloud
from src.dpsr import DPSR
from src.training import Trainer
from src.model import Encode2Points
from src.utils import load_config, load_model_manual, scale2onet, is_url, load_url, compute_iou, generate_mesh
from tqdm import tqdm
from pdb import set_trace as st
from src.libmesh import check_mesh_contains

def main():
    parser = argparse.ArgumentParser(description='MNIST toy experiment')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')    
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--iter', type=int, metavar='S', help='the training iteration to be evaluated.')
    parser.add_argument('--gpu', type=str, help='which gpu to use.')

    
    args = parser.parse_args()
    cfg = load_config(args.config, 'configs/default.yaml')
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:"+args.gpu if use_cuda else "cpu")
    data_type = cfg['data']['data_type']
    input_type = cfg['data']['input_type']
    vis_n_outputs = cfg['generation']['vis_n_outputs']
    if vis_n_outputs is None:
        vis_n_outputs = -1
    # Shorthands
    out_dir = cfg['train']['out_dir']
    if not out_dir:
        os.makedirs(out_dir)
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
    out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
    out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

    # PYTORCH VERSION > 1.0.0
    assert(float(torch.__version__.split('.')[-3]) > 0)

    dataset = config.get_dataset('test', cfg, return_idx=True)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=int(cfg['train']['n_workers_val']), shuffle=False)

    model = Encode2Points(cfg).to(device)
    
    # load model
    try:
        if is_url(cfg['test']['model_file']):
            state_dict = load_url(cfg['test']['model_file'])
        elif cfg['generation'].get('iter', 0)!=0:
            state_dict = torch.load(os.path.join(out_dir, 'model', '%04d.pt'% cfg['generation']['iter']))
            generation_dir += '_%04d'%cfg['generation']['iter']
        elif args.iter is not None:
            state_dict = torch.load(os.path.join(out_dir, 'model', '%04d.pt'% args.iter))
        else:
            state_dict = torch.load(os.path.join(out_dir, 'models' ,str(cfg['train']['load_model'])+'.pt'))
            print("Loaded model from ", os.path.join(out_dir, 'models' ,str(cfg['train']['load_model'])+'.pt'))

        load_model_manual(state_dict['state_dict'], model)

    except:
        print('Model loading error. Exiting.')
        exit()
    
    
    # Generator
    generator = config.get_generator(model, cfg, device=device)
    
    # Determine what to generate
    generate_pointcloud = cfg['generation']['generate_pointcloud']
    
    # Statistics
    time_dicts = []

    # Generate
    model.eval()
    dpsr = DPSR(res=(cfg['generation']['psr_resolution'], 
                     cfg['generation']['psr_resolution'], 
                     cfg['generation']['psr_resolution']), 
                sig= cfg['generation']['psr_sigma']).to(device)

    

    # Count how many models already created
    model_counter = defaultdict(int)



    print('Generating...')
    iou = []
    for it, data in enumerate(tqdm(test_loader)):

        # Output folders
        mesh_dir = os.path.join(generation_dir, 'meshes')
        in_dir = os.path.join(generation_dir, 'input')
        pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
        generation_vis_dir = os.path.join(generation_dir, 'vis', )

        # Get index etc.
        idx = data['idx'].item()
        
        try:
            model_dict = dataset.get_model_dict(idx)
        except AttributeError:
            model_dict = {'model': str(idx), 'category': 'n/a'}

        modelname = model_dict['model']
        category_id = model_dict['category']

        c_it = model_counter[category_id]
        if c_it >= vis_n_outputs:
            continue

        try:
            category_name = dataset.metadata[category_id].get('name', 'n/a')
        except AttributeError:
            category_name = 'n/a'
        
        if category_id != 'n/a':
            mesh_dir = os.path.join(mesh_dir, str(category_id))
            pointcloud_dir = os.path.join(pointcloud_dir, str(category_id))
            in_dir = os.path.join(in_dir, str(category_id))

            folder_name = str(category_id)
            if category_name != 'n/a':
                folder_name = str(folder_name) + '_' + category_name.split(',')[0]

            generation_vis_dir = os.path.join(generation_vis_dir, folder_name)

        # Create directories if necessary
        if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
            os.makedirs(generation_vis_dir)
        
        if cfg['generation']['generate_mesh'] and not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)
        
        if generate_pointcloud and not os.path.exists(pointcloud_dir):
            os.makedirs(pointcloud_dir)
        
        if not os.path.exists(in_dir):
            os.makedirs(in_dir)

        # Timing dict
        time_dict = {
            'idx': idx,
            'class id': category_id,
            'class name': category_name,
            'modelname':modelname,
        }
        time_dicts.append(time_dict)

        # Generate outputs
        out_file_dict = {}
        
        if cfg['generation']['generate_mesh']:
            #! deploy the generator to a separate class
            out = generator.generate_mesh(data)

            v, f, points, normals, stats_dict = out


            # Write output
            mesh_out_file = os.path.join(mesh_dir, '%s.off' % modelname)
            # export_mesh(mesh_out_file, scale2onet(v), f)
            out_file_dict['mesh'] = mesh_out_file

            recon_mesh = generate_mesh(scale2onet(v), f)

            occ_points = data.get('occupancies.occ_points').squeeze().detach().cpu().numpy()
            occ_tgt = data.get('occupancies.occ').squeeze().detach().cpu().numpy()

            if len(recon_mesh.vertices) != 0 and len(recon_mesh.faces) != 0:
                occ = check_mesh_contains(recon_mesh, occ_points)
                stats_dict['iou'] = compute_iou(occ, occ_tgt)
                # iou.append(compute_iou(occ, occ_tgt))
            else:
                state_dict['iou'] = 0.0
                # iou.append(0.0)

            time_dict.update(stats_dict)

        if cfg['generation']['exp_mesh']:
            recon_mesh.export(mesh_out_file)


        if generate_pointcloud:
            pointcloud_out_file = os.path.join(
                pointcloud_dir, '%s.ply' % modelname)                
            export_pointcloud(pointcloud_out_file, scale2onet(points), normals)
            out_file_dict['pointcloud'] = pointcloud_out_file
        
        if cfg['generation']['copy_input']:
            inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
            p = data.get('inputs').to(device)
            export_pointcloud(inputs_path, scale2onet(p))
            out_file_dict['in'] = inputs_path
        
        # Copy to visualization directory for first vis_n_output samples
        # c_it = model_counter[category_id]
        # if c_it < vis_n_outputs:
        #     # Save output files
        #     img_name = '%02d.off' % c_it
        #     for k, filepath in out_file_dict.items():
        #         ext = os.path.splitext(filepath)[1]
        #         out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
        #                                 % (c_it, k, ext))
        #         shutil.copyfile(filepath, out_file)
        #
        #     # Also generate oracle meshes
        #     if cfg['generation']['exp_oracle']:
        #         points_gt = data.get('gt_points').to(device)
        #         normals_gt = data.get('gt_points.normals').to(device)
        #         psr_gt = dpsr(points_gt, normals_gt)
        #         v, f, _ = mc_from_psr(psr_gt,
        #                 zero_level=cfg['data']['zero_level'])
        #         out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
        #                                 % (c_it, 'mesh_oracle', '.off'))
        #         export_mesh(out_file, scale2onet(v), f)
        
        model_counter[category_id] += 1


    # Create pandas dataframe and save
    time_df = pd.DataFrame(time_dicts)
    time_df.set_index(['idx'], inplace=True)
    time_df.to_pickle(out_time_file)

    iou_df = time_df
    iou_df = iou_df.drop(columns=['pcl', 'dpsr', 'mc', 'total'])
    iou_df.to_pickle(os.path.join(out_dir, 'iou_all_'+cfg['data']['dataset']+'.pkl'))

    # Create pickle files  with main statistics
    time_df_class = time_df.groupby(by=['class name']).mean()
    time_df_class.loc['mean'] = time_df_class.mean()
    time_df_class.to_csv(os.path.join(out_dir, 'iou_class_' + cfg['data']['dataset'] + '.csv'))
    print("export results to ", os.path.join(out_dir, 'iou_class_' + cfg['data']['dataset'] + '.csv'))

    # Print results
    print('Timings [s]:')
    print(time_df_class)
    print(iou_df)
    # print("Mean IoU ",np.array(iou).mean())

if __name__ == '__main__':
    main()