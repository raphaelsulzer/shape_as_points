import os
import glob
import time
import random
from PIL import Image
import numpy as np
import trimesh
from src.data.core import Field
from pdb import set_trace as st
from src.data.add_sensor import *
import trimesh.transformations as trafo
from scipy.interpolate import RegularGridInterpolator

class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category, rand_rot):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True

class FullPSRField(Field):
    def __init__(self, transform=None, multi_files=None):
        self.transform = transform
        # self.unpackbits = unpackbits
        self.multi_files = multi_files
    
    def load(self, model_path, idx, category, rand_rot=None):

        # try:
        # t0 = time.time()
        if self.multi_files is not None:
            psr_path = os.path.join(model_path, 'psr', 'psr_{:02d}.npz'.format(idx))
        else:
            psr_path = os.path.join(model_path, 'psr.npz')
        psr_dict = np.load(psr_path)
        # t1 = time.time()
        psr = psr_dict['psr']
        psr = psr.astype(np.float32)
        # t2 = time.time()
        # print('load PSR: {:.4f}, change type: {:.4f}, total: {:.4f}'.format(t1 - t0, t2 - t1, t2-t0))
        data = {None: psr}
        
        if self.transform is not None:
            data = self.transform(data)

        # TODO: even here I should apply rand_rot!
        # could maybe be done with this: https://medium.com/vitrox-publication/rotation-of-voxels-in-3d-space-using-python-c3b2fc0afda1

        if (rand_rot is not None):
            trans_mat_inv = np.linalg.inv(rand_rot)
            Nz, Ny, Nx = data[None].shape
            x = np.linspace(0, Nx - 1, Nx)
            y = np.linspace(0, Ny - 1, Ny)
            z = np.linspace(0, Nz - 1, Nz)
            zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
            coor = np.array([xx, yy, zz])
            coor_prime = np.tensordot(trans_mat_inv, coor, axes=((1), (0)))
            xx_prime = coor_prime[0]
            yy_prime = coor_prime[1]
            zz_prime = coor_prime[2]
            x_valid1 = xx_prime >= 0
            x_valid2 = xx_prime <= Nx - 1
            y_valid1 = yy_prime >= 0
            y_valid2 = yy_prime <= Ny - 1
            z_valid1 = zz_prime >= 0
            z_valid2 = zz_prime <= Nz - 1
            valid_voxel = x_valid1 * x_valid2 * y_valid1 * y_valid2 * z_valid1 * z_valid2
            z_valid_idx, y_valid_idx, x_valid_idx = np.where(valid_voxel > 0)
            image_transformed = np.zeros((Nz, Ny, Nx))

            data_w_coor = RegularGridInterpolator((z, y, x), data[None])
            interp_points = np.array([zz_prime[z_valid_idx, y_valid_idx, x_valid_idx],
                                         yy_prime[z_valid_idx, y_valid_idx, x_valid_idx],
                                         xx_prime[z_valid_idx, y_valid_idx, x_valid_idx]]).T
            interp_result = data_w_coor(interp_points)
            data[None][z_valid_idx, y_valid_idx, x_valid_idx] = interp_result

        return data

class PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, file_name, data_type=None, transform=None, multi_files=None, padding=0.1, scale=1.2, sensor_options=None, workers=None, cfg=None):
        self.file_name = file_name
        self.data_type = data_type # to make sure the range of input is correct
        self.transform = transform
        self.multi_files = multi_files
        self.padding = padding
        self.scale = scale
        self.sensor_options = sensor_options
        self.workers = workers
        self.cfg = cfg


    def load(self, model_path, idx, category, rand_rot):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:

            file_path = os.path.join(model_path, self.file_name)
        else:
            # num = np.random.randint(self.multi_files)
            # file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))
            file_path = os.path.join(model_path, self.file_name, 'pointcloud_%02d.npz' % (idx))

        pointcloud_dict = np.load(file_path)


        if (self.sensor_options): # sensor_options is only set for input field (not for gt_points field)
            asc = AddSensor(self.sensor_options,self.workers)
            data = asc.add(pointcloud_dict)
        else:
            data = {
                None: pointcloud_dict['points'].astype(np.float32),
                'normals': pointcloud_dict['normals'].astype(np.float32)
            }

        if(self.cfg["data"]["transform"]):
            R=np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]],dtype=np.float32)
            data[None][:,:3]=np.matmul(data[None][:,:3], R)
            data['normals']=np.matmul(data["normals"],R)
            if("gt_normals" in data):
                data['gt_normals']=np.matmul(data["gt_normals"],R)
            if("sensors" in data):
                data['sensors']=np.matmul(data["sensors"],R)

        if (self.cfg["data"]["augmentation"]):
            data[None][:, :3]=trafo.transform_points(data[None][:, :3], rand_rot).astype(np.float32)
            data['normals']=trafo.transform_points(data["normals"], rand_rot).astype(np.float32)
            if("gt_normals" in data):
                data['gt_normals']=trafo.transform_points(data["gt_normals"], rand_rot).astype(np.float32)
            if("sensors" in data):
                data['sensors']=trafo.transform_points(data["sensors"], rand_rot).astype(np.float32)


        if self.transform is not None:
            data = self.transform(data)
        
        if self.data_type == 'psr_full':
            # scale the point cloud to the range of (0, 1)
            data[None][:,:3] = data[None][:,:3] / self.scale + 0.5
            # TODO: be sure that I do not have to do this for the sensor vector
            # in doubt, I could do it to the sensor position for sure
            # so the conclusion is, rotations should be applied to sensor_vector and normal vector as well
            # translations should not!



        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete



class PointsField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    # TODO: for me: I need to generate such files for the reconbench shapes!!

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    '''
    def __init__(self, file_name, transform=None, unpackbits=True, multi_files=None, workers=None, cfg=None):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits
        self.multi_files = multi_files
        self.cfg=cfg

    def load(self, model_path, idx, category, rand_rot):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        points_dict = np.load(file_path)
        points = points_dict['points']


        if(self.cfg["data"]["transform"]):
            R=np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]],dtype=np.float32)
            points=np.matmul(points,R)

        if (self.cfg["data"]["augmentation"]):
            points = trafo.transform_points(points, rand_rot).astype(np.float32)

        #
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        data = {
            'occ_points': points,
            'occ': occupancies,
        }

        return data