import sys, os
import numpy as np
# import open3d as o3d

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.network.utils import normalize_3d_coordinate
from scipy.spatial import cKDTree

class AddSensor():
    def __init__(self, sensor_options, workers):
        self.sensor_options = sensor_options
        self.workers= workers


    def add(self,pointcloud_dict):

        if(self.sensor_options["mode"] == "norm+"):
            if(self.sensor_options["sampling"] == 'uniform'):
                if(self.sensor_options["factor"] == 'los'):
                    data = self.add_uniform_los(pointcloud_dict)
                elif(self.sensor_options["factor"] == "neighborhood"):
                    data = self.add_uniform_neighborhood(pointcloud_dict)
                else:
                    print("ERROR: no valid factor for auxiliary sensor information points.")
                    sys.exit(1)
            elif(self.sensor_options["sampling"] == 'non-uniform'):
                data = self.add_non_uniform(pointcloud_dict)
            else:
                print("ERROR: no valid sampling strategy for auxiliary sensor information points.")
                sys.exit(1)
        elif(self.sensor_options["mode"] == "read"):
            points = pointcloud_dict['points'].astype(np.float32)
            normals = pointcloud_dict['normals']
            normal_zeros = np.zeros(shape=(points.shape[0]-normals.shape[0],3))
            normals = np.concatenate((normals,normal_zeros)).astype(np.float32)

            gt_normals = pointcloud_dict['gt_normals']
            normal_zeros = np.zeros(shape=(points.shape[0]-gt_normals.shape[0],3))
            gt_normals = np.concatenate((gt_normals,normal_zeros)).astype(np.float32)

            if 'sensor_position' in pointcloud_dict.files:
                sensors = pointcloud_dict['sensor_position'].astype(np.float32)
            elif 'sensors' in pointcloud_dict.files:
                sensors = pointcloud_dict['sensors'].astype(np.float32)
            else:
                print('no sensor infor in file')
                sys.exit(1)
            data = {
                None: points,
                'normals': normals,
                'gt_normals': gt_normals,
                'sensors': sensors,
            }
        elif(self.sensor_options["mode"] == "sensor_vec_norm"):
            points = pointcloud_dict['points'].astype(np.float32)[:,:3]
            normals = pointcloud_dict['normals'].astype(np.float32)
            if('gt_normals' in pointcloud_dict):
                gt_normals = pointcloud_dict['gt_normals'].astype(np.float32)[:]
            else:
                gt_normals = np.zeros(shape=points.shape)

            if 'sensor_position' in pointcloud_dict.files:
                sensors = pointcloud_dict['sensor_position'].astype(np.float32)
            elif 'sensors' in pointcloud_dict.files:
                sensors = pointcloud_dict['sensors'].astype(np.float32)
            else:
                print('no sensor infor in file')
                sys.exit(1)
            sensors = sensors - points
            sensors = sensors / np.linalg.norm(sensors, axis=1)[:, np.newaxis]

            data = {
                None: points,
                'normals': normals,
                'gt_normals': gt_normals,
                'sensors': sensors,
            }
        else:
            points = pointcloud_dict['points'].astype(np.float32)
            points = points[:,:3]
            normals = pointcloud_dict['normals'].astype(np.float32)
            if ('gt_normals' in pointcloud_dict):
                gt_normals = pointcloud_dict['gt_normals'].astype(np.float32)[:]
            else:
                gt_normals = np.zeros(shape=points.shape)
            if ('sensors' in pointcloud_dict):
                sensors = pointcloud_dict['sensors'].astype(np.float32)
            else:
                sensors = np.zeros(shape=points.shape)
            data = {
                None: points,
                'normals': normals,
                'gt_normals': gt_normals,
                'sensors': sensors,
            }

        return data


    def add_uniform_los(self, pointcloud_dict):
        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)
        gt_normals = pointcloud_dict['gt_normals'].astype(np.float32)
        sensors = pointcloud_dict['sensors'].astype(np.float32)

        # make the sensor vec
        sensor_vec = sensors - points
        # normalize the sensors
        sensor_vec_norm = sensor_vec / np.linalg.norm(sensor_vec, axis=1)[:, np.newaxis]
        # add the point identifier
        points = np.concatenate((points, np.zeros(shape=(points.shape[0], 2), dtype=np.float32)), axis=1)
        p_dim = points.shape[0]
        for step in self.sensor_options["steps"]:

            apoints = points[:, :3] + step * sensor_vec
            if step > 0:  # full = 01
                apoints = np.concatenate(
                    (apoints, np.zeros(shape=(p_dim, 1), dtype=np.float32), np.ones(shape=(p_dim, 1), dtype=np.float32)),
                    axis=1)
            else:  # empty = 10
                apoints = np.concatenate(
                    (apoints, np.ones(shape=(p_dim, 1), dtype=np.float32), np.zeros(shape=(p_dim, 1), dtype=np.float32)),
                    axis=1)
            points = np.concatenate((points, apoints))
            normals = np.concatenate((normals, normals))
            gt_normals = np.concatenate((gt_normals, gt_normals))
            sensor_vec_norm = np.concatenate((sensor_vec_norm, sensor_vec_norm))

        data = {
            None: points,
            'normals': normals,
            'gt_normals': gt_normals,
            'sensors': sensor_vec_norm,
        }

        return data


    def add_uniform_neighborhood(self, pointcloud_dict):
        # make los-points that are close ( <= average neighborhoodsize) to end point of los

        # take mean of this vector: factor = np.array(o3d.geometry.PointCloud.compute_nearest_neighbor_distance(pc)).mean()
        # make sensor vector a unit vector and then do:
        # sampled_point = points + norm_sensor_vec * factor

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)
        if 'gt_normals' in pointcloud_dict.files:
            gt_normals = pointcloud_dict['gt_normals'].astype(np.float32)
        else:
            gt_normals = np.zeros(shape=points.shape)
        if 'sensor_position' in pointcloud_dict.files:
            sensors = pointcloud_dict['sensor_position'].astype(np.float32)
        elif 'sensors' in pointcloud_dict.files:
            sensors = pointcloud_dict['sensors'].astype(np.float32)
        else:
            print('no sensor infor in file')
            sys.exit(1)

        # get the factor for where to put the point
        # pc = o3d.geometry.PointCloud()
        # pc.points = o3d.utility.Vector3dVector(points)
        # mean_dist = np.array(o3d.geometry.PointCloud.compute_nearest_neighbor_distance(pc)).mean()

        tree = cKDTree(points)
        mean_dist=tree.query(points,k=2,workers=self.workers)[0][:,1].mean()

        # make the sensor vec
        sensor_vec = sensors - points
        # normalize the sensors
        sensor_vec_norm = sensor_vec / np.linalg.norm(sensor_vec, axis=1)[:, np.newaxis]
        # add the point identifier
        points = np.concatenate((points, np.zeros(shape=(points.shape[0], 2), dtype=np.float32)), axis=1)

        opoints = []
        ipoints = []
        for i in self.sensor_options["stepsi"]:
            ipoints.append(points[:,:3] + i * mean_dist * sensor_vec_norm)
        for o in self.sensor_options["stepso"]:
            opoints.append(points[:, :3] + o * mean_dist * sensor_vec_norm)

        opoints = np.array(opoints).reshape(points.shape[0]*len(self.sensor_options["stepso"]),3)
        ipoints = np.array(ipoints).reshape(points.shape[0]*len(self.sensor_options["stepsi"]),3)

        opoints = np.concatenate((opoints, np.zeros(shape=(opoints.shape[0], 1), dtype=np.float32),
                                  np.ones(shape=(opoints.shape[0], 1), dtype=np.float32)), axis=1)
        ipoints = np.concatenate((ipoints, np.ones(shape=(ipoints.shape[0], 1), dtype=np.float32),
                                  np.zeros(shape=(ipoints.shape[0], 1), dtype=np.float32)), axis=1)

        points = np.concatenate((points, opoints, ipoints))
        normals = np.repeat(normals, len(self.sensor_options["stepso"]) + len(self.sensor_options["stepsi"]) + 1, axis=0)
        gt_normals = np.repeat(gt_normals, len(self.sensor_options["stepso"]) + len(self.sensor_options["stepsi"]) + 1, axis=0)
        sensor_vec_norm = np.repeat(sensor_vec_norm, len(self.sensor_options["stepso"]) + len(self.sensor_options["stepsi"]) + 1, axis=0)

        assert (points.shape[0] == normals.shape[0] == gt_normals.shape[0] == sensor_vec_norm.shape[0])

        data = {
            None: points,
            'normals': normals,
            'gt_normals': gt_normals,
            'sensors': sensor_vec_norm,
        }

        return data


    def add_non_uniform(self, pointcloud_dict):
        # t0 = time.time()
        res = self.sensor_options["grid_res"]

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)
        gt_normals = pointcloud_dict['gt_normals'].astype(np.float32)
        if 'sensor_position' in pointcloud_dict.files:
            sensors = pointcloud_dict['sensor_position'].astype(np.float32)
        elif 'sensors' in pointcloud_dict.files:
            sensors = pointcloud_dict['sensors'].astype(np.float32)
        else:
            print('no sensor infor in file')
            sys.exit(1)

        npoints = normalize_3d_coordinate(points, padding=0.1)
        pindex = (npoints * res).astype(int)

        pgrid = np.zeros(shape=(res, res, res), dtype=bool)

        # apply buffering / dilation, with 5x5x5 kernel, and active the pgrid voxels
        # this could maybe be sped up by using openCV: dilation = cv2.dilate(img,kernel,iterations = 1)
        a = -self.sensor_options["grid_kernel"]
        b = self.sensor_options["grid_kernel"] + 1
        temp = np.arange(a, b)
        kernel = np.array(np.meshgrid(temp, temp, temp)).T.reshape(-1, 3)  # 5x5x5 Kernel
        for k in kernel:
            pgrid[pindex[:, 0] + k[0], pindex[:, 1] + k[1], pindex[:, 2] + k[2]] = True

        sensor_vecs = sensors - points
        sensor_vecs = sensor_vecs / np.linalg.norm(sensor_vecs, axis=1)[:, np.newaxis]

        n = 50
        steps = np.expand_dims(np.linspace(0.01, 0.5, n), axis=1)

        ## inside:
        m = 2
        npoints = np.repeat(points, m, axis=0)
        ident = np.arange(points.shape[0])
        ident = np.repeat(ident, m, axis=0)
        ident = np.expand_dims(ident, axis=1)
        nsensors = np.repeat(sensor_vecs, m, axis=0)
        nsteps = np.tile(steps[:m], [points.shape[0], 3])
        in_points = npoints - nsteps * nsensors
        in_points = np.concatenate((in_points, ident), axis=1)

        nin_points = normalize_3d_coordinate(in_points[:, :3], padding=0.1)
        iindex = (nin_points * res).astype(int)
        igrid = np.zeros(shape=(res, res, res), dtype=int)
        # if a voxel includes more than one los_points, this will simply choose the first los_point in the list!
        igrid[iindex[:, 0], iindex[:, 1], iindex[:, 2]] = np.arange(iindex.shape[0])
        selected_iindex = igrid[igrid > 0]
        in_points = in_points[selected_iindex]

        ## outside:
        npoints = np.repeat(points, n, axis=0)
        ident = np.arange(points.shape[0])
        ident = np.repeat(ident, n, axis=0)
        ident = np.expand_dims(ident, axis=1)
        nsensors = np.repeat(sensor_vecs, n, axis=0)
        nsteps = np.tile(steps, [points.shape[0], 3])
        los_points = npoints + nsteps * nsensors
        los_points = np.concatenate((los_points, ident), axis=1)

        nlos_points = normalize_3d_coordinate(los_points[:, :3], padding=0.1)
        lindex = (nlos_points * res).astype(int)

        lgrid = np.zeros(shape=(res, res, res), dtype=int)
        # if a voxel includes more than one los_points, this will simply choose the first los_point in the list!
        lgrid[lindex[:, 0], lindex[:, 1], lindex[:, 2]] = np.arange(lindex.shape[0])

        # if there is a (buffered) point, keep the los_point
        active = lgrid * pgrid
        selected_lindex = active[active > 0]
        los_points = los_points[selected_lindex]

        ### put everything together
        cident = np.zeros(shape=(points.shape[0], 2))
        ins = np.concatenate((np.ones(shape=(in_points.shape[0], 1)), np.zeros(shape=(in_points.shape[0], 1))), axis=1)
        out = np.concatenate((np.zeros(shape=(los_points.shape[0], 1)), np.ones(shape=(los_points.shape[0], 1))), axis=1)
        cident = np.concatenate((cident,
                                 ins,
                                 out))

        sensor_vecs = np.concatenate((sensor_vecs,
                                      sensor_vecs[in_points[:, 3].astype(int)],
                                      sensor_vecs[los_points[:, 3].astype(int)]))
        normals = np.concatenate((normals,
                                  normals[in_points[:, 3].astype(int)],
                                  normals[los_points[:, 3].astype(int)]))
        gt_normals = np.concatenate((gt_normals,
                                     gt_normals[in_points[:, 3].astype(int)],
                                     gt_normals[los_points[:, 3].astype(int)]))

        points = np.concatenate((points,
                                 in_points[:, :3],
                                 los_points[:, :3]))
        points = np.concatenate((points, cident), axis=1)

        # print("time: ", time.time() - t0)

        data = {
            None: points.astype(np.float32),
            'normals': normals.astype(np.float32),
            'gt_normals': gt_normals.astype(np.float32),
            'sensors': sensor_vecs.astype(np.float32),
        }

        return data