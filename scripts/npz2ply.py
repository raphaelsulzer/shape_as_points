import open3d as o3d
import numpy as np
import glob
import os

path = "/home/adminlocal/PhD/data/ModelNet/paper/"
files = glob.glob(path+"*.npz")


for f in files:

    data = np.load(f)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data["points"].astype(np.float16))
    pcd.normals = o3d.utility.Vector3dVector(data["normals"])
    o3d.io.write_point_cloud(f[:-4]+".ply", pcd)
    os.remove(f)


a=5

