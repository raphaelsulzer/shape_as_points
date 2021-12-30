import sys, os
import open3d as o3d
import glob
import vedo
from vtkplotter import *


cam_dict = {}
cam_dict["bathtub_0113"] = dict(pos=(-0.1967, -1.534, 1.253),
           focalPoint=(0.07037, 0.05843, -0.1489),
           viewup=(0.1221, 0.6442, 0.7551),
           distance=2.138,
           clippingRange=(4.567e-3, 4.567))
cam_dict["desk_0234"] = dict(pos=(-1.766, 1.298, 1.061),
           focalPoint=(-0.02866, -0.02603, -0.04862),
           viewup=(0.3814, -0.2463, 0.8910),
           distance=2.450,
           clippingRange=(0.02693, 5.373))
cam_dict["desk_0201"] = dict(pos=(-1.950, 0.9284, 0.4101),
           focalPoint=(0.08787, -0.02095, -0.04916),
           viewup=(0.1802, -0.08722, 0.9798),
           distance=2.295,
           clippingRange=(0.3031, 4.935))
cam_dict["toilet_0403"] = dict(pos=(-1.025, -1.226, 1.068),
           focalPoint=(0.03091, 0.06724, -0.04808),
           viewup=(0.3263, 0.4504, 0.8311),
           distance=2.008,
           clippingRange=(5.136e-3, 5.136))
cam_dict["nightstand_0271"] = dict(pos=(-2.486, 1.033, 0.6039),
           focalPoint=(0.09281, -0.09465, -0.09869),
           viewup=(0.2143, -0.1144, 0.9700),
           distance=2.900,
           clippingRange=(0.5270, 5.727))
cam_dict["chair_0976"] = dict(pos=(-1.631, 0.4546, 0.8033),
           focalPoint=(0.07062, -0.09862, -0.08443),
           viewup=(0.3726, -0.2760, 0.8860),
           distance=1.997,
           clippingRange=(4.522e-3, 4.522))

path = "/home/adminlocal/PhD/data/ModelNet/paper/meshes/"
img_path = "/home/adminlocal/PhD/data/ModelNet/paper/img/"
models = ["desk_0234","desk_0201","bathtub_0113","toilet_0403","nightstand_0271","chair_0976"]
for m in models:

    files = glob.glob(path+m+"*")

    for f in files:

        print(f)

        data = vedo.load(f)


        if(f.find('scan') == -1):
            continue
            data = vedo.Mesh(data,c=[180,180,180])

        else:
            # data = vedo.Points(data.points(), r=25.0, c=[0, 128, 255])
            data = vedo.Points(data.points(), r=25.0, c=[113, 189, 247])

            # mesh = o3d.io.read_point_cloud(f)
            # continue
        #
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(width=600, height=600, visible=True)
        # vis.add_geometry(mesh)
        #
        #
        # vis.get_render_option().light_on = True
        # vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Default
        #
        # vis.run()
        #
        image_file = os.path.join(img_path, os.path.basename(f)[:-4] + ".png")
        # print("save to: ", image_file)
        # vis.capture_screen_image(image_file, do_render=True)
        # vis.close()

        p1 = vedo.Point([1, 1, 1], c='y')
        light = vedo.Light(p1, c='w', intensity=1)

        # vp = Plotter(axes=0, offscreen=True)
        # vp+=pts
        # vp+=p1
        # vp+=light

        cam = cam_dict[m]

        # .lighting("default")
        p = vedo.show(data.lighting("default"),p1,light,size=(700,700),camera=cam,interactive=False)
        vedo.io.screenshot(image_file)
        p.close()




        a=5
        # recon_mesh = o3d.io.read_triangle_mesh(mesh_file)






