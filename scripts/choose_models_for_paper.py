import pandas as pd
import os

data_path = "/mnt/raphael/ModelNet10_out"
fname = "iou_all_ModelNet10.pkl"



# conv_one plane
file = os.path.join(data_path,"conv_onet","scan4","conventional_plane","generation","eval_meshes_full.pkl")
df_co_conv = pd.read_pickle(file)
df_co_conv=df_co_conv.rename({"iou (mesh)":"iou1"},axis=1)

# file = os.path.join(data_path,"conv_onet","scan4","sensor_vec_norm",fname)
# df_co_senv = pd.read_pickle(file)


file = os.path.join(data_path,"conv_onet","scan4","uniform_neighborhood_1","generation","eval_meshes_full.pkl")
df_co_norm = pd.read_pickle(file)
df_co_norm=df_co_norm.rename({"iou (mesh)":"iou2"},axis=1)

# conv_one grid
file = os.path.join(data_path,"conv_onet","scan4","grid64","generation","eval_meshes_full.pkl")
df_cog_conv = pd.read_pickle(file)
df_cog_conv=df_cog_conv.rename({"iou (mesh)":"iou7"},axis=1)

# file = os.path.join(data_path,"conv_onet","scan4","sensor_vec_norm",fname)
# df_co_senv = pd.read_pickle(file)


file = os.path.join(data_path,"conv_onet","scan4","grid64_aux","generation","eval_meshes_full.pkl")
df_cog_norm = pd.read_pickle(file)
df_cog_norm=df_cog_norm.rename({"iou (mesh)":"iou8"},axis=1)


# sap
file = os.path.join(data_path,"sap","scan43","conventional","backup",fname)
df_sp_conv = pd.read_pickle(file)
df_sp_conv=df_sp_conv.rename({"iou":"iou3"},axis=1)


file = os.path.join(data_path,"sap","scan43","sensor_vec_norm",fname)
df_sp_senv = pd.read_pickle(file)


file = os.path.join(data_path,"sap","scan43","uniform_neighborhood_1","backup",fname)
df_sp_norm = pd.read_pickle(file)
df_sp_norm=df_sp_norm.rename({"iou":"iou4"},axis=1)

# # p2s
file = os.path.join(data_path,"p2s","conventional","iou_all_ModelNet_backup.pkl")
df_p2_conv = pd.read_pickle(file)
df_p2_conv=df_p2_conv.rename({"iou":"iou5"},axis=1)


file = os.path.join(data_path,"p2s","sensor_vec_norm",fname)
df_p2_senv = pd.read_pickle(file)
# df_p2_norm=df_p2_senv.rename({"iou":"iou6"},axis=1)

file = os.path.join(data_path,"p2s","uniform_neighborhood_1","iou_all_ModelNet_backup.pkl")
df_ps_norm = pd.read_pickle(file)
df_p2_norm=df_p2_senv.rename({"iou":"iou6"},axis=1)


df_co_grid=pd.merge(df_cog_conv,df_cog_norm,on=["class id","class name","modelname"])
df_co=pd.merge(df_co_conv,df_co_norm,on=["class id","class name","modelname"])
df_sp=pd.merge(df_sp_conv,df_sp_norm,on=["class id","class name","modelname"])
df_p2=pd.merge(df_p2_conv,df_p2_norm,on=["class","id"])

df_p2["class id"] = df_p2["class"]

df_p2= df_p2.rename(columns={'class': 'class name', 'id': 'modelname'})


df=pd.merge(df_co,df_sp,on=["class id","class name","modelname"])
df=pd.merge(df,df_p2,on=["class id","class name","modelname"])
df=pd.merge(df,df_co_grid,on=["class id","class name","modelname"])

df["iou11"] = df["iou1"] - df["iou2"]
df["iou33"] = df["iou3"] - df["iou4"]
df["iou55"] = df["iou5"] - df["iou6"]
df["iou77"] = df["iou7"] - df["iou8"]
df["iou"] = df["iou11"] + df["iou33"] + df["iou55"]
print(df.sort_values(by="iou"))

a = 5






