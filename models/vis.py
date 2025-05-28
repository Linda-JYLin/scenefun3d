import open3d as o3d

# 加载点云文件（支持 .ply, .pcd, .xyz 等格式）
pcd = o3d.io.read_point_cloud(r"D:\Github_documents\scenefun3d\models\Mask3D\annotated.ply")
# pcd = o3d.io.read_point_cloud(r"D:\Github_documents\scenefun3d\models\fun3du\data\train\420683\420683_laser_scan.ply")

# 可视化点云
o3d.visualization.draw_geometries([pcd])