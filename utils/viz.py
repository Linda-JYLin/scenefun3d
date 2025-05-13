#
# Visualization helpers
#
# SceneFun3D Toolkit
#

import open3d as o3d
import numpy as np
import copy
# from utils.pc_process import pc_estimate_normals
from pc_process import pc_estimate_normals
import pyviz3d.visualizer as viz
# from utils.viz_constants import SCANNET_COLOR_MAP_200, VIZ_TOOL_OPTIONS
from viz_constants import SCANNET_COLOR_MAP_200, VIZ_TOOL_OPTIONS

def viz_3d(to_plot_list, viz_tool="pyviz3d"):
    """
    Visualize 3D point clouds.

    This function takes a list of 3D geometries (e.g., point clouds) and visualizes them using Open3D's visualization tools.
    An optional coordinate system can be displayed for reference.

    Args:
        to_plot_list (list of open3d.geometry.PointCloud): A list of Open3D point cloud geometry objects to be visualized.

    Returns:
        (None): This function does not return any value. It opens a visualizer window displaying the geometries.
    
    Example:
        >>> import open3d as o3d

        >>> pcd = o3d.geometry.PointCloud()

        >>> pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))

        >>> viz_3d([pcd])
    """
    if viz_tool not in VIZ_TOOL_OPTIONS:
        assert False, f"Unknown viz tool option {viz_tool}. Visualization tool option must only be 'open3d' or 'pyviz3d'."

    if viz_tool == "pyviz3d":
        v = viz.Visualizer()
        for i, plot_list_item in enumerate(to_plot_list):
            pcd = copy.deepcopy(plot_list_item)
            
            if not pcd.has_normals():
                pcd = pc_estimate_normals(pcd, radius = 0.1, max_nn = 50)

            pcd_points = np.array(pcd.points) 
            pcd_points -= np.mean(pcd_points, axis=0)
            pcd.points = o3d.utility.Vector3dVector(pcd_points)

            v.add_points(f'element_{i}', np.array(pcd.points), np.array(pcd.colors)*255., np.array(pcd.normals), point_size=8, visible=True)
        v.save('pyviz3d_output')
    else:
        o3d.visualization.draw_geometries(to_plot_list)

def viz_masks(laser_scan_pcd, mask_indices, mask_labels = None, use_normals=True, viz_tool="pyviz3d"):
    """
    Visualize point cloud masks.

    Args:
        laser_scan_pcd (open3d.geometry.PointCloud): The original point cloud to be visualized.
        mask_indices (list of lists of int or list of np.array of int): A list where each element is a list (or a np.array) of indices representing a mask in the point cloud.
        mask_labels (list of str, optional): Labels for each mask to be used with Pyviz3D. Must be the same length as mask_indices. Default is None.
        use_normals (bool, optional): Whether to use normals in the visualization. If True and the point cloud does not have normals, they will be estimated. Default is True.
        viz_tool (str, optional): The visualization tool to use, either 'pyviz3d' or 'open3d'. Default is 'pyviz3d'.

    Returns:
        (None): The function visualizes the point cloud masks and does not return any value.
    """
    if viz_tool not in VIZ_TOOL_OPTIONS:
        assert False, f"Unknown viz tool option {viz_tool}. Visualization tool option must only be 'open3d' or 'pyviz3d'."

    if mask_labels is not None and viz_tool == "open3d":
        assert False, f"The mask_labels input is only supported for the visualization option 'pyviz3d'"

    if mask_labels is not None:
        assert len(mask_indices) == len(mask_labels), f"The length of mask_labels must be equal to the length of mask_indices. Each label must correspond to a single mask."

    pcd = copy.deepcopy(laser_scan_pcd)
    if not pcd.has_normals() and use_normals:
        pcd = pc_estimate_normals(pcd, radius = 0.1, max_nn = 50)

    # center the point cloud
    pcd_points = np.array(pcd.points) 
    pcd_points -= np.mean(pcd_points, axis=0)
    pcd.points = o3d.utility.Vector3dVector(pcd_points)

    if viz_tool == "pyviz3d":
        assert use_normals, f"use_normals must be set to True if the visualization tool option is 'pyviz3d'."
        v = viz.Visualizer()
        v.add_points('RGB Color', np.array(pcd.points), np.array(pcd.colors)*255., np.array(pcd.normals), point_size=8, visible=False)

    pcd_colors = np.array(pcd.colors) 
    new_pcd_colors = np.ones((pcd_colors.shape))*0+ (0.77, 0.77, 0.77)

    if viz_tool == "pyviz3d":
        v.add_points('Background Color', np.array(pcd.points), new_pcd_colors*255., np.array(pcd.normals), point_size=8, visible=True)

    COLOR_MAP = list(SCANNET_COLOR_MAP_200.values())

    for annot_i, mask_idx in enumerate(mask_indices):
        cur_color = COLOR_MAP[annot_i % len(COLOR_MAP)]

        new_pcd_colors[mask_idx] = cur_color 
        new_pcd_colors[mask_idx] /= 255.

        if viz_tool == "pyviz3d":
            mask_points = np.array(pcd.points)[mask_idx]
            mask_colors = new_pcd_colors[mask_idx]
            mask_normals = np.array(pcd.normals)[mask_idx]

            if mask_labels is not None:
                cur_label = f"{mask_labels[annot_i]}_{annot_i}"
            else:
                cur_label = f"label_{annot_i}"

            if "exclude" in cur_label:
                v.add_points(cur_label, mask_points, mask_colors*255., mask_normals, point_size=8, visible=False)
            else:
                v.add_points(cur_label, mask_points, mask_colors*255., mask_normals, point_size=8, visible=True)
        
    if viz_tool == "pyviz3d":
        v.save('pyviz3d_output')
    else:
        pcd.colors = o3d.utility.Vector3dVector(new_pcd_colors)
        viz_3d([pcd])


def viz_motions(laser_scan_pcd, motion_types, motion_dirs, motion_origins, motion_viz_orients, motion_labels = None):
    """
    Visualize point cloud masks.

    Args:
        laser_scan_pcd (open3d.geometry.PointCloud): The original point cloud to be visualized.
        motion_types (list of str): A list where each element represents the motion type. Motion type can be either "trans" for translational motions or "rot" for rotational motions.
        motion_dirs (list of np.array): A list where each element represents the motion direction.
        motion_origins (list of np.array): A list where each element represents the motion origin.
        motion_viz_orients (list of str): A list where each element represents the motion orientation. Motion orientation can be either "inwards" or "outwards".
        motion_labels (list of str, optional): Labels for each mask to be used with Pyviz3D.

    Returns:
        (None): The function visualizes the point cloud masks and does not return any value.
    """
    pcd = copy.deepcopy(laser_scan_pcd)
    if not pcd.has_normals():
        pcd = pc_estimate_normals(pcd, radius = 0.1, max_nn = 50)

    # center the point cloud
    pcd_points = np.array(pcd.points) 
    pcd_mean = np.mean(pcd_points, axis=0)
    pcd.points = o3d.utility.Vector3dVector(pcd_points - pcd_mean)

    v = viz.Visualizer()
    v.add_points('RGB Color', np.array(pcd.points), np.array(pcd.colors)*255., np.array(pcd.normals), point_size=8, visible=False)

    pcd_colors = np.array(pcd.colors) 
    new_pcd_colors = np.ones((pcd_colors.shape))*0+ (0.77, 0.77, 0.77)

    if motion_labels is not None:
        motion_labels = [f"{motion_labels[i]}_{i}" for i in range(len(motion_labels))]
    else:
        motion_labels = [f"motion_{i}" for i in range(len(motion_types))]

    v.add_points('Background Color', np.array(pcd.points), new_pcd_colors*255., np.array(pcd.normals), point_size=8, alpha=.8, visible=True)

    COLOR_MAP = list(SCANNET_COLOR_MAP_200.values())
    for idx, (m_type, m_dir, m_origin, m_orient, m_label) in enumerate(zip(motion_types, motion_dirs, motion_origins, motion_viz_orients, motion_labels)):
        cur_color = np.array(COLOR_MAP[idx % len(COLOR_MAP)]) / 255.
        # print(f"[Motion {idx}] Label: {m_label}, Type: {m_type}, Dir: {m_dir}, Origin: {m_origin}, Orient: {m_orient}, Color: {cur_color}")
        v.add_motion(
            m_label,
            m_type,
            np.array(m_dir),
            np.array(m_origin - pcd_mean),
            m_orient,
            cur_color,
            cur_color,
            visible=True
        )
        # breakpoint()
    v.save('pyviz3d_output')

if __name__ == "__main__":
    import open3d as o3d
    import json

    # viz_3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    # pcd.colors = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    # pcd = o3d.io.read_point_cloud(r"D:\Github_documents\scenefun3d\data\420683\420683_laser_scan.ply")
    # viz_3d([pcd])

    # viz_masks
    # def load_mask_annotations(json_file_path):
    #     with open(json_file_path, 'r') as f:
    #         data = json.load(f)
    #
    #     mask_indices = []
    #     mask_labels = []
    #
    #     for annot in data['annotations']:
    #         mask_indices.append(annot['indices'])
    #         mask_labels.append(annot['label'])
    #
    #     return mask_indices, mask_labels
    # mask_indices, mask_labels = load_mask_annotations(r"D:\Github_documents\scenefun3d\data\420683\420683_annotations.json")
    # laser_scan_pcd = o3d.geometry.PointCloud()
    # laser_scan_pcd = o3d.io.read_point_cloud(r"D:\Github_documents\scenefun3d\data\420683\420683_laser_scan.ply")
    # viz_masks(laser_scan_pcd, mask_indices, mask_labels, use_normals=True, viz_tool='pyviz3d')

    # viz_motions
    def load_motion_annotations(json_file_path):
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        motion_types = []
        motion_dirs = []
        motion_origins = []
        motion_viz_orients = []

        for annot in data['motions']:
            motion_types.append(annot['motion_type'])
            motion_dirs.append(annot['motion_dir'])
            motion_origins.append(annot['motion_origin_idx'])
            motion_viz_orients.append(annot['motion_viz_orient'])

        return motion_types, motion_dirs, motion_origins, motion_viz_orients

    motion_types, motion_dirs, motion_origins, motion_viz_orients = load_motion_annotations(r"D:\Github_documents\scenefun3d\data\420683\420683_motions.json")
    laser_scan_pcd = o3d.geometry.PointCloud()
    laser_scan_pcd = o3d.io.read_point_cloud(r"D:\Github_documents\scenefun3d\data\420683\420683_laser_scan.ply")
    points = np.asarray(laser_scan_pcd.points)
    origin_points = points[motion_origins]
    viz_motions(laser_scan_pcd, motion_types, motion_dirs, origin_points, motion_viz_orients, motion_labels=None)