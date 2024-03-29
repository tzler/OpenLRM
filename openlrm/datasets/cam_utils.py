# cam_utils.py 
# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import torch
import pickle, os
import numpy as np

"""
R: (N, 3, 3)
T: (N, 3)
E: (N, 4, 4)
vector: (N, 3)
"""


def compose_extrinsic_R_T(R: torch.Tensor, T: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from R and T.
    Batched I/O.
    """
    RT = torch.cat((R, T.unsqueeze(-1)), dim=-1)
    return compose_extrinsic_RT(RT)


def compose_extrinsic_RT(RT: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from RT.
    Batched I/O.
    """
    return torch.cat([
        RT,
        torch.tensor([[[0, 0, 0, 1]]], dtype=RT.dtype, device=RT.device).repeat(RT.shape[0], 1, 1)
        ], dim=1)


def decompose_extrinsic_R_T(E: torch.Tensor):
    """
    Decompose the standard extrinsic matrix into R and T.
    Batched I/O.
    """
    RT = decompose_extrinsic_RT(E)
    return RT[:, :, :3], RT[:, :, 3]


def decompose_extrinsic_RT(E: torch.Tensor):
    """
    Decompose the standard extrinsic matrix into RT.
    Batched I/O.
    """
    return E[:, :3, :]


def camera_normalization_objaverse(normed_dist_to_center, poses: torch.Tensor, ret_transform: bool = False):
    assert normed_dist_to_center is not None
    pivotal_pose = compose_extrinsic_RT(poses[:1])
    dist_to_center = pivotal_pose[:, :3, 3].norm(dim=-1, keepdim=True).item() \
        if normed_dist_to_center == 'auto' else normed_dist_to_center

    # compute camera norm (new version)
    canonical_camera_extrinsics = torch.tensor([[
        [1, 0, 0, 0],
        [0, 0, -1, -dist_to_center],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]], dtype=torch.float32)
    pivotal_pose_inv = torch.inverse(pivotal_pose)
    camera_norm_matrix = torch.bmm(canonical_camera_extrinsics, pivotal_pose_inv)

    # normalize all views
    poses = compose_extrinsic_RT(poses)
    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)
    poses = decompose_extrinsic_RT(poses)

    if ret_transform:
        return poses, camera_norm_matrix.squeeze(dim=0)
    return poses


def get_normalized_camera_intrinsics(intrinsics: torch.Tensor):
    """
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    Return batched fx, fy, cx, cy
    """
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 0, 1]
    cx, cy = intrinsics[:, 1, 0], intrinsics[:, 1, 1]
    width, height = intrinsics[:, 2, 0], intrinsics[:, 2, 1]
    fx, fy = fx / width, fy / height
    cx, cy = cx / width, cy / height
    return fx, fy, cx, cy


def build_camera_principle(RT: torch.Tensor, intrinsics: torch.Tensor):
    """
    RT: (N, 3, 4)
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    """
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    return torch.cat([
        RT.reshape(-1, 12),
        fx.unsqueeze(-1), fy.unsqueeze(-1), cx.unsqueeze(-1), cy.unsqueeze(-1),
    ], dim=-1)


def build_camera_standard(RT: torch.Tensor, intrinsics: torch.Tensor):
    """
    RT: (N, 3, 4)
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    """
    E = compose_extrinsic_RT(RT)
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    I = torch.stack([
        torch.stack([fx, torch.zeros_like(fx), cx], dim=-1),
        torch.stack([torch.zeros_like(fy), fy, cy], dim=-1),
        torch.tensor([[0, 0, 1]], dtype=torch.float32, device=RT.device).repeat(RT.shape[0], 1),
    ], dim=1)
    return torch.cat([
        E.reshape(-1, 16),
        I.reshape(-1, 9),
    ], dim=-1)


def center_looking_at_camera_pose(
    camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None,
    device: torch.device = torch.device('cpu'),
    ):
    """
    camera_position: (M, 3)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4)
    """
    # by default, looking at the origin and world up is pos-z
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
    look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
    up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    z_axis = camera_position - look_at
    z_axis = z_axis / z_axis.norm(dim=-1, keepdim=True)
    x_axis = torch.cross(up_world, z_axis)
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True)
    y_axis = torch.cross(z_axis, x_axis)
    y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True)
    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    return extrinsics


def surrounding_views_linspace(n_views: int, radius: float = 2.0, height: float = 0.8, device: torch.device = torch.device('cpu')):
    """
    n_views: number of surrounding views
    radius: camera dist to center
    height: height of the camera
    return: (M, 3, 4)
    """
    assert n_views > 0
    assert radius > 0

    theta = torch.linspace(-torch.pi / 2, 3 * torch.pi / 2, n_views, device=device)
    projected_radius = math.sqrt(radius ** 2 - height ** 2)
    x = torch.cos(theta) * projected_radius
    y = torch.sin(theta) * projected_radius
    z = torch.full((n_views,), height, device=device)

    camera_positions = torch.stack([x, y, z], dim=1)

    #print('camera_positions\n\n\n', camera_positions)

    extrinsics = center_looking_at_camera_pose(camera_positions, device=device)

    # print('\nsurrounding_views_linespace')
    # print('\nprojected radius: ', projected_radius)
    # print('\nn_views', n_views)
    # print('\npositions', camera_positions.shape)
    # print('\npositions examples: \n', camera_positions[0,:].round())
    # print('\nextrinsics', extrinsics.shape)
    # print('\nextrinsics examples: \n', extrinsics[0,:,:].round())

    return extrinsics

def create_intrinsics(
    f: float,
    c: float = None, 
    cx: float = None, 
    cy: float = None,
    w: float = 1., 
    h: float = 1.,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu'),
    ):
    """
    add values from camera_info['intrinsics']
    return: (3, 2)
    """
    fx = fy = f
    if c is not None:
        assert cx is None and cy is None, "c and cx/cy cannot be used together"
        cx = cy = c
    else:
        assert cx is not None and cy is not None, "cx/cy must be provided when c is not provided"
    fx, fy, cx, cy, w, h = fx/w, fy/h, cx/w, cy/h, 1., 1.
    intrinsics = torch.tensor([
        [fx, fy],
        [cx, cy],
        [w, h],
    ], dtype=dtype, device=device)
    return intrinsics

# all tyler mods below

def extract_dustr_info(self): 

    # determine which trial this image is associated with 
    i_trial = self.cfg.image_input.split('/')[-1].split('_image')[0]
    
    # determine path to dust3r generated viewpoints of trial images 
    _file = os.path.join(self.cfg.dustr_viewpoint_path, i_trial + '.pickle')
    
    # load dust3r generated data 
    with open(_file, 'rb') as handle:
        dustr = pickle.load(handle)
    
    # determine name of all images in this trial
    imagenames = [i[i.find('image'):-4] for i in dustr['imagenames']]
    
    # determine the index of this trial 
    _idx = int(self.cfg.image_input.split('/')[-1].split('_')[3][-1])

    # extract rotation matrices for cameras from all images in this trial
    rotation = [i[:3,:3] for i in dustr['poses']]
    
    # determine rotation matrix relative to the camera pose from this image
    dustr['rotation'] = [rotation[_idx].T @ i for i in rotation]

    # extract xyz positions for cameras from all images in this trial
    xyz_duster = np.array([i[:3,3] for i in dustr['poses']])

    #  determine xyz positions relative to the camera from this image 
    dustr['xyz'] = np.array([xyz_duster[_idx] - i for i in xyz_duster])

    dustr['this_image'] = self.cfg.image_input.split('/')[-1][:-4]

    dustr['fx'] = dustr['intrinsics'][0][0,0]
    dustr['fy'] = dustr['intrinsics'][0][1,1]
    dustr['cx'] = dustr['intrinsics'][0][0,2]
    dustr['cy'] = dustr['intrinsics'][0][1,2]

    return dustr 


def relative_extrinsics(self, radius: float = 2.0, height: float = 0.8, device: torch.device = torch.device('cpu')):
    """
    custom for tyler 
    ref: surrounding_views_linspace() 
    """

    print('relative_views')
    
    # extract camera extrinsics for all images in this trial 
    dustr = extract_dustr_info(self)
    
    # default values from openLRM
    projected_radius = math.sqrt(radius ** 2 - height ** 2)

    # extract into vector format
    x, y, z = dustr['xyz'][:,0], dustr['xyz'][:,1], dustr['xyz'][:,2]
    
    ### WRONG ### WRONG ### WRONG ### WRONG ### WRONG ### WRONG ### WRONG 
    #compute rescale factor to make sure we're using the values needed by LRM
    #should this be computed from relative or absolute values? my guess: absolute
    # max_duster = dustr['xyz'].flatten().max()
    # max_lrm = projected_radius
    # scaleby = max_lrm / max_duster
    # x, y, z = x * scaleby, y * scaleby, z * scaleby

    # # # manually determined by viewing openLRM visualizations
    # origin = [0, -2, 0]
    # # #  adjust position based on where openLRM camera viewing poses start from 
    # x, y, z = x + origin[0], y + origin[1], z + origin[2] 

    # convert to torch 
    x, y, z = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)

    # format for openLRM scripts
    camera_positions = torch.stack([x, y, z], dim=1).cuda() # added cuda()
    
    # generate camera extrinsics in the correct format
    extrinsics = center_looking_at_camera_pose(camera_positions, device=device)

    imagenames = [i[i.find('image'):-4] for i in dustr['imagenames']]
    this_image = self.cfg.image_input.split('/')[-1]
    
    return extrinsics, imagenames, this_image 

def relative_intrinsics(self,  w: float = 1., h: float = 1., dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu'),):
    """
    ref: create_intrinsics(
    add values from camera_info['intrinsics']
    return: (3, 2)
    """
    # extract camera extrinsics for all images in this trial 
    dustr = extract_dustr_info(self)

    fx = dustr['fx']
    fy = dustr['fy']
    cx = dustr['cx']
    cy = dustr['cy']
    fx, fy, cx, cy, w, h = fx/w, fy/h, cx/w, cy/h, 1., 1.
    intrinsics = torch.tensor([
        [fx, fy],
        [cx, cy],
        [w, h],
    ], dtype=dtype, device=device)
    return intrinsics