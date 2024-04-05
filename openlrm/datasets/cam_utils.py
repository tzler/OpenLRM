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
    RT = render_camera_extrinsics
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

    #print('axes',z_axis.shape, '\n', z_axis.round() )

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    
    #print('extrinsics.shape:', extrinsics.shape, '\nextrinsics:', extrinsics)

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

def dust3r_to_lrm_coordinates(dust3r, reference_camera_idx = 0, lrm_point = [0, -2, 0]): 
  """ NOT USING THIS FUNCTION BUT KEEPING FOR REFERENCE —— ALMOST WORKING"""
  # get origin for this trial 
  i_origin = dust3r['origin']

  # get translation vectors for each camera
  i_xyz = [i[:3,3] for i in dust3r['poses']]

  # translate cameras relative to origin 
  xyz_relative_to_origin = [i - i_origin for i in i_xyz]

  # set refernce point for LRM (determined from data)
  lrm_point = [0, -2, 0]

  # determine reference camera norm (ie radius)
  r_camera = np.linalg.norm(xyz_relative_to_origin[reference_camera_idx])

  # determine lrm camera norm (ie radius)
  r_lrm_point = np.linalg.norm(lrm_point)

  # determine scaling factor 
  scaleby = r_lrm_point/ r_camera 

  # scale dust3r cameras 
  xyz_rescaled = [i * scaleby for i in xyz_relative_to_origin]

  # determine translation needed for reference point 
  reference_translation = lrm_point - xyz_rescaled[reference_camera_idx]

  # translate all points 
  xyz_rescaled_translated = [i + reference_translation for i in xyz_rescaled]

  # # determine rotation needed for reference point 
  rotation_matrix = find_rotation(xyz_rescaled[reference_camera_idx], lrm_point)

  # Apply rotation to all points
  xyz_rescaled_rotated = [rotate_point(i,rotation_matrix) for i in xyz_rescaled]

  return i_xyz, xyz_relative_to_origin, xyz_rescaled, xyz_rescaled_translated, xyz_rescaled_rotated

# from chatGPT
# def find_rotation(point1, point2):

#     # Step 1: Find the axis of rotation
#     u = np.cross(point1, point2).astype(np.float64)
#     u /= np.linalg.norm(u).astype(np.float64)

#     # Step 2: Find the angle of rotation
#     theta = np.arccos(np.dot(point1, point2))

#     # Step 3: Construct the rotation matrix
#     K = np.array([[0, -u[2], u[1]],
#                   [u[2], 0, -u[0]],
#                   [-u[1], u[0], 0]])

#     R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

#     return R

# from claude 
def find_rotation(p0, p1):
    """
    Compute the rotation matrix that rotates vector p1 onto vector p0.
    
    Args:
        p0 (numpy.ndarray): The target vector (3x1).
        p1 (numpy.ndarray): The vector to be rotated (3x1).
        
    Returns:
        numpy.ndarray: The rotation matrix (3x3).
    """
    # Normalize the vectors
    p0 = p0 / np.linalg.norm(p0)
    p1 = p1 / np.linalg.norm(p1)
    
    # Compute the cosine of the angle between the vectors
    cos_angle = np.dot(p0, p1)
    
    # If the vectors are already aligned, return the identity matrix
    if np.isclose(cos_angle, 1.0):
        return np.eye(3)
    
    # Compute the cross product to get the rotation axis
    rotation_axis = np.cross(p0, p1)
    
    # Compute the sine of the angle between the vectors
    sin_angle = np.linalg.norm(rotation_axis)
    
    # Normalize the rotation axis
    rotation_axis /= sin_angle
    
    # Compute the skew-symmetric matrix of the rotation axis
    skew_matrix = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                             [rotation_axis[2], 0, -rotation_axis[0]],
                             [-rotation_axis[1], rotation_axis[0], 0]])
    
    # Compute the rotation matrix using Rodrigues' rotation formula
    R = np.eye(3) + sin_angle * skew_matrix + (1 - cos_angle) * np.dot(skew_matrix, skew_matrix)
    
    return R

def rotate_point(point, rotation_matrix):
    return np.dot(rotation_matrix, point)

def transform_extrinsics(_dust3r, reference_camera_idx):

  # if target_extrinsics:

  target_extrinsics = np.array([[ 1.0000e+00,  7.5212e-34, -1.1075e-17, -2.2149e-17],
       [-1.1075e-17,  6.7914e-17, -1.0000e+00, -2.0000e+00],
       [ 0.0000e+00,  1.0000e+00,  6.7914e-17,  1.3583e-16],
       [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
  
  # get camera extrinsics
  extrinsics = np.copy(_dust3r['poses'])

  # get origin for this trial
  i_origin = _dust3r['origin']

  for i in range(len(extrinsics)):
    extrinsics[i][:3,3] = extrinsics[i][:3,3] - _dust3r['origin']

  # get transform relative to reference camera
  transform = target_extrinsics @ np.linalg.inv(extrinsics[reference_camera_idx])

  # transform extrinsics
  extrinsics_new = [(transform @ extrinsics[i]) for i in range(4)]

  # make sure that the reference camera has the desired position
  assert sum(extrinsics_new[reference_camera_idx][:3,3].round(5) == target_extrinsics[:3,3].round(5))==3

  return np.array(extrinsics_new)

def extract_dustr_info(self, _rescale='single'): 

    # determine which trial this image is associated with 
    i_trial = self.cfg.image_input.split('/')[-1].split('_image')[0]
    
    # determine path to dust3r generated viewpoints of trial images 
    _file = os.path.join(self.cfg.dustr_viewpoint_path, i_trial + '.pickle')
    
    # load dust3r generated data 
    with open(_file, 'rb') as handle:
        dust3r = pickle.load(handle)
    
    # determine name of all images in this trial
    imagenames = [i[i.find('image'):-4] for i in dust3r['imagenames']]
    
    # determine the index of this trial 
    _idx = int(self.cfg.image_input.split('/')[-1].split('_')[3][-1])

    ########################### FIRST: THE TRANSLATION VENCTOR    

    # get origin for this trial 
    i_origin = dust3r['origin']

    # get translation vectors for each camera
    xyz = [i[:3,3] for i in dust3r['poses']]

    # translate cameras relative to origin 
    xyz_relative_to_origin = [i - i_origin for i in xyz]

    # set refernce point for LRM (determined from data)
    lrm_point = [0, -2, 0]

    # determine lrm camera norm (ie radius)
    r_lrm_point = np.linalg.norm(lrm_point)
  
    # option to keep dust3r's relative distance from origin or make it uniform 
    if _rescale=='single': 

        print('all vectors preserve their relative norms')
        # determine reference camera norm (ie radius)
        r_camera = np.linalg.norm(xyz_relative_to_origin[_idx])

        # determine scaling factor 
        scaleby = r_lrm_point/ r_camera 

        # scale dust3r cameras 
        xyz_rescaled = [i * scaleby for i in xyz_relative_to_origin]

    if _rescale=='all':
        
        print('all vectors now have the same norm')

        rs = [np.linalg.norm(xyz_relative_to_origin[i]) for i in [0,1,2,3]]

        _scale = [r_lrm_point/rs[i] for i in range(len(xyz))]

        # scale dust3r cameras 
        xyz_rescaled = [xyz_relative_to_origin[i] * _scale[i] for i in [0,1,2,3]]

    # determine rotation needed for reference point 
    rotation_matrix = find_rotation(xyz_rescaled[_idx], lrm_point)

    # Apply rotation to all points
    xyz_rescaled_rotated = [rotate_point(i, rotation_matrix) for i in xyz_rescaled]

    # final translated, rescaled, and rotated xyz position 
    dust3r['xyz'] = np.array( xyz_rescaled_rotated ) 

    # OR WE COULD LOOK AT THE POSITIONS THAT ARE NOT ROTATED:  
    # dust3r['xyz'] = np.array( xyz_rescaled ) 
    
    ########################### SECOND: THE ROTATION VECTOR    
    
    # THIS IS WRONG BUT I THINK ITS THE RIGHT STRATEGY: 
    # APPLY SOME TRANSFORMATION TO THE CAMERA ROTATION MATRICES 
    # IN ORDER TO ENSURE WE'RE LOOKING AT THE ORIGIN (BUT!)
    # (WE NEED TO MAKE SURE THAT WE'RE PRESERVING THE CAMERA'S ORIENTATION)
    
    # # extract rotation matrices for cameras from all images in this trial
    # rotation = [i[:3,:3] for i in dust3r['poses']]

    # # determine rotation matrix relative to the camera pose from this image
    # camera_matrices_relative= [rotation[_idx].T @ i for i in rotation]

    # ###### THIS IS WHAT IS WRONG 
    # ###### HERE'S I'M JUST USING THE ROTATION MATRIX USED ON CAMERA POSITION
    # cameras_rotated = [i @ rotation_matrix for i in camera_matrices_relative]
    
    # extract from the LRM scripts manually (AUTOMATE!)
    
    # I CHANGED THE TRANSLATION MATRIX TO 0, -3, 0 to troubleshoot 
    target_extrinsics = np.array([[ 1.0000e+00,  7.5212e-34, -1.1075e-17, -2.2149e-17],
        [-1.1075e-17,  6.7914e-17, -1.0000e+00, -3.0000e+00],
        [ 0.0000e+00,  1.0000e+00,  6.7914e-17,  1.3583e-16],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    
    # get camera extrinsics
    extrinsics = np.copy(dust3r['poses'])

    print('using xyz_relative_to_origin')
    for i in range(len(extrinsics)):
      extrinsics[i][:3,3] = xyz_relative_to_origin[i] #extrinsics[i][:3,3] - _dust3r['origin']

    # get transform relative to reference camera
    transform = target_extrinsics @ np.linalg.inv(extrinsics[_idx])

    # transform extrinsics
    extrinsics_new = [(transform @ extrinsics[i]) for i in range(4)]

    # make sure that the reference camera has the desired position
    assert sum(extrinsics_new[_idx][:3,3].round(5) == target_extrinsics[:3,3].round(5))==3
    
    dust3r['xyz'] = np.array([i[:3,3] for i in extrinsics_new])

    # determine rotation matrix relative to the camera pose from  this image
    dust3r['rotation'] = [i[:3,:3] for i in extrinsics_new] #[rotation[_idx].T @ i for i in rotation]
    
    #dust3r['rotation'] = np.array( [i[:3,:3] for i in dust3r['poses']] ) 

    # intrinsics: focal lendth x     
    dust3r['fx'] = dust3r['intrinsics'][0][0,0]
    
    # intrinsics: focal length y 
    dust3r['fy'] = dust3r['intrinsics'][0][1,1]
    
    # intrinsics: center of x 
    dust3r['cx'] = dust3r['intrinsics'][0][0,2]
    
    # intrinsics: center of y 
    dust3r['cy'] = dust3r['intrinsics'][0][1,2]

    # for reference later
    dust3r['this_image'] = self.cfg.image_input.split('/')[-1][:-4]

    return dust3r 

def relative_extrinsics(self, radius: float = 2.0, height: float = 0.8, device: torch.device = torch.device('cpu')):
    """
    custom for tyler 
    ref: surrounding_views_linspace() 
    """

    print('relative_views!')
    
    # extract camera extrinsics for all images in this trial 
    dustr = extract_dustr_info(self)

    imagenames = [i[i.find('image'):-4] for i in dustr['imagenames']]
    
    this_image = self.cfg.image_input.split('/')[-1]
    
    # default values from openLRM
    projected_radius = math.sqrt(radius ** 2 - height ** 2)

    # extract into vector format
    x, y, z = dustr['xyz'][:,0], dustr['xyz'][:,1], dustr['xyz'][:,2]

    # convert to torch 
    x, y, z = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)

    if self.use_dust3r_camera_rotations == False: 
        
        print('using default camera rotations')
        
        # prep the camera info into the right format for default scripts
        camera_positions = torch.stack([x.float(), y.float(), z.float()], dim=1).cuda() # added cuda()
        
        # default function which generates cameras looking at the origin
        extrinsics = center_looking_at_camera_pose(camera_positions, device=device)

    else:   

        print('using the dust3r camera rotations')

        # THE DEFAULT FUNCTION
        #extrinsics = center_looking_at_camera_pose(camera_positions, device=device)
        # BASICALLY TAKES THE FOLLOWING DATA STRUCTURE 
        # > camera_positions = torch.stack([x.float(), y.float(), z.float()], dim=1).cuda() # added cuda()
        # AND STACKS THE TRANSLATIONS CORDINATES ONTO THE 3x3 ROTATION MATRICES, I.E.,  
        # > extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
        # WHICH IS EQUIVALENT TO THE FOLLOWING 
        rotation_matrices = torch.tensor(np.array(dustr['rotation']), dtype=torch.float32).cuda()  
        traslation_vector = torch.tensor(np.array(dustr['xyz']), dtype=torch.float32).cuda() 
        traslation_vector = traslation_vector.unsqueeze(-1).cuda() 
        extrinsics = torch.concatenate([rotation_matrices, traslation_vector], axis=-1).cuda() 
        
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
    h  = dustr['masks'][0].shape[1] # 100#cy * 2 #1000#
    w  = dustr['masks'][0].shape[0] # 400#cx * 2 #1000#

    fx, fy, cx, cy, w, h = fx/w, fy/h, cx/w, cy/h, 1., 1.
    intrinsics = torch.tensor([
        [fx, fy],
        [cx, cy],
        [w, h],
    ], dtype=dtype, device=device)
    return intrinsics



# PROBABLY GARBAGE


# for i_camera in range(len(xyz)):

#     R = rotation[i_camera]
#     #new_camera_pos = dust3r['xyz'][i_camera]
#     # NO CLUE JUST TRYING TO SEE IF THIS WORKS :/ 
#     # # Compute the vector pointing from the new camera position to the origin
#     # look_vector = -new_camera_pos

#     # # Create a rotation matrix that aligns the -Z axis with the look vector
#     # look_at = np.array([0, 0, -1])  # -Z axis
#     # rotation_vector = np.cross(look_at, look_vector)
#     # rotation_angle = np.arccos(np.dot(look_at, look_vector) / (np.linalg.norm(look_at) * np.linalg.norm(look_vector)))
#     # rotation_matrix = np.eye(3)

#     # if np.linalg.norm(rotation_vector) > 0:
#     #     rotation_matrix = np.eye(3) + np.sin(rotation_angle) * np.cross(np.eye(3), rotation_vector / np.linalg.norm(rotation_vector)) + (1 - np.cos(rotation_angle)) * np.square(rotation_vector / np.linalg.norm(rotation_vector))

#     # # Update the camera's rotation matrix
#     # new_camera_rot = np.dot(rotation_matrix, camera_rot)


#     # Step 1: Find the current direction the camera is facing
#     current_direction = R @ np.array([0, 0, 1])  # Assuming the camera is initially pointing along the positive z-axis
#     print('spagetti wall method')
#     # Step 2: Calculate the rotation matrix that rotates the current direction to the desired direction (negative z-axis)
#     desired_direction = np.array([0, 0, -1])
#     rotation_axis = np.cross(current_direction, desired_direction)
#     rotation_angle = np.arccos(np.dot(current_direction, desired_direction) / (np.linalg.norm(current_direction) * np.linalg.norm(desired_direction)))
#     rotation_matrix = np.eye(3)
#     if np.linalg.norm(rotation_axis) > 0:
#         rotation_matrix = np.eye(3) + np.sin(rotation_angle) * np.array([[0, -rotation_axis[2], rotation_axis[1]],
#                                                                           [rotation_axis[2], 0, -rotation_axis[0]],
#                                                                           [-rotation_axis[1], rotation_axis[0], 0]]) + \
#                           (1 - np.cos(rotation_angle)) * (np.outer(rotation_axis, rotation_axis))

#     # Step 3: Apply the rotation matrix to the current rotation matrix
#     new_R = rotation_matrix @ R

#     print( 'old \n', dust3r['poses'][i_camera][:3,:3].round(2))
#     dust3r['poses'][i_camera][:3,:3] = new_R
#     print( 'new \n', dust3r['poses'][i_camera][:3,:3].round(2))
