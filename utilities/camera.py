import math

import glm
import torch
import random

import numpy as np
import torchvision.transforms as transforms

from .resize_right import resize

blurs = [
    transforms.Compose([
        transforms.GaussianBlur(11, sigma=(5, 5))
    ]),
    transforms.Compose([
        transforms.GaussianBlur(11, sigma=(2, 2))
    ]),
    transforms.Compose([
        transforms.GaussianBlur(5, sigma=(5, 5))
    ]),
    transforms.Compose([
        transforms.GaussianBlur(5, sigma=(2, 2))
    ]),
]


def get_random_bg(h, w, rand_solid=False):
    p = torch.rand(1)

    if p > 0.66666:
        if rand_solid:
            background = torch.vstack([
                torch.full((1, h, w), torch.rand(1).item()),
                torch.full((1, h, w), torch.rand(1).item()),
                torch.full((1, h, w), torch.rand(1).item()),
            ]).unsqueeze(0) + torch.rand(1, 3, h, w)
            background = ((background - background.amin()) / (background.amax() - background.amin()))
            background = blurs[random.randint(0, 3)](background).permute(0, 2, 3, 1)
        else:
            background = blurs[random.randint(0, 3)](torch.rand((1, 3, h, w))).permute(0, 2, 3, 1)
    elif p > 0.333333:
        size = random.randint(5, 10)
        background = torch.vstack([
            torch.full((1, size, size), torch.rand(1).item() / 2),
            torch.full((1, size, size), torch.rand(1).item() / 2),
            torch.full((1, size, size), torch.rand(1).item() / 2),
        ]).unsqueeze(0)

        second = torch.rand(3)

        background[:, 0, ::2, ::2] = second[0]
        background[:, 1, ::2, ::2] = second[1]
        background[:, 2, ::2, ::2] = second[2]

        background[:, 0, 1::2, 1::2] = second[0]
        background[:, 1, 1::2, 1::2] = second[1]
        background[:, 2, 1::2, 1::2] = second[2]

        background = blurs[random.randint(0, 3)](resize(background, out_shape=(h, w)))

        background = background.permute(0, 2, 3, 1)

    else:
        background = torch.vstack([
            torch.full((1, h, w), torch.rand(1).item()),
            torch.full((1, h, w), torch.rand(1).item()),
            torch.full((1, h, w), torch.rand(1).item()),
        ]).unsqueeze(0).permute(0, 2, 3, 1)

    return background


def cosine_sample(N: np.ndarray) -> np.ndarray:
    """
    #----------------------------------------------------------------------------
    # Cosine sample around a vector N
    #----------------------------------------------------------------------------

    Copied from nvdiffmodelling

    """
    # construct local frame
    N = N / np.linalg.norm(N)

    dx0 = np.array([0, N[2], -N[1]])
    dx1 = np.array([-N[2], 0, N[0]])

    dx = dx0 if np.dot(dx0, dx0) > np.dot(dx1, dx1) else dx1
    dx = dx / np.linalg.norm(dx)
    dy = np.cross(N, dx)
    dy = dy / np.linalg.norm(dy)

    # cosine sampling in local frame
    phi = 2.0 * np.pi * np.random.uniform()
    s = np.random.uniform()
    costheta = np.sqrt(s)
    sintheta = np.sqrt(1.0 - s)

    # cartesian vector in local space
    x = np.cos(phi) * sintheta
    y = np.sin(phi) * sintheta
    z = costheta

    # local to world
    return dx * x + dy * y + N * z


def persp_proj(fov_x=45, ar=1, near=1.0, far=50.0):
    """
    From https://github.com/rgl-epfl/large-steps-pytorch by @bathal1 (Baptiste Nicolet)

    Build a perspective projection matrix.
    Parameters
    ----------
    fov_x : float
        Horizontal field of view (in degrees).
    ar : float
        Aspect ratio (w/h).
    near : float
        Depth of the near plane relative to the camera.
    far : float
        Depth of the far plane relative to the camera.
    """
    fov_rad = np.deg2rad(fov_x)

    tanhalffov = np.tan((fov_rad / 2))
    max_y = tanhalffov * near
    min_y = -max_y
    max_x = max_y * ar
    min_x = -max_x

    z_sign = -1.0
    proj_mat = np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]])

    proj_mat[0, 0] = 2.0 * near / (max_x - min_x)
    proj_mat[1, 1] = 2.0 * near / (max_y - min_y)
    proj_mat[0, 2] = (max_x + min_x) / (max_x - min_x)
    proj_mat[1, 2] = (max_y + min_y) / (max_y - min_y)
    proj_mat[3, 2] = z_sign

    proj_mat[2, 2] = z_sign * far / (far - near)
    proj_mat[2, 3] = -(far * near) / (far - near)

    return proj_mat


def get_tet_params(length):
    tet_vertices = np.array([
        [-0.5, -0.5, -0.5],  # Vertex 0
        [0.5, 0.5, -0.5],  # Vertex 1
        [0.5, -0.5, 0.5],  # Vertex 2
        [-0.5, 0.5, 0.5]  # Vertex 3
    ])
    tet_vertices = tet_vertices * length
    tet_center = np.mean(tet_vertices, axis=0)
    camera_params = []
    for vertex in tet_vertices:
        direction = tet_center - vertex
        direction /= np.linalg.norm(direction)  # Normalize the direction vector

        # Compute the up vector (assuming positive y-axis is up)
        up = np.array([0, 1, 0])
        # Compute the right vector using cross product
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)  # Normalize the right vector

        # Compute the up vector using cross product again
        up = np.cross(direction, right)
        up /= np.linalg.norm(up)  # Normalize the up vector

        # Construct the rotation matrix
        rotation_matrix = np.column_stack((right, up, direction))

        # Translation vector is the vertex itself
        translation_vector = vertex

        # Construct the camera matrix
        camera_matrix = np.array([
            [focal_length, 0, principal_point[0]],
            [0, focal_length, principal_point[1]],
            [0, 0, 1]
        ])

        extrinsic_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
        camera_params.append(np.dot(camera_matrix, extrinsic_matrix))
    return camera_params

def get_camera_params(elev_angle, azim_angle, distance, resolution, fov=60, look_at=[0, 0, 0], up=[0, -1, 0]):
    elev = np.radians(elev_angle)
    azim = np.radians(azim_angle)

    # Generate random view
    cam_z = distance * np.cos(elev) * np.sin(azim)
    cam_y = distance * np.sin(elev)
    cam_x = distance * np.cos(elev) * np.cos(azim)

    modl = glm.mat4()
    view = glm.lookAt(
        glm.vec3(cam_x, cam_y, cam_z),
        glm.vec3(look_at[0], look_at[1], look_at[2]),
        glm.vec3(up[0], up[1], up[2]),
    )

    a_mv = view * modl
    a_mv = np.array(a_mv.to_list()).T
    proj_mtx = persp_proj(fov)

    a_mvp = np.matmul(proj_mtx, a_mv).astype(np.float32)[None, ...]

    a_lightpos = np.linalg.inv(a_mv)[None, :3, 3]
    a_campos = a_lightpos

    return {
        'mvp': a_mvp,
        'lightpos': a_lightpos,
        'campos': a_campos,
        'resolution': [resolution, resolution],
    }


# Returns a batch of camera parameters
class CameraBatch(torch.utils.data.Dataset):
    def __init__(
            self,
            image_resolution,
            distances,
            azimuths,
            elevation_params,
            fovs,
            aug_loc,
            aug_light,
            aug_bkg,
            bs,
            look_at=[0, 0, 0], up=[0, -1, 0],
            rand_solid=False
    ):

        self.res = image_resolution

        self.dist_min = distances[0]
        self.dist_max = distances[1]

        self.azim_min = azimuths[0]
        self.azim_max = azimuths[1]

        self.fov_min = fovs[0]
        self.fov_max = fovs[1]

        self.elev_alpha = elevation_params[0]
        self.elev_beta = elevation_params[1]
        self.elev_max = elevation_params[2]

        self.aug_loc = aug_loc
        self.aug_light = aug_light
        self.aug_bkg = aug_bkg

        self.look_at = look_at
        self.up = up

        self.batch_size = bs
        self.rand_solid = rand_solid

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):

        elev = np.radians(np.random.beta(self.elev_alpha, self.elev_beta) * self.elev_max)
        # elev = np.radians(np.random.uniform(0.0, 1.0) * 30)
        azim = np.radians(np.random.uniform(self.azim_min, self.azim_max + 1.0))
        dist = np.random.uniform(self.dist_min, self.dist_max)
        fov = np.random.uniform(self.fov_min, self.fov_max)

        proj_mtx = persp_proj(fov)

        # Generate random view
        cam_z = dist * np.cos(elev) * np.sin(azim)
        cam_y = dist * np.sin(elev)
        cam_x = dist * np.cos(elev) * np.cos(azim)

        if self.aug_loc:

            # Random offset
            limit = self.dist_min // 2
            rand_x = np.random.uniform(-limit, limit)
            rand_y = np.random.uniform(-limit, limit)

            modl = glm.translate(glm.mat4(), glm.vec3(rand_x, rand_y, 0))

        else:

            modl = glm.mat4()

        view = glm.lookAt(
            glm.vec3(cam_x, cam_y, cam_z),
            glm.vec3(self.look_at[0], self.look_at[1], self.look_at[2]),
            glm.vec3(self.up[0], self.up[1], self.up[2]),
        )

        r_mv = view * modl
        r_mv = np.array(r_mv.to_list()).T

        mvp = np.matmul(proj_mtx, r_mv).astype(np.float32)
        campos = np.linalg.inv(r_mv)[:3, 3]

        if self.aug_light:
            lightpos = cosine_sample(campos) * dist
        else:
            lightpos = campos * dist

        if self.aug_bkg:
            bkgs = get_random_bg(self.res, self.res, self.rand_solid).squeeze(0)
        else:
            bkgs = torch.ones(self.res, self.res, 3)

        return {
            'mvp': torch.from_numpy(mvp).float(),
            'lightpos': torch.from_numpy(lightpos).float(),
            'campos': torch.from_numpy(campos).float(),
            'bkgs': bkgs,
            'azim': torch.tensor(azim).float(),
            'elev': torch.tensor(elev).float(),
            'c2w': torch.from_numpy(r_mv).float()
        }


def generate_init_params(length):
    # tet_vertices = np.array([
    #     [-0.5, -0.5, -0.5],  # Vertex 0
    #     [0.5, 0.5, -0.5],  # Vertex 1
    #     [0.5, -0.5, 0.5],  # Vertex 2
    #     [-0.5, 0.5, 0.5]  # Vertex 3
    # ])

    vertices = np.array([
        [-0.5, 0, 0],  # Vertex 0
        [0.5, 0, 0],  # Vertex 1
        [0, 0, -0.5],  # Vertex 2
        [0, 0, 0.5]  # Vertex 3
    ])

    vertices = vertices * length
    center = np.mean(vertices, axis=0)
    # print(tet_center)
    camera_views = []
    for vertex in vertices:
        up = np.array([0, -1, 0])
        view = glm.lookAt(glm.vec3(vertex), glm.vec3(center), glm.vec3(up))
        camera_views.append(view)

    return camera_views


class MultiCameraBatch(torch.utils.data.Dataset):
    def __init__(self,
                 image_res,
                 distances,
                 azimuths,
                 elevation_params,
                 fovs,
                 aug_loc,
                 aug_light,
                 aug_bkg,
                 bs,
                 look_at=[0, 0, 0], up=[0, -1, 0],
                 rand_solid=False):
        self.res = image_res

        self.dist_min = distances[0]
        self.dist_max = distances[1]

        self.azim_min = azimuths[0]
        self.azim_max = azimuths[1]

        self.fov_min = fovs[0]
        self.fov_max = fovs[1]

        self.elev_alpha = elevation_params[0]
        self.elev_beta = elevation_params[1]
        self.elev_max = elevation_params[2]

        self.look_at = look_at
        self.up = up
        # self.dist = math.sqrt(3) * 0.5 * length
        # self.fov_min = 30
        # self.fov_max = 45
        # self.fov = fov
        self.aug_light = aug_light
        self.aug_bkg = aug_bkg
        self.aug_loc = aug_loc

        self.batch_size = bs
        self.rand_solid = rand_solid
        self.oro = np.radians(np.array([0, 90, 180, 270]))
        # self.camera = generate_init_params(length)

    def __len__(self):
        return self.batch_size

    def __getitem__(self, item):
        mvps = []
        light_pos = []
        camposes = []
        bkgs = []
        azims = []
        elevs = []
        c2ws = []
        azim_base = np.radians(np.random.uniform(self.azim_min, self.azim_max + 1.0))
        fov = np.random.uniform(self.fov_min, self.fov_max)
        proj_mtx = persp_proj(fov)
        for angle in self.oro:
            elev = np.radians(np.random.beta(self.elev_alpha, self.elev_beta) * self.elev_max)
            # elev = np.radians(np.random.uniform(0.0, 1.0) * 30)
            dist = np.random.uniform(self.dist_min, self.dist_max)

            azim = azim_base + angle
            # fov = np.random.uniform(self.fov_min, self.fov_max)
            # proj_mtx = persp_proj(fov)
            # fov = np.random.uniform(self.fov_min, self.fov_max)
            # rotate_angle = np.random.uniform(0, 360)
            cam_z = dist * np.cos(elev) * np.sin(azim)
            cam_y = dist * np.sin(elev)
            cam_x = dist * np.cos(elev) * np.cos(azim)

            if self.aug_loc:

                # Random offset
                limit = self.dist_min // 2
                rand_x = np.random.uniform(-limit, limit)
                rand_y = np.random.uniform(-limit, limit)

                modl = glm.translate(glm.mat4(), glm.vec3(rand_x, rand_y, 0))

            else:

                modl = glm.mat4()

            view = glm.lookAt(
                glm.vec3(cam_x, cam_y, cam_z),
                glm.vec3(self.look_at[0], self.look_at[1], self.look_at[2]),
                glm.vec3(self.up[0], self.up[1], self.up[2]),
            )

            r_mv = view * modl
            r_mv = np.array(r_mv.to_list()).T

            mvp = np.matmul(proj_mtx, r_mv).astype(np.float32)

            campos = np.linalg.inv(r_mv)[:3, 3]

        # modl = glm.rotate(glm.radians(rotate_angle), glm.vec3(0, 1, 0))


        # for view in self.camera:
        #     r_mv = view * modl
        #     r_mv = np.array(r_mv.to_list()).T
        #     pos = np.linalg.inv(r_mv)[:3, 3]
        #
        #     campos.append(torch.from_numpy(pos).float())


            # if self.aug_bkg:
            #     bkgs = get_random_bg(self.res, self.res, self.rand_solid).squeeze(0)
            # else:
            #     bkgs = torch.ones(self.res, self.res, 3)
            if self.aug_light:
                l_pos = cosine_sample(campos) * dist
            else:
                l_pos = campos * dist


            if self.aug_bkg:
                bkg = get_random_bg(self.res, self.res, self.rand_solid).squeeze(0)
            else:
                bkg = torch.ones(self.res, self.res, 3)

            # azims.append(torch.from_numpy(azim))

            mvps.append(torch.from_numpy(mvp).float())
            light_pos.append(torch.from_numpy(l_pos).float())
            camposes.append(torch.from_numpy(campos).float())
            bkgs.append(bkg)
            azims.append(torch.tensor(azim).float())
            elevs.append(torch.tensor(elev).float())
            c2ws.append(torch.from_numpy(r_mv).float())
        return {
            'mvp': mvps,
            'lightpos': light_pos,
            'campos': camposes,
            'bkgs': bkgs,
            'azim': azims,
            'elev': elevs,
            'c2ws': c2ws
        }


class ListCameraBatch(torch.utils.data.Dataset):
    def __init__(self, datasets, bs, weights=None):
        self.datasets = datasets
        self.batch_size = bs
        self.weights = weights

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):
        d = random.choices(self.datasets, weights=self.weights)[0]
        return d[index]

def normalize_camera(camera_matrix):
    ''' normalize the camera location onto a unit-sphere'''
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix = camera_matrix.reshape(-1,4,4)
        translation = camera_matrix[:,:3,3]
        translation = translation / (np.linalg.norm(translation, axis=1, keepdims=True) + 1e-8)
        camera_matrix[:,:3,3] = translation
    else:
        camera_matrix = camera_matrix.reshape(-1,4,4)
        translation = camera_matrix[:,:3,3]
        translation = translation / (torch.norm(translation, dim=1, keepdim=True) + 1e-8)
        camera_matrix[:,:3,3] = translation
    return camera_matrix.reshape(-1,16)

def convert_opengl_to_blender(camera_matrix):
    if isinstance(camera_matrix, np.ndarray):
        # Construct transformation matrix to convert from OpenGL space to Blender space
        flip_yz = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        camera_matrix_blender = np.dot(flip_yz, camera_matrix)
    else:
        # Construct transformation matrix to convert from OpenGL space to Blender space
        flip_yz = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        if camera_matrix.ndim == 3:
            flip_yz = flip_yz.unsqueeze(0)
        camera_matrix_blender = torch.matmul(flip_yz.to(camera_matrix), camera_matrix)
    return camera_matrix_blender
