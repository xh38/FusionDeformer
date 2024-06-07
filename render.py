import pathlib

import torch
import numpy as np
import nvdiffrast.torch as dr

from nvdiffmodeling.src import obj
from nvdiffmodeling.src import mesh
from nvdiffmodeling.src import render
from nvdiffmodeling.src import texture
from easydict import EasyDict
from utilities.helpers import create_scene
import kornia

dr_ctx = dr.RasterizeGLContext()

mesh_path = "./outputs/moving/fish-180-shark/tmp/mesh.obj"

def render_mesh(render_mesh_path, save_image_path):
    load_mesh = obj.load_obj(render_mesh_path)
    load_mesh = mesh.unit_size(load_mesh)

    texture_map = texture.create_trainable(np.random.uniform(size=[512] * 2 + [3], low=0.0, high=1.0), [512] * 2, True)
    normal_map = texture.create_trainable(np.array([0, 0, 1]), [512] * 2, True)
    specular_map = texture.create_trainable(np.array([0, 0, 0]), [512] * 2, True)

    load_mesh = mesh.Mesh(
            material={
                'bsdf': 'diffuse',
                'kd': texture_map,
                'ks': specular_map,
                'normal': normal_map,
            },
            base=load_mesh  # Get UVs from original loaded mesh
        )

    ready_texture = texture.Texture2D(torch.full_like(load_mesh.material['kd'].data, 0.5))
    ready_specular = texture.Texture2D(
        kornia.filters.gaussian_blur2d(
            load_mesh.material['ks'].data.permute(0, 3, 1, 2),
            kernel_size=(7, 7),
            sigma=(3, 3),
        ).permute(0, 2, 3, 1).contiguous()
    )

    ready_normal = texture.Texture2D(
        kornia.filters.gaussian_blur2d(
            load_mesh.material['normal'].data.permute(0, 3, 1, 2),
            kernel_size=(7, 7),
            sigma=(3, 3),
        ).permute(0, 2, 3, 1).contiguous()
    )

    m = mesh.Mesh(
                load_mesh.v_pos,
                load_mesh.t_pos_idx,
                material={
                    'bsdf': 'diffuse',
                    'kd': ready_texture,
                    'ks': ready_specular,
                    'normal': ready_normal,
                },
                base=load_mesh  # gets uvs etc from here
            )

    cfg = EasyDict(bsdf='diffuse')
    render_mesh = create_scene([m.eval()], cfg, sz=512)
    render_mesh = mesh.auto_normals(render_mesh)
    render_mesh = mesh.compute_tangents(render_mesh)
    elev = 15
    azim = 80
    dist = 3
    fov = 40

    from utilities.camera import persp_proj
    import glm
    proj_mtx = persp_proj(fov)

    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    # print("azim rad: " + str(azim_rad))
    cam_z = dist * np.cos(elev_rad) * np.sin(azim_rad)
    cam_y = dist * np.sin(elev_rad)
    cam_x = dist * np.cos(elev_rad) * np.cos(azim_rad)
    # print("cam_x: " + str(cam_x))
    # print("cam_z: " + str(cam_z))
    view = glm.lookAt(
        glm.vec3(cam_x, cam_y, cam_z),
        glm.vec3(0, 0, 0),
        glm.vec3(0, -1, 0),
    )

    modl = glm.mat4()
    r_mv = view * modl
    r_mv = np.array(r_mv.to_list()).T
    mvp = np.matmul(proj_mtx, r_mv).astype(np.float32)
    campos = np.linalg.inv(r_mv)[:3, 3]
    bkgs = torch.ones(512, 512, 3)
    lightpos = campos * dist

    params_camera = {
                'mvp': torch.from_numpy(mvp).float(),
                'lightpos': torch.from_numpy(lightpos).float(),
                'campos': torch.from_numpy(campos).float(),
                'bkgs': bkgs,
                'azim': torch.tensor(azim).float(),
                'elev': torch.tensor(elev).float(),
            }

    for k in params_camera:
        params_camera[k] = params_camera[k].unsqueeze(0).to("cuda")
    final_mesh = render_mesh.eval(params_camera)

    train_render = render.render_mesh(
        dr_ctx,
        final_mesh,
        params_camera['mvp'],
        params_camera['campos'],
        params_camera['lightpos'],
        5.0,
        512,
        spp=1,
        num_layers=1,
        msaa=False,
        background=params_camera['bkgs']
    ).permute(0, 3, 1, 2)

    s_log = train_render[0, :, :, :]
    import torchvision
    from PIL import Image
    s_log = torchvision.utils.make_grid(s_log)
    ndarr = s_log.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(save_image_path)


exp_base_dir = f"F:/cvpr2024/user-study"
base_dir_path = pathlib.Path(exp_base_dir)
split = 'text'
save_mesh_dir = base_dir_path / ('render-' + split + '-test')
mesh_dir_path = base_dir_path / split
target_folder_name = 'mesh_final'
# target_folder_name = 'mesh_0'
all_folder = []
for folder in mesh_dir_path.glob('**/*'):
    if folder.is_dir() and folder.name == target_folder_name:
        all_folder.append(folder)

        # render_mesh(target_mesh_path, )
from tqdm import tqdm
for folder in tqdm(all_folder):
    target_mesh_path = folder / "mesh.obj"

    parent_folder = folder.parent
    save_mesh_folder = save_mesh_dir / parent_folder.name
    # print(save_mesh_folder)
    save_mesh_path = save_mesh_folder / 'render.png'
    save_mesh_folder.mkdir(parents=True, exist_ok=True)
    render_mesh(str(target_mesh_path), str(save_mesh_path))



