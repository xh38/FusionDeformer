import random
import time

import kornia
import os
import pathlib
import pymeshlab
import shutil
import torch
import torchvision
import logging
import yaml

import numpy as np
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt

from easydict import EasyDict
from torch.cuda.amp import autocast, GradScaler

from NeuralJacobianFields import SourceMesh

from nvdiffmodeling.src import obj
from nvdiffmodeling.src import util
from nvdiffmodeling.src import mesh
from nvdiffmodeling.src import render
from nvdiffmodeling.src import texture
from nvdiffmodeling.src import regularizer

from PIL import Image
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from utilities.video import Video
from utilities.helpers import cosine_avg, create_scene, get_vp_map
from utilities.camera import CameraBatch, get_camera_params, MultiCameraBatch
from utilities.clip_spatial import CLIPVisualEncoder
from utilities.resize_right import resize, cubic, linear, lanczos2, lanczos3
from utilities.diffusion_guidance import StableDiffusion


def loop_sds(cfg):
    output_path = pathlib.Path(cfg['output_path'])
    os.makedirs(output_path, exist_ok=True)

    with open(output_path / 'config.yml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    cfg = EasyDict(cfg)
    os.makedirs(pathlib.Path(cfg.output_path + '/guidance'), exist_ok=True)
    print(f'Output directory {cfg.output_path} created')

    device = torch.device(f'cuda:{cfg.gpu}')
    torch.cuda.set_device(device)

    # print('Loading CLIP Models')
    # model, _ = clip.load(cfg.clip_model, device=device)
    if cfg.use_view_consistency:
        fe = CLIPVisualEncoder(cfg.consistency_clip_model, cfg.consistency_vit_stride, device)
        #
        clip_mean = torch.tensor([0.48154660, 0.45782750, 0.40821073], device=device)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

    print('Loading Diffusion Model')
    # sd = StableDiffusion(device)
    # do fp16 to save memory
    sd = StableDiffusion(device, fp16=True)
    sd.requires_grad_(False)
    # save_guidance_f = pathlib.Path(cfg.output_path + '/guidance')

    # output video
    video = Video(cfg.output_path)

    # GL Context
    glctx = dr.RasterizeGLContext()

    print(f'Target text prompt is {cfg.text_prompt}')
    # print(f'Base text prompt is {cfg.base_text_prompt}')
    embeddings = {}
    with torch.no_grad():
        # diffusion guidance
        embeddings['default'] = sd.get_text_embeds(cfg.text_prompt)
        embeddings['empty'] = sd.get_text_embeds('')

        if cfg.view_prompt:
            for d in ['front', 'side', 'back']:
                embeddings[d] = sd.get_text_embeds(f'{cfg.text_prompt}, {d} view')

    os.makedirs(output_path / 'tmp', exist_ok=True)
    ms = pymeshlab.MeshSet()
    if cfg.load_mesh_from_file:
        ms.load_new_mesh(cfg.mesh)

        if cfg.retriangulate:
            print('Retriangulating shape')
            ms.meshing_isotropic_explicit_remeshing()

        if not ms.current_mesh().has_wedge_tex_coord():
            # some arbitrarily high number
            ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)
    else:
        import open3d as o3d
        init_shape = o3d.geometry.TriangleMesh.create_sphere(resolution=60, create_uv_map=True)
        init_shape = init_shape.compute_triangle_normals()
        # o3d.io.write_triangle_mesh(str(output_path / 'tmp' / 'mesh.obj'), init_shape)
        vertices = np.asarray(init_shape.vertices)
        faces = np.asarray(init_shape.triangles)

        ms.add_mesh(
            pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
        )
        if not ms.current_mesh().has_wedge_tex_coord():
            # some arbitrarily high number
            ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)

    # ms.save_current_mesh(str(output_path / 'tmp' / 'mesh.obj'), save_vertex_color=False)
    # load_mesh = obj.load_obj(str(output_path / 'tmp' / 'mesh.obj'))

    ms.save_current_mesh(str(output_path / 'tmp' / 'mesh.obj'))
    load_mesh = obj.load_obj(str(output_path / 'tmp' / 'mesh.obj'))
    load_mesh = mesh.unit_size(load_mesh)

    ms.add_mesh(
        pymeshlab.Mesh(vertex_matrix=load_mesh.v_pos.cpu().numpy(), face_matrix=load_mesh.t_pos_idx.cpu().numpy()))
    ms.save_current_mesh(str(output_path / 'tmp' / 'mesh.obj'), save_vertex_color=False)

    # TODO: Need these for rendering even if we don't optimize textures
    texture_map = texture.create_trainable(np.random.uniform(size=[512] * 2 + [3], low=0.0, high=1.0), [512] * 2, True)
    normal_map = texture.create_trainable(np.array([0, 0, 1]), [512] * 2, True)
    specular_map = texture.create_trainable(np.array([0, 0, 0]), [512] * 2, True)

    load_mesh = mesh.Mesh(
        material={
            'bsdf': cfg.bsdf,
            'kd': texture_map,
            'ks': specular_map,
            'normal': normal_map,
        },
        base=load_mesh  # Get UVs from original loaded mesh
    )
    if cfg.use_laplacian:
        from utilities.laplacian import find_edges, compute_laplacian_uniform
        edges = find_edges(load_mesh.t_pos_idx)
        laplacian = compute_laplacian_uniform(edges, load_mesh.v_pos.shape[0], device)
        laplacian.requires_grad_(False)
    if cfg.use_normal_sim:
        from utilities.normal_consistency import find_connected_faces, compute_normals, normal_consistency_loss
        connected_faces = find_connected_faces(load_mesh.t_pos_idx)
        connected_faces.requires_grad_(False)

    if cfg.part_deform:
        deform_part = obj.load_obj(cfg.part_file)
        vertices_all = load_mesh.v_pos
        faces_all = load_mesh.t_pos_idx
        vertices_deform = deform_part.v_pos
        first_vertice = vertices_deform[0]
        deform_indices = torch.empty(vertices_deform.shape[0], dtype=torch.int64)

        for i, vertex in enumerate(vertices_deform):
            is_equal = torch.all(torch.eq(vertices_all, vertex), dim=1)
            index = torch.nonzero(is_equal, as_tuple=False)

            if index.numel() > 0:
                deform_indices[i] = index[0][0]
            else:
                raise RuntimeError("NO matching vertex found!")
                # deform_indices[i] = -1  # No match found, use -1 as a placeholder

        # base_vertices = torch.tensor([list(vertex) for vertex in (set_all - set_deform)])

        jacobian_source = SourceMesh.SourceMesh(0, cfg.part_file, {}, 1, ttype=torch.float)
    elif cfg.part_jacobian:
        with open(cfg.part_file, "r") as file:
            origin_index = []
            for line in file:
                values = line.split()
                index = int(values[1])
                origin_index.append(index)

        face_indices = torch.tensor(origin_index)
        jacobian_source = SourceMesh.SourceMesh(0, str(output_path / 'tmp' / 'mesh.obj'), {}, 1, ttype=torch.float)
    else:
        jacobian_source = SourceMesh.SourceMesh(0, str(output_path / 'tmp' / 'mesh.obj'), {}, 1, ttype=torch.float)

    if len(list((output_path / 'tmp').glob('*.npz'))) > 0:
        logging.warning(
            f'Using existing Jacobian .npz files in {str(output_path)}/tmp/ ! Please check if this is intentional.')
    jacobian_source.load()
    jacobian_source.to(device)

    if cfg.use_jacobian:
        if cfg.part_deform:
            with torch.no_grad():
                gt_jacobians = jacobian_source.jacobians_from_vertices(deform_part.v_pos.unsqueeze(0))
        elif cfg.part_jacobian:
            with torch.no_grad():
                gt_jacobians = jacobian_source.jacobians_from_vertices(load_mesh.v_pos.unsqueeze(0))
        else:
            with torch.no_grad():
                gt_jacobians = jacobian_source.jacobians_from_vertices(load_mesh.v_pos.unsqueeze(0))

    train_params = []
    if cfg.use_jacobian:
        if cfg.part_jacobian:
            # gt_jacobians.requires_grad_(True)
            train_jacobians = gt_jacobians[:, face_indices].clone().detach().requires_grad_(True)
            # all_indices = torch.arange(gt_jacobians.shape[1])
            mask_within_slice = torch.zeros(gt_jacobians.shape[1], dtype=torch.bool)
            mask_within_slice[face_indices] = True
            mask_outside_slice = ~mask_within_slice
            indices_outside_slice = torch.nonzero(mask_outside_slice).squeeze()
            no_train_jacobians = gt_jacobians[:, indices_outside_slice].clone().detach().requires_grad_(True)
            # train_jacobians = torch.tensor(gt_jacobians[:, face_indices].cpu().numpy(), dtype=torch.float32, requires_grad=True)
            # gt_jacobians.requires_grad_(True)
            # train_jacobians.requires_grad_(True)
            train_params += [train_jacobians]
        else:
            gt_jacobians.requires_grad_(True)
            train_params += [gt_jacobians]
    else:
        load_mesh.v_pos.requires_grad_(True)
        train_params += [load_mesh.v_pos]

    if cfg.train_tex:
        train_params += texture_map.getMips()

    optimizer = torch.optim.Adam(train_params, lr=cfg.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)
    if cfg.multi_image:
        if cfg.oro_view:
            cams_data = MultiCameraBatch(
                cfg.train_res,
                [cfg.dist_min, cfg.dist_max],
                [cfg.azim_min, cfg.azim_max],
                [cfg.elev_alpha, cfg.elev_beta, cfg.elev_max],
                [cfg.fov_min, cfg.fov_max],
                cfg.aug_loc,
                cfg.aug_light,
                cfg.aug_bkg,
                cfg.batch_size,
                rand_solid=True
            )
        else:
            cams_data = CameraBatch(
                cfg.train_res,
                [cfg.dist_min, cfg.dist_max],
                [cfg.azim_min, cfg.azim_max],
                [cfg.elev_alpha, cfg.elev_beta, cfg.elev_max],
                [cfg.fov_min, cfg.fov_max],
                cfg.aug_loc,
                cfg.aug_light,
                cfg.aug_bkg,
                cfg.batch_size * 4,
                rand_solid=True
            )
    else:
        cams_data = CameraBatch(
            cfg.train_res,
            [cfg.dist_min, cfg.dist_max],
            [cfg.azim_min, cfg.azim_max],
            [cfg.elev_alpha, cfg.elev_beta, cfg.elev_max],
            [cfg.fov_min, cfg.fov_max],
            cfg.aug_loc,
            cfg.aug_light,
            cfg.aug_bkg,
            cfg.batch_size,
            rand_solid=True
        )
    # if cfg.multi_image:
    #     # we concat 4 view to make one
    #     # print(f"batch size {cfg.batch_size * 4}")
    #     cams = torch.utils.data.DataLoader(cams_data, cfg.batch_size, num_workers=0, pin_memory=True)
    # else:
    cams = torch.utils.data.DataLoader(cams_data, cams_data.batch_size, num_workers=0, pin_memory=True)

    # best_losses = {'CLIP': np.inf, 'total': np.inf}

    for out_type in ['final', 'best_clip', 'best_total']:
        os.makedirs(output_path / f'mesh_{out_type}', exist_ok=True)
    os.makedirs(output_path / 'images', exist_ok=True)
    # logger = SummaryWriter(str(output_path / 'logs'))

    rot_ang = 0.0
    t_loop = tqdm(range(cfg.epochs + 1), leave=False)

    if cfg.resize_method == 'cubic':
        resize_method = cubic
    elif cfg.resize_method == 'linear':
        resize_method = linear
    elif cfg.resize_method == 'lanczos2':
        resize_method = lanczos2
    elif cfg.resize_method == 'lanczos3':
        resize_method = lanczos3

    # scaler = GradScaler()
    for it in t_loop:
        # print(it)
        # jacboians_before = train_jacobians.clone()
        if cfg.use_jacobian:
            # updated vertices from jacobians
            if cfg.part_deform:
                deformed_part = jacobian_source.vertices_from_jacobians(gt_jacobians).squeeze()
                deformed_part += first_vertice
                n_vert = vertices_all.clone()
                n_vert[deform_indices] = deformed_part
                # for i, idx in enumerate(deform_indices):
                #     n_vert[idx] = deformed_part[i]
            elif cfg.part_jacobian:
                all_jacobians = torch.zeros_like(gt_jacobians, requires_grad=False)
                all_jacobians[:, face_indices] = train_jacobians.clone()
                all_jacobians[:, indices_outside_slice] = no_train_jacobians.clone()
                # gt_jacobians[:, face_indices] = train_jacobians
                n_vert = jacobian_source.vertices_from_jacobians(all_jacobians).squeeze()
            else:
                n_vert = jacobian_source.vertices_from_jacobians(gt_jacobians).squeeze()
        else:
            n_vert = load_mesh.v_pos
        # TODO: More texture code required to make it work ...
        if cfg.train_tex and it > 15000:
            ready_texture = texture.Texture2D(
                kornia.filters.gaussian_blur2d(
                    load_mesh.material['kd'].data.permute(0, 3, 1, 2),
                    kernel_size=(7, 7),
                    sigma=(3, 3),
                ).permute(0, 2, 3, 1).contiguous()
            )
            grad_clamp = True
        else:
            ready_texture = texture.Texture2D(torch.full_like(load_mesh.material['kd'].data, 0.5))
            grad_clamp = False

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

        # Final mesh

        m = mesh.Mesh(
            n_vert,
            load_mesh.t_pos_idx,
            material={
                'bsdf': cfg.bsdf,
                'kd': ready_texture,
                'ks': ready_specular,
                'normal': ready_normal,
            },
            base=load_mesh  # gets uvs etc from here
        )

        render_mesh = create_scene([m.eval()], cfg, sz=512)
        # if it == 0:
        #     base_mesh = render_mesh.clone()
        #     base_mesh = mesh.auto_normals(base_mesh)
        #     base_mesh = mesh.compute_tangents(base_mesh)
        render_mesh = mesh.auto_normals(render_mesh)
        render_mesh = mesh.compute_tangents(render_mesh)

        # Logging mesh
        if it % cfg.log_interval == 0:
            with torch.no_grad():
                params = get_camera_params(
                    cfg.log_elev,
                    rot_ang,
                    cfg.log_dist,
                    cfg.log_res,
                    cfg.log_fov,
                )
                rot_ang += 1
                log_mesh = mesh.unit_size(render_mesh.eval(params))
                log_image = render.render_mesh(
                    glctx,
                    log_mesh,
                    params['mvp'],
                    params['campos'],
                    params['lightpos'],
                    cfg.log_light_power,
                    cfg.log_res,
                    1,
                    background=torch.ones(1, cfg.log_res, cfg.log_res, 3).to(device)
                )

                log_image = video.ready_image(log_image)
                # logger.add_mesh('predicted_mesh', vertices=log_mesh.v_pos.unsqueeze(0),
                #                 faces=log_mesh.t_pos_idx.unsqueeze(0), global_step=it)

        if cfg.adapt_dist and it > 0:
            with torch.no_grad():
                v_pos = m.v_pos.clone()
                vmin = v_pos.amin(dim=0)
                vmax = v_pos.amax(dim=0)
                v_pos -= (vmin + vmax) / 2
                mult = torch.cat([v_pos.amin(dim=0), v_pos.amax(dim=0)]).abs().amax().cpu()
                cams.dataset.dist_min = cfg.dist_min * mult
                cams.dataset.dist_max = cfg.dist_max * mult

        params_camera = next(iter(cams))
        if cfg.multi_image and cfg.oro_view:
            for key in params_camera:
                # make it looks like 4 batch
                params_camera[key] = torch.cat(params_camera[key], dim=0)
        
        for key in params_camera:
            params_camera[key] = params_camera[key].to(device)
        # print(params_camera.size())
        final_mesh = render_mesh.eval(params_camera)

        train_render = render.render_mesh(
            glctx,
            final_mesh,
            params_camera['mvp'],
            params_camera['campos'],
            params_camera['lightpos'],
            cfg.light_power,
            cfg.train_res,
            spp=1,
            num_layers=1,
            msaa=False,
            background=params_camera['bkgs']
        ).permute(0, 3, 1, 2)
        # train_render = resize(train_render, out_shape=(224, 224), interp_method=resize_method)
        square_concated = []
        if cfg.multi_image:
            index = [0, 1, 2, 3]
            random.shuffle(index)
            for i in range(cfg.batch_size):
                top = torch.cat((train_render[index[0]+i*4], train_render[index[1]+i*4]), dim=2)
                bot = torch.cat((train_render[index[2]+i*4], train_render[index[3]+i*4]), dim=2)
                square_concated.append(torch.cat((top, bot), dim=1))
            train_render = torch.stack(square_concated, dim=0)
        # print(f"shape: {train_render.shape}")
        train_rast_map = render.render_mesh(
            glctx,
            final_mesh,
            params_camera['mvp'],
            params_camera['campos'],
            params_camera['lightpos'],
            cfg.light_power,
            cfg.train_res,
            spp=1,
            num_layers=1,
            msaa=False,
            background=params_camera['bkgs'],
            return_rast_map=True
        )

        if it == 0:
            params_camera = next(iter(cams))
            if cfg.multi_image and cfg.oro_view:
                for key in params_camera:
                    # make it looks like 4 batch
                    params_camera[key] = torch.cat(params_camera[key], dim=0)

            for key in params_camera:
                params_camera[key] = params_camera[key].to(device)
        # base_render = render.render_mesh(
        #     glctx,
        #     base_mesh.eval(params_camera),
        #     params_camera['mvp'],
        #     params_camera['campos'],
        #     params_camera['lightpos'],
        #     cfg.light_power,
        #     cfg.train_res,
        #     spp=1,
        #     num_layers=1,
        #     msaa=False,
        #     background=params_camera['bkgs'],
        # ).permute(0, 3, 1, 2)
        # base_render = resize(base_render, out_shape=(224, 224), interp_method=resize_method)

        if it % cfg.log_interval_im == 0:
            log_idx = torch.randperm(cfg.batch_size)[:5]
            s_log = train_render[log_idx, :, :, :]
            s_log = torchvision.utils.make_grid(s_log)
            ndarr = s_log.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(str(output_path / 'images' / f'epoch_{it}.png'))

        if it % cfg.log_interval_mesh == 0 and it > 0:
            os.makedirs(str(output_path / f'mesh_{it}'), exist_ok=True)
            obj.write_obj(
                str(output_path / f'mesh_{it}'),
                m.eval()
            )

        # SDS loss
        # train_render is the image of the mesh
        # add view prompt

        batch_size = cfg.batch_size
        text_z = [embeddings['empty']] * batch_size

        if cfg.view_prompt:
            azim = params_camera['azim']
            for b in range(batch_size):
                if azim[b] >= 90 and azim[b] < 90:
                    if azim[b] >= 0:
                        r = 1 - azim[b] / 90
                    else:
                        r = 1 + azim[b] / 90
                    start_z = embeddings['front']
                    end_z = embeddings['side']
                else:
                    if azim[b] >= 0:
                        r = 2 - azim[b] / 90
                    else:
                        r = 2 + azim[b] / 90
                    start_z = embeddings['back']
                    end_z = embeddings['side']
                text_z.append(r * start_z + (1 - r) * end_z - start_z)
        else:
            for b in range(batch_size):
                text_z.append(embeddings['default'])

        input_text_embeds = torch.cat(text_z, dim=0).to(device)

        # with autocast():
        #     start_time = time.time()
        # if it < 10000:
        #     guidance_scale = 100
        # else:
        #     guidance_scale = 100 - (100 - 20) * (it - 10000) / (cfg.epochs - 10000)
        if it % cfg.log_interval_im == 0:
            guidance_path = pathlib.Path(cfg.output_path + '/guidance/epoch_'+str(it)+'.png')
            loss_diffusion = sd.train_step(input_text_embeds, train_render, it, guidance_scale=cfg.guidance_scale, save_guidance_path=guidance_path, anneal=cfg.anneal, grad_clamp=grad_clamp)
        else:
            loss_diffusion = sd.train_step(input_text_embeds, train_render, it, guidance_scale=cfg.guidance_scale, anneal=cfg.anneal, grad_clamp=grad_clamp)
        # end_time = time.time()
        # sds_time = end_time - start_time
        # Jacobian regularization
        # start_time = time.time()
        if cfg.use_view_consistency:
            normalized_clip_render = (train_render - clip_mean[None, :, None, None]) / clip_std[None, :, None, None]
            curr_vp_map = get_vp_map(final_mesh.v_pos, params_camera['mvp'], 224)
            for idx, rast_faces in enumerate(train_rast_map[:, :, :, 3].view(cfg.batch_size, -1)):
                u_faces = rast_faces.unique().long()[1:] - 1
                t = torch.arange(len(final_mesh.v_pos), device=device)
                u_ret = torch.cat([t, final_mesh.t_pos_idx[u_faces].flatten()]).unique(return_counts=True)
                non_verts = u_ret[0][u_ret[1] < 2]
                curr_vp_map[idx][non_verts] = torch.tensor([224, 224], device=device)

            # Get mapping from vertex to patch
            med = (fe.old_stride - 1) / 2
            curr_vp_map[curr_vp_map < med] = med
            curr_vp_map[(curr_vp_map > 224 - fe.old_stride) & (curr_vp_map < 224)] = 223 - med
            curr_patch_map = ((curr_vp_map - med) / fe.new_stride).round()
            flat_patch_map = curr_patch_map[..., 0] * (((224 - fe.old_stride) / fe.new_stride) + 1) + curr_patch_map[..., 1]

            # Deep features
            patch_feats = fe(normalized_clip_render)
            flat_patch_map[flat_patch_map > patch_feats[0].shape[-1] - 1] = patch_feats[0].shape[-1]
            flat_patch_map = flat_patch_map.long()[:, None, :].repeat(1, patch_feats[0].shape[1], 1)

            deep_feats = patch_feats[cfg.consistency_vit_layer]
            deep_feats = torch.nn.functional.pad(deep_feats, (0, 1))
            deep_feats = torch.gather(deep_feats, dim=2, index=flat_patch_map)
            deep_feats = torch.nn.functional.normalize(deep_feats, dim=1, eps=1e-6)

            elev_d = torch.cdist(params_camera['elev'].unsqueeze(1),
                                 params_camera['elev'].unsqueeze(1)).abs() < torch.deg2rad(
                torch.tensor(cfg.consistency_elev_filter))
            azim_d = torch.cdist(params_camera['azim'].unsqueeze(1),
                                 params_camera['azim'].unsqueeze(1)).abs() < torch.deg2rad(
                torch.tensor(cfg.consistency_azim_filter))

            cosines = torch.einsum('ijk, lkj -> ilk', deep_feats, deep_feats.permute(0, 2, 1))
            cosines = (cosines * azim_d.unsqueeze(-1) * elev_d.unsqueeze(-1)).permute(2, 0, 1).triu(1)
            consistency_loss = cosines[cosines != 0].mean()

        # end_time = time.time()
        # regularization_time = end_time - start_time
        # logger.add_scalar('jacobian_regularization', r_loss, global_step=it)
        total_loss = cfg.diffusion_weight * loss_diffusion

        if cfg.use_jacobian:
            r_loss = (((gt_jacobians) - torch.eye(3, 3, device=device)) ** 2).mean()
            total_loss += cfg.regularize_jacobians_weight * r_loss
        if cfg.use_laplacian:
            l_loss = laplacian.mm(n_vert).norm(dim=1) ** 2
            l_loss = l_loss.mean()
            total_loss += cfg.regularize_laplacian_weight * l_loss
        if cfg.use_normal_sim:
            face_norms = compute_normals(n_vert, load_mesh.t_pos_idx)
            n_loss = normal_consistency_loss(face_norms, connected_faces)
            total_loss += cfg.regularize_normal_weight * n_loss

        # diffusion
        optimizer.zero_grad()
        # start_time = time.time()
        total_loss.backward()
        # end_time = time.time()
        # if cfg.use_jacobian:
            # t_loop.set_description(f"jacobian: {r_loss * cfg.regularize_jacobians_weight},"
            #                    f" laplacian: {l_loss * cfg.regularize_laplacian_weight},"
            #                    f" normal: {n_loss * cfg.regularize_normal_weight},"
            #                    )
        optimizer.step()

        # if torch.equal(jacboians_before, train_jacobians):
        #     print("same!")
        # else:
        #     print("not same")
        # scheduler.step()

    video.close()
    obj.write_obj(
        str(output_path / 'mesh_final'),
        m.eval()
    )

    return
