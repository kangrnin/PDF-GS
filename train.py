import os
import sys
import uuid
import torch
import torch.nn.functional as F
from random import randint
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.mask_utils import DINOv3FeatureExtractor
from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False


def compute_clean_mask(gt_feat, prev_feat, sim_thr, img_hw):
    cos = F.cosine_similarity(gt_feat, prev_feat, dim=0, eps=1e-8)
    clean = (cos >= sim_thr).float().unsqueeze(0).unsqueeze(0)
    H, W = img_hw
    clean = F.interpolate(clean, size=(H, W), mode="nearest").squeeze(1)
    distractor = 1.0 - clean
    distractor = F.max_pool2d(distractor.unsqueeze(0), kernel_size=15, stride=1, padding=7).squeeze(0)
    return (1.0 - distractor).clamp(0, 1)


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, phase, prev_renders,
             sim_thr, prev_model, num_phases, color_update_interval):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("Sparse Adam requested but not installed. `pip install 3dgs_accel`")

    feature_extractor = DINOv3FeatureExtractor().cuda()

    prepare_output_and_logger(dataset)
    base_model_path = dataset.model_path
    phase_model_path = os.path.join(base_model_path, f"phase_{phase}")
    os.makedirs(phase_model_path, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, load_iteration=None)
    scene.model_path = phase_model_path

    train_cams = scene.getTrainCameras()
    up_res = 50

    for cam in tqdm(train_cams, desc=f"[Phase {phase}] DINO GT feature"):
        gt_img_cuda = cam.original_image.to("cuda", non_blocking=True)
        with torch.no_grad():
            cam.gt_feat = feature_extractor(gt_img_cuda, up_res).cpu()

    for cam in tqdm(train_cams, desc=f"[Phase {phase}] DINO prev_phase feature"):
        if prev_renders is not None and cam.image_name in prev_renders:
            prev_img_cuda = prev_renders[cam.image_name].to("cuda", non_blocking=True)
            with torch.no_grad():
                cam.prev_feat = feature_extractor(prev_img_cuda, up_res).cpu()
        else:
            cam.prev_feat = None

    gaussians.training_setup(opt)
    device = gaussians._xyz.device

    if checkpoint and phase == 1:
        (model_params, _) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    elif prev_model is not None:
        gaussians.restore(prev_model, opt)
        gaussians._features_rest.data = torch.zeros_like(gaussians._features_rest.data, device=device)
        gaussians.active_sh_degree = 0

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0
    prev_mask_dict = {}

    progress_bar = tqdm(range(1, opt.iterations + 1), desc=f"Phase {phase} Training")
    for iteration in range(1, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        rand_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        if visibility_filter is not None and hasattr(visibility_filter, "device") and visibility_filter.device != gaussians._xyz.device:
            visibility_filter = visibility_filter.to(gaussians._xyz.device)
        radii = render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            image *= viewpoint_cam.alpha_mask.cuda()

        gt_orig = viewpoint_cam.original_image.cuda()
        gt_feat = viewpoint_cam.gt_feat
        prev_feat = viewpoint_cam.prev_feat

        if prev_feat is None:
            clean_mask = torch.ones((1, gt_orig.shape[1], gt_orig.shape[2]),
                                    device=gt_orig.device, dtype=gt_orig.dtype)
        else:
            clean_mask = compute_clean_mask(
                gt_feat=gt_feat.to(gt_orig.device),
                prev_feat=prev_feat.to(gt_orig.device),
                sim_thr=sim_thr,
                img_hw=(gt_orig.shape[1], gt_orig.shape[2]),
            )
            prev = prev_mask_dict.get(viewpoint_cam.image_name)
            if prev is not None:
                clean_mask = clean_mask * prev.to(clean_mask.device)
            prev_mask_dict[viewpoint_cam.image_name] = clean_mask.detach().cpu()

        loss_map_l1 = F.l1_loss(image, gt_orig, reduction='none') * clean_mask.unsqueeze(0)
        Ll1 = loss_map_l1.mean()

        loss_map_ssim = ssim(image, gt_orig, size_average=False) * clean_mask.unsqueeze(0)
        ssim_value = loss_map_ssim.mean()

        dssim = 1.0 if phase == 1 else 0.2
        loss = (1.0 - dssim) * Ll1 + dssim * (1.0 - ssim_value)
        loss_dssim_default = 0.8 * Ll1 + 0.2 * (1.0 - ssim_value)
        loss = loss * (loss_dssim_default.item() / max(loss.item(), 1e-8))

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render,
                            (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                            dataset.train_test_exp, phase=phase)
            if iteration in saving_iterations:
                print(f"\n[PHASE {phase} | ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                if iteration > opt.densify_from_iter:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005,
                                                scene.cameras_extent, None, radii)
                if iteration % opt.opacity_reset_interval == 0 or \
                   (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    features_dc_orig = gaussians._features_dc.data.clone()
                    features_rest_orig = gaussians._features_rest.data.clone()
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    if phase != num_phases and iteration % color_update_interval != 0:
                        gaussians._features_dc.data = features_dc_orig
                        gaussians._features_rest.data = features_rest_orig


    prev_model_out = gaussians.capture()
    next_renders = render_train_to_map(scene, gaussians, pipe, background, dataset.train_test_exp) if phase < num_phases else None
    return prev_model_out, next_renders


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID') or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,
                    scene, renderFunc, renderArgs, train_test_exp, phase):
    if iteration not in testing_iterations:
        return

    import numpy as np
    from PIL import Image

    def _to_uint8(t):
        t = torch.clamp(t, 0.0, 1.0).detach().cpu()
        if t.dim() == 4:
            t = t[0]
        return (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    torch.cuda.empty_cache()

    train_cams = scene.getTrainCameras()
    train_sample = [train_cams[idx] for idx in range(0, len(train_cams), max(1, len(train_cams) // 50))]
    validation_configs = (
        {'name': 'test',  'cameras': scene.getTestCameras()},
        {'name': 'train', 'cameras': train_sample},
    )
    renders_root = os.path.join(scene.model_path, "renders")
    os.makedirs(renders_root, exist_ok=True)

    for config in validation_configs:
        cams = config['cameras']
        if not cams:
            continue
        l1_accum = 0.0
        psnr_accum = 0.0
        out_dir = os.path.join(renders_root, f"phase{phase}_iter_{iteration}_{config['name']}")
        os.makedirs(out_dir, exist_ok=True)

        for idx, viewpoint in enumerate(cams):
            pack = renderFunc(viewpoint, scene.gaussians, *renderArgs)
            image = torch.clamp(pack["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            if train_test_exp:
                image = image[..., image.shape[-1] // 2:]
                gt_image = gt_image[..., gt_image.shape[-1] // 2:]

            l1_accum += l1_loss(image, gt_image).mean().double()
            psnr_accum += psnr(image, gt_image).mean().double()

            base = os.path.splitext(viewpoint.image_name)[0]
            Image.fromarray(_to_uint8(image)).save(os.path.join(out_dir, f"{base}.png"))
            Image.fromarray(_to_uint8(gt_image)).save(os.path.join(out_dir, f"{base}_gt.png"))

        psnr_mean = (psnr_accum / len(cams)).item()
        l1_mean = (l1_accum / len(cams)).item()
        print(f"\n[PHASE {phase} | ITER {iteration}] Evaluating {config['name']}: L1 {l1_mean:.6f} PSNR {psnr_mean:.2f}")

    torch.cuda.empty_cache()


@torch.no_grad()
def render_train_to_map(scene, gaussians, pipe, background, use_trained_exp):
    out = {}
    for cam in scene.getTrainCameras():
        pack = render(cam, gaussians, pipe, background, use_trained_exp=use_trained_exp,
                      separate_sh=SPARSE_ADAM_AVAILABLE)
        out[cam.image_name] = torch.clamp(pack["render"], 0.0, 1.0).detach()
    return out


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=randint(10000, 20000))
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--num_phases", type=int, default=4)
    parser.add_argument("--iter_per_phase", type=int, required=True)
    parser.add_argument("--sim_thr", type=float, nargs='+', required=True)
    parser.add_argument('--color_update_interval', type=int, required=True)

    args = parser.parse_args(sys.argv[1:])
    args.iterations = args.iter_per_phase
    args.position_lr_max_steps = args.iter_per_phase
    args.densify_until_iter = args.iter_per_phase
    args.save_iterations.append(args.iterations)

    print("Optimizing " + str(args.model_path))
    safe_state(args.quiet)

    dataset = lp.extract(args)
    pipe = pp.extract(args)

    assert len(args.sim_thr) == 1 or len(args.sim_thr) == args.num_phases - 1

    renders = None
    prev_model = None
    for phase in range(1, args.num_phases + 1):
        opt = op.extract(args)
        if phase == args.num_phases:
            opt.densify_until_iter = args.iter_per_phase // 2
        sim_thr = args.sim_thr[0] if len(args.sim_thr) == 1 else args.sim_thr[max(phase - 2, 0)]

        prev_model, renders = training(
            dataset, opt, pipe,
            testing_iterations=[opt.iterations],
            saving_iterations=[opt.iterations],
            checkpoint_iterations=[],
            checkpoint=args.start_checkpoint if (phase == 1 and args.start_checkpoint) else None,
            phase=phase,
            prev_renders=renders,
            sim_thr=sim_thr,
            prev_model=prev_model,
            num_phases=args.num_phases,
            color_update_interval=args.color_update_interval,
        )

        if phase != args.num_phases - 1:
            prev_model = None

    print("\nTraining complete.")
