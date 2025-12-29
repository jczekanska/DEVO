import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
from devo.config import cfg
import numpy as np
import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND","Agg"))
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from utils.load_utils import load_gt_us, rpg_evs_iterator
from utils.eval_utils import assert_eval_config, run_voxel
from utils.eval_utils import log_results, write_raw_results, compute_median_results
from utils.viz_utils import viz_flow_inference

def _read_stamped_traj(path):
    """
    reads stamped trajectory files and returns (times_s, positions Nx3)
    """
    times = []
    pos = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # some files are "timestamp tx ty tz qx qy qz qw"
            # other formats may be "timestamp tx ty tz ..." - we only care about first 4 numbers
            try:
                t = float(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
            except Exception:
                continue
            times.append(t)
            pos.append([x, y, z])
    return np.array(times, dtype=float), np.array(pos, dtype=float)


def _compute_vel_central(times_s, pos):
    """
    computes vel per timestamp using central differences
    times_s: (N,) in seconds
    pos: (N,3)
    returns times_s (same), vel (N,3)
    """
    times_s = np.asarray(times_s, dtype=float)
    pos = np.asarray(pos, dtype=float)
    N = pos.shape[0]
    if N == 0:
        return times_s, np.zeros((0,3))
    vel = np.zeros_like(pos)
    if N == 1:
        return times_s, vel
    
    # central differences
    for i in range(1, N-1):
        dt = times_s[i+1] - times_s[i-1]
        if dt == 0:
            vel[i] = np.zeros(3)
        else:
            vel[i] = (pos[i+1] - pos[i-1]) / dt

    # endpoints
    dt0 = times_s[1] - times_s[0] if N > 1 else 1.0
    vel[0] = (pos[1] - pos[0]) / (dt0 if dt0 != 0 else 1.0)
    dtN = times_s[-1] - times_s[-2] if N > 1 else 1.0
    vel[-1] = (pos[-1] - pos[-2]) / (dtN if dtN != 0 else 1.0)
    return times_s, vel


def _interp_to_times(src_times, src_vals, tgt_times):
    """
    linearly interpolates src_vals (NxD) given src_times -> evaluate at tgt_times
    src_times in seconds
    """
    src_times = np.asarray(src_times, dtype=float)
    tgt_times = np.asarray(tgt_times, dtype=float)
    src_vals = np.asarray(src_vals, dtype=float)
    if src_vals.ndim == 1:
        src_vals = src_vals[:, None]
    D = src_vals.shape[1]
    out = np.zeros((tgt_times.shape[0], D), dtype=float)
    for d in range(D):
        out[:, d] = np.interp(tgt_times, src_times, src_vals[:, d])
    return out

def umeyama_align(A, B, need_scale=True):
    assert A.shape == B.shape and A.shape[1] == 3
    N = A.shape[0]
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    X = A - mu_A
    Y = B - mu_B
    cov = (Y.T @ X) / float(N)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    Rm = U @ S @ Vt
    varA = (X**2).sum() / float(N)
    if need_scale:
        s = np.trace(np.diag(D) @ S) / varA
    else:
        s = 1.0
    t = mu_B - s * Rm @ mu_A
    return s, Rm, t

def apply_sim3(pts, s, Rm, t):
    return (s * (pts @ Rm.T) + t)

def estimate_time_offset_scalar(a, b, dt, max_offset_s=0.5):
    a = a - np.mean(a)
    b = b - np.mean(b)
    N = len(a)
    if N < 3:
        return 0.0
    maxlags = min(N-1, int(max_offset_s / (dt + 1e-12)))
    corr = np.correlate(b, a, mode='full')
    lags = np.arange(-N+1, N)
    mask = (lags >= -maxlags) & (lags <= maxlags)
    sub = corr[mask]
    sel = np.argmax(sub)
    lag = lags[mask][sel]
    return lag * dt

@torch.no_grad()
def evaluate(config, args, net, train_step=None, datapath="", split_file=None,
             stride=1, trials=1, plot=False, save=False, return_figure=False, viz=False, timing=False, side='left', viz_flow=False):
    dataset_name = "rpg_evs"
    assert side == "left" or side == "right"

    if config is None:
        config = cfg
        config.merge_from_file("config/eval_rpg.yaml")

    scenes = open(split_file).read().split()
    scenes = [s for s in scenes if '#' not in s]

    results_dict_scene, figures = {}, {}
    all_results = []
    for i, scene in enumerate(scenes):
        if "simulation_3planes" in scene:
            H, W = 260, 346
            pose_freq = 1000
        else:
            H, W = 180, 240
            pose_freq = 200

        traj_hf_path = os.path.join(datapath, scene, f"gt_stamped_{side}.txt")
        if not os.path.exists(traj_hf_path):
            print(f"scene {scene} has no GT, skipping")
            continue

        print(f"Eval on {scene}")
        results_dict_scene[scene] = []

        for trial in range(trials):
            datapath_val = os.path.join(datapath, scene)

            # run the slam system
            traj_est, tstamps, flowdata = run_voxel(datapath_val, config, net, viz=viz, 
                                          iterator=rpg_evs_iterator(datapath_val, side=side, stride=stride, timing=timing, dT_ms=None, H=H, W=W), # optionally pass DELTA_MS
                                          timing=timing, H=H, W=W, viz_flow=viz_flow)

            # load traj
            tss_traj_us, traj_hf = load_gt_us(traj_hf_path)
 
            # do evaluation 
            data = (traj_hf, tss_traj_us, traj_est, tstamps)
            hyperparam = (train_step, net, dataset_name, scene, trial, cfg, args)
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                   plot=plot, save=save, return_figure=return_figure, stride=stride,
                                                                   expname=args.expname)
            
            if viz_flow:
                viz_flow_inference(outfolder, flowdata)

            # --- compute velocities from saved stamped files produced by log_results ---
            stamped_est = None
            stamped_gt = None
            
            for root, dirs, files in os.walk(outfolder):
                for fname in files:
                    if fname == "stamped_traj_estimate.txt":
                        stamped_est = os.path.join(root, fname)
                    if fname == "stamped_groundtruth.txt":
                        stamped_gt = os.path.join(root, fname)
            if stamped_est is None:
                print("Warning: stamped_traj_estimate.txt not found under", outfolder)
            if stamped_gt is None:
                print("Warning: stamped_groundtruth.txt not found under", outfolder)

            if stamped_est is not None and stamped_gt is not None:
                try:
                    est_times, est_pos = _read_stamped_traj(stamped_est)
                    gt_times, gt_pos = _read_stamped_traj(stamped_gt)

                    # --- sim(3) alignment (umeyama) using close timestamp correspondences ---
                    est_pos_aligned = est_pos.copy()
                    s_val = 1.0
                    
                    max_match_dt = 0.02
                    matches_est = []
                    matches_gt = []
                    for ie, te in enumerate(est_times):
                        j = np.argmin(np.abs(gt_times - te))
                        if abs(gt_times[j] - te) <= max_match_dt:
                            matches_est.append(ie)
                            matches_gt.append(j)
                    if len(matches_est) >= 6:
                        A = est_pos[np.array(matches_est)]
                        B = gt_pos[np.array(matches_gt)]
                        s_val, Rm, tvec = umeyama_align(A, B, need_scale=True)
                        est_pos_aligned = apply_sim3(est_pos, s_val, Rm, tvec)
                        print(f"[Umeyama] applied scale={s_val:.6f}, correspondences={len(matches_est)}")
                    else:
                        print(f"[Umeyama] too few correspondences ({len(matches_est)}), skipping Umeyama (max_match_dt={max_match_dt}s)")

                    est_times_s, est_vel_raw = _compute_vel_central(est_times, est_pos_aligned)
                    gt_times_s, gt_vel = _compute_vel_central(gt_times, gt_pos)

                    if est_vel_raw.shape[0] >= 10 and gt_vel.shape[0] >= 10:
                        try:
                            gt_on_est = _interp_to_times(gt_times_s, gt_vel, est_times_s)
                            median_dt = float(np.median(np.diff(est_times_s)))
                            lag = estimate_time_offset_scalar(est_vel_raw[:, 0], gt_on_est[:, 0], dt=median_dt, max_offset_s=0.5)
                            if abs(lag) > 1e-6:
                                est_times_s = est_times_s - lag
                                est_times = est_times - lag
                                print(f"[TimeOffset] applied lag={lag:.4f}s")
                                gt_on_est = _interp_to_times(gt_times_s, gt_vel, est_times_s)
                            else:
                                gt_on_est = _interp_to_times(gt_times_s, gt_vel, est_times_s)
                        except Exception as e:
                            print("Time-offset estimation failed:", e)
                            gt_on_est = _interp_to_times(gt_times_s, gt_vel, est_times_s)
                    else:
                        gt_on_est = _interp_to_times(gt_times_s, gt_vel, est_times_s)

                    def _rmse(a, b):
                        return np.sqrt(np.mean((a - b)**2, axis=0))
                    try:
                        rmse_raw = _rmse(est_vel_raw, gt_on_est)
                        print(f"Vel RMSE raw   (vx,vy,vz): {rmse_raw}")
                    except Exception:
                        pass

                    saved_dir = os.path.join(outfolder, "saved_results", "traj_est")
                    os.makedirs(saved_dir, exist_ok=True)
                    aligned_file = os.path.join(outfolder, "stamped_traj_estimate_pos_aligned.txt")
                    with open(aligned_file, 'w') as fa:
                        for t, p in zip(est_times_s, est_pos_aligned):
                            fa.write(f"{t:.9f} {p[0]:.9e} {p[1]:.9e} {p[2]:.9e} 0 0 0 1\n")

                    fname_raw = os.path.join(saved_dir, "vel_est_raw.txt")
                    fname_gt = os.path.join(saved_dir, "vel_gt_on_est_times.txt")
                    with open(fname_raw, "w") as fr, open(fname_gt, "w") as fg:
                        fr.write("# time[s] vx vy vz\n")
                        fg.write("# time[s] vx vy vz\n")
                        for t, vr, vg in zip(est_times_s, est_vel_raw, gt_on_est):
                            fr.write(f"{t:.9f} {vr[0]:.9e} {vr[2]:.9e}\n")
                            fg.write(f"{t:.9f} {vg[0]:.9e} {vg[2]:.9e}\n")

                    plot_dir = os.path.join(outfolder, "plots")
                    os.makedirs(plot_dir, exist_ok=True)
                    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
                    axs[0].plot(est_times_s - est_times_s[0], gt_on_est[:, 0], label="gt vx", linewidth=1.2)
                    axs[0].plot(est_times_s - est_times_s[0], est_vel_raw[:, 0], label="est_raw vx", alpha=0.7)
                    axs[0].set_ylabel("vx [m/s]"); axs[0].legend(); axs[0].grid(True)
                    axs[1].plot(est_times_s - est_times_s[0], gt_on_est[:, 1], label="gt vy", linewidth=1.2)
                    axs[1].plot(est_times_s - est_times_s[0], est_vel_raw[:, 1], label="est_raw vy", alpha=0.7)
                    axs[1].set_ylabel("vy [m/s]"); axs[1].legend(); axs[1].grid(True)
                    axs[2].plot(est_times_s - est_times_s[0], gt_on_est[:, 2], label="gt vz", linewidth=1.2)
                    axs[2].plot(est_times_s - est_times_s[0], est_vel_raw[:, 2], label="est_raw vz", alpha=0.7)
                    axs[2].set_ylabel("vz [m/s]"); axs[2].set_xlabel("time [s]"); axs[2].legend(); axs[2].grid(True)
                    fig.tight_layout()
                    plot_file = os.path.join(plot_dir, "velocities_est_vs_gt_improved.png")
                    fig.savefig(plot_file, dpi=200)
                    plt.close(fig)
                    print("Saved velocities and improved plot to", outfolder)
                except Exception as e:
                    print("Warning: failed to compute or save velocities:", e)

        print(scene, sorted(results_dict_scene[scene]))

    # write output to file with timestamp
    write_raw_results(all_results, outfolder)
    results_dict = compute_median_results(results_dict_scene, all_results, dataset_name, outfolder)
        
    if return_figure:
        return results_dict, figures
    return results_dict, None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/eval_rpg.yaml")
    parser.add_argument('--datapath', default='', help='path to dataset directory')
    parser.add_argument('--weights', default="DEVO.pth")
    parser.add_argument('--val_split', type=str, default="splits/rpg/rpg_val.txt")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--return_figs', action="store_true")
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--timing', action="store_true")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--side', type=str, default="left")
    parser.add_argument('--viz_flow', action="store_true")
    parser.add_argument('--expname', type=str, default="")

    args = parser.parse_args()
    assert_eval_config(args)

    cfg.merge_from_file(args.config)
    print("Running eval_event_camera_dataset.py with config...")
    print(cfg)

    torch.manual_seed(1234)
    
    args.plot = True
    args.save_trajectory = True
    # args.viz_flow = True
    val_results, val_figures = evaluate(cfg, args, args.weights, datapath=args.datapath, split_file=args.val_split, trials=args.trials, \
                       plot=args.plot, save=args.save_trajectory, return_figure=args.return_figs, viz=args.viz, timing=args.timing, \
                        side=args.side, stride=args.stride, viz_flow=args.viz_flow)
    
    print("val_results= \n")
    for k in val_results:
        print(k, val_results[k])
