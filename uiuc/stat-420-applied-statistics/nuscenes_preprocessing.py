"""
Preprocess nuScenes scenes for STAT420 cut-in project (ego-relative velocity, fixed categories)
-----------------------------------------------------------------------------------------------
Generates one row per actor (vehicle only):
  predictors:  distance_ahead, lateral_offset, forward_velocity, side_velocity, vehicle_type
  response:    lateral_pos_5s  (lateral position 5 s after anchor)

Anchor condition:
  x in [-10, 40] m ahead/behind, |y| in [3, 8] m lateral

Notes:
  • Relative velocity computed from world-frame relative displacement (obj - ego), then rotated into ego frame.
  • Filters cross-traffic: |side_velocity| > 5 m/s.
  • Threading (Windows-safe), ASCII logs, resume support.
"""

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
import os, time, re

# ------------------------- CONFIG -------------------------
DATA_ROOT   = r"C:\nuScenes"
VERSION     = "v1.0-trainval"
SCENE_LIMIT = 200
NUM_WORKERS = 3
HORIZON_S   = 5.0
OUTPUT_CSV  = "clean_cutin_dataset_ego_corrected.csv"
LOG_FILE    = "progress_log.txt"

# Anchor window (restore broader range)
X_MIN, X_MAX = 0.0, 40.0
Y_ABS_MIN, Y_ABS_MAX = 3.0, 8.0

# Keep true vehicle types
ALLOWED_CATEGORIES = {"car", "truck", "bus", "trailer"}

# -------------------------- INIT --------------------------
nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)

# ------------------------ HELPERS -------------------------
def get_sample_time(nusc, sample_token):
    s = nusc.get("sample", sample_token)
    ts = nusc.get("sample_data", s["data"]["LIDAR_TOP"])["timestamp"]
    return ts / 1e6  # µs → s

def ego_pose_from_sample(nusc, sample):
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    pos = np.array(ego["translation"], dtype=float)
    rot = Quaternion(ego["rotation"]).rotation_matrix
    return pos, rot

def world_to_ego(ego_pos, ego_rot, world_xyz):
    R_T = ego_rot.T
    return R_T @ (world_xyz - ego_pos)

def rotate_world_vec_to_ego(ego_rot, vec_world):
    # rotate a world-frame vector into ego frame at current time
    return ego_rot.T @ vec_world

def category_bucket(ann):
    parts = ann["category_name"].split(".")
    return parts[-1] if len(parts) > 1 else parts[0]

def compute_rel_velocity_ego(nusc, ann):
    """Relative velocity of actor w.r.t. ego, expressed in current ego frame (m/s)."""
    prev_token = ann["prev"]
    if not prev_token:
        return np.array([0.0, 0.0])

    prev_ann = nusc.get("sample_annotation", prev_token)
    samp_now  = nusc.get("sample", ann["sample_token"])
    samp_prev = nusc.get("sample", prev_ann["sample_token"])

    ts_now  = nusc.get("sample_data", samp_now["data"]["LIDAR_TOP"])["timestamp"]
    ts_prev = nusc.get("sample_data", samp_prev["data"]["LIDAR_TOP"])["timestamp"]
    dt = (ts_now - ts_prev) / 1e6
    if dt <= 0:
        return np.array([0.0, 0.0])

    ego_pos_now,  ego_rot_now  = ego_pose_from_sample(nusc, samp_now)
    ego_pos_prev, ego_rot_prev = ego_pose_from_sample(nusc, samp_prev)

    p_obj_now  = np.array(ann["translation"], dtype=float)
    p_obj_prev = np.array(prev_ann["translation"], dtype=float)

    # relative displacement in world frame
    rel_now_world  = (p_obj_now  - ego_pos_now)
    rel_prev_world = (p_obj_prev - ego_pos_prev)
    drel_world = rel_now_world - rel_prev_world

    v_rel_world = drel_world / dt
    v_rel_ego_now = rotate_world_vec_to_ego(ego_rot_now, v_rel_world)  # x=fwd, y=left
    return v_rel_ego_now[:2]

def find_ann_of_instance(nusc, sample, instance_token):
    for a in sample["anns"]:
        ann = nusc.get("sample_annotation", a)
        if ann["instance_token"] == instance_token:
            return a
    return None

# --------------------- CORE PROCESSING --------------------
def process_scene(scene_idx):
    start_time = time.time()
    scene = nusc.scene[scene_idx]
    out_rows = []

    # all samples + times
    sample_tokens, sample_times = [], []
    tok = scene["first_sample_token"]
    while tok:
        sample_tokens.append(tok)
        sample_times.append(get_sample_time(nusc, tok))
        s = nusc.get("sample", tok)
        tok = s["next"] if s["next"] != "" else None
    time_map = dict(zip(sample_tokens, sample_times))

    anchored_instances = set()

    for token in sample_tokens:
        sample = nusc.get("sample", token)
        timestamp = time_map[token]
        ego_pos, ego_rot = ego_pose_from_sample(nusc, sample)

        for a in sample["anns"]:
            ann = nusc.get("sample_annotation", a)
            inst = ann["instance_token"]
            if inst in anchored_instances:
                continue

            cat = category_bucket(ann)
            if cat not in ALLOWED_CATEGORIES:
                continue

            # position in ego frame
            rel_now = world_to_ego(ego_pos, ego_rot, np.array(ann["translation"], dtype=float))
            x_t0, y_t0 = float(rel_now[0]), float(rel_now[1])
            if not (X_MIN <= x_t0 <= X_MAX and Y_ABS_MIN <= abs(y_t0) <= Y_ABS_MAX):
                continue

            # ego-relative velocity
            v_lon_t0, v_lat_t0 = compute_rel_velocity_ego(nusc, ann)

            # filter cross-traffic
            if abs(v_lat_t0) > 5.0:
                continue

            # 5s future sample
            t_target = timestamp + HORIZON_S
            future_token = next((tok2 for tok2 in sample_tokens if time_map[tok2] >= t_target), None)
            if not future_token:
                print(f"Scene {scene_idx}: no future sample >= {HORIZON_S}s")
                continue

            future_sample = nusc.get("sample", future_token)
            fut_ann_token = find_ann_of_instance(nusc, future_sample, inst)
            if not fut_ann_token:
                print(f"Scene {scene_idx}: actor lost before {HORIZON_S}s horizon (instance={inst})")
                continue

            # future lateral pos in ego frame at future time
            fut_ego_pos, fut_ego_rot = ego_pose_from_sample(nusc, future_sample)
            fut_world = np.array(nusc.get("sample_annotation", fut_ann_token)["translation"], dtype=float)
            rel_fut = world_to_ego(fut_ego_pos, fut_ego_rot, fut_world)
            y_f = float(rel_fut[1])

            out_rows.append(dict(
                scene_id=scene_idx,
                instance_token=inst,
                distance_ahead=x_t0,
                lateral_offset=y_t0,
                forward_velocity=float(v_lon_t0),
                side_velocity=float(v_lat_t0),
                vehicle_type=cat,
                lateral_pos_5s=y_f
            ))
            anchored_instances.add(inst)

    # log
    elapsed = time.time() - start_time
    log_line = f"Scene {scene_idx:03d} done -> {len(out_rows)} rows in {elapsed/60:.1f} min"
    with open(LOG_FILE, "a", encoding="ascii", errors="replace") as f:
        f.write(log_line + "\n")
    print(log_line)
    return out_rows

# --------------------- RESUME SUPPORT ---------------------
completed_scenes = set()
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, encoding="ascii", errors="ignore") as f:
        for line in f:
            m = re.search(r"Scene\s+(\d+)\s+done", line)
            if m:
                completed_scenes.add(int(m.group(1)))

if completed_scenes:
    print(f"\nResume mode: {len(completed_scenes)} scenes already done -> skipping them.\n")

scene_indices = [i for i in range(min(SCENE_LIMIT, len(nusc.scene))) if i not in completed_scenes]

# ------------------------- MAIN --------------------------
print(f"Processing {len(scene_indices)} remaining scenes (vehicles only) with {NUM_WORKERS} threads...\n")
results = Parallel(n_jobs=NUM_WORKERS, backend="threading")(
    delayed(process_scene)(i) for i in tqdm(scene_indices)
)

rows = [r for s in results for r in s]
df = pd.DataFrame(rows)
if not df.empty:
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["distance_ahead","lateral_offset","lateral_pos_5s"])

out_path = os.path.join(DATA_ROOT, OUTPUT_CSV)
mode = "a" if os.path.exists(out_path) and completed_scenes else "w"
header = not (os.path.exists(out_path) and completed_scenes)
df.to_csv(out_path, index=False, mode=mode, header=header, encoding="ascii", errors="replace")

print(f"\nSaved {len(df)} new rows -> {out_path}")
print(f"Progress log written to {os.path.abspath(LOG_FILE)}")
