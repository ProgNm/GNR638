"""
GNR638 — inference.py
======================
Run:
    python inference.py --test_dir <absolute_path_to_test_dir>

Expects inside test_dir:
    patches/patch_0.png  patch_1.png  ...
    test.csv

Writes:
    ./submission.csv   (in the current working directory, NOT test_dir)
"""

import os, re, cv2, math, glob, argparse, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ──────────────────────────────────────────────────────────────
#  CLI  —  only --test_dir is required
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", type=str, required=True,
                    help="Absolute path to directory containing patches/ and test.csv")
args = parser.parse_args()

TEST_DIR  = Path(args.test_dir)
PATCH_DIR = TEST_DIR / "patches"
TEST_CSV  = TEST_DIR / "test.csv"
MAP_OUT   = TEST_DIR / "reconstructed_map.png"   # save map alongside test data
OUT_CSV   = Path("submission.csv")               # always in CWD

MODEL_NAME = "OpenGVLab/InternVL2-8B"

# ── GPU info ─────────────────────────────────────────────────
if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}  VRAM={vram:.1f}GB")
else:
    vram = 0
    print("[WARN] No CUDA GPU detected — will run on CPU (slow)")

# ══════════════════════════════════════════════════════════════
#  STEP 1 — PATCH LOADING
# ══════════════════════════════════════════════════════════════
def load_patches(patch_dir: Path):
    paths = sorted(
        glob.glob(str(patch_dir / "patch_*.png")),
        key=lambda p: int(re.search(r"patch_(\d+)", p).group(1))
    )
    if not paths:
        raise FileNotFoundError(f"No patches found in {patch_dir}")
    images, grays = [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Cannot read {p}")
        images.append(img)
        grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    ph, pw = images[0].shape[:2]
    print(f"[INFO] Loaded {len(images)} patches  size={pw}x{ph}px")
    return images, grays


def rotate_img(img, angle):
    if angle == 0:   return img
    if angle == 90:  return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


# ══════════════════════════════════════════════════════════════
#  STEP 2 — MULTI-PASS BFS STITCHER
#
#  patch_0 = top-left anchor (guaranteed by problem statement)
#  Algorithm: BFS template matching on boundary strips, 4 rotations
#  Pass 1 (conf=0.90) → Pass 2 (0.82) → Pass 3 (0.70) → Pass 4 (0.60)
#  Any remaining unplaced → grid fallback (no black tiles)
# ══════════════════════════════════════════════════════════════

CV2_ROTS = {
    0: None,
    90:  cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,
}


def bfs_pass(images, grays, ans, visited, confidence,
             patch_h, patch_w, f_size, s_zone, boundary_slack):
    queue = deque(sorted(visited))
    newly = 0

    while queue:
        i = queue.popleft()
        curr_x, curr_y, rot_i = ans[i]

        gi_rot = rotate_img(grays[i], rot_i) if rot_i != 0 else grays[i]
        fp = {
            "right":  gi_rot[:, -f_size:],
            "left":   gi_rot[:, :f_size],
            "bottom": gi_rot[-f_size:, :],
            "top":    gi_rot[:f_size, :],
        }

        unvisited = [idx for idx in range(len(images)) if idx not in visited]

        for j in unvisited:
            best_score = -1
            best_state = None

            for angle in [0, 90, 180, 270]:
                rot_j = rotate_img(grays[j], angle) if angle != 0 else grays[j]

                configs = [
                    ("right",  rot_j[:, :s_zone],   1,  0),
                    ("bottom", rot_j[:s_zone, :],    0,  1),
                    ("left",   rot_j[:, -s_zone:],  -1,  0),
                    ("top",    rot_j[-s_zone:, :],   0, -1),
                ]

                for side_i, search_strip, dx_dir, dy_dir in configs:
                    if np.std(fp[side_i]) < 5:
                        continue
                    if search_strip.shape[0] < fp[side_i].shape[0] or \
                       search_strip.shape[1] < fp[side_i].shape[1]:
                        continue

                    try:
                        res = cv2.matchTemplate(
                            search_strip, fp[side_i], cv2.TM_CCOEFF_NORMED)
                    except cv2.error:
                        continue

                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val < confidence:
                        continue

                    mx, my = max_loc
                    if dx_dir == 1:    tx, ty = (patch_w - f_size - mx), -my
                    elif dx_dir == -1: tx, ty = (-patch_w + s_zone - mx), -my
                    elif dy_dir == 1:  tx, ty = -mx, (patch_h - f_size - my)
                    else:              tx, ty = -mx, (-patch_h + s_zone - my)

                    gx = curr_x + tx
                    gy = curr_y + ty

                    if gx < -boundary_slack or gy < -boundary_slack:
                        continue

                    if max_val > best_score:
                        best_score = max_val
                        best_state = (gx, gy, angle)

            if best_state is not None:
                ans[j]   = best_state
                visited.add(j)
                queue.append(j)
                newly += 1

    return newly


def grid_dims(n):
    cols = round(math.sqrt(n))
    while cols > 1 and n % cols != 0:
        cols -= 1
    rows = n // cols
    if rows < cols:
        rows, cols = cols, rows
    return rows, cols


def stitch_patches(images, grays):
    num              = len(images)
    patch_h, patch_w = images[0].shape[:2]
    f_size           = max(10, patch_h // 10)
    s_zone           = max(20, patch_h // 4)
    boundary_slack   = patch_w * 0.6

    print(f"[STITCH] {num} patches  patch={patch_w}x{patch_h}  "
          f"f_size={f_size}  s_zone={s_zone}")

    ans     = {0: (0, 0, 0)}
    visited = {0}

    for pass_name, conf in [("Pass1 strict", 0.90), ("Pass2 medium", 0.82),
                             ("Pass3 relaxed", 0.70), ("Pass4 loose", 0.60)]:
        newly = bfs_pass(images, grays, ans, visited, conf,
                         patch_h, patch_w, f_size, s_zone, boundary_slack)
        print(f"  [{pass_name}] conf={conf:.2f}  +{newly}  "
              f"total={len(visited)}/{num}")
        if len(visited) == num:
            break

    # Grid fallback — guarantee no black tiles
    unplaced = set(range(num)) - visited
    if unplaced:
        print(f"[STITCH] Grid fallback for {len(unplaced)} patches ...")
        rows, cols = grid_dims(num)
        occupied   = set()
        for pid, (gx, gy, _) in ans.items():
            occupied.add((round(gy / patch_h), round(gx / patch_w)))
        ul = sorted(unplaced)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= len(ul): break
                if (r, c) not in occupied:
                    ans[ul[idx]] = (c * patch_w, r * patch_h, 0)
                    visited.add(ul[idx])
                    occupied.add((r, c))
                    idx += 1

    # Render
    min_x = min(v[0] for v in ans.values())
    min_y = min(v[1] for v in ans.values())
    max_x = max(v[0] for v in ans.values()) + patch_w
    max_y = max(v[1] for v in ans.values()) + patch_h
    canvas = np.zeros((int(max_y - min_y), int(max_x - min_x), 3), dtype=np.uint8)

    for pid in sorted(ans.keys()):
        gx, gy, angle = ans[pid]
        img = rotate_img(images[pid], angle)
        y, x = int(gy - min_y), int(gx - min_x)
        h = min(patch_h, canvas.shape[0] - y)
        w = min(patch_w, canvas.shape[1] - x)
        if h > 0 and w > 0 and x >= 0 and y >= 0:
            canvas[y:y+h, x:x+w] = img[:h, :w]

    black_pct = (canvas.sum(axis=2) == 0).mean() * 100
    print(f"[STITCH] Done. Canvas={canvas.shape[1]}x{canvas.shape[0]}  "
          f"black={black_pct:.1f}%")
    return canvas


# ══════════════════════════════════════════════════════════════
#  STEP 3 — VLM INFERENCE  (InternVL2-8B, full bfloat16)
#
#  On L40s (48GB) this fits comfortably with no quantization.
#  Dtype is unified after loading to prevent Half/BFloat16 crash.
# ══════════════════════════════════════════════════════════════

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def dynamic_preprocess(image: Image.Image, image_size=448, max_num=12):
    w, h    = image.size
    aspect  = w / h
    cols    = max(1, round(math.sqrt(max_num * aspect)))
    rows    = max(1, round(max_num / cols))
    resized = image.resize((image_size * cols, image_size * rows), Image.BICUBIC)
    tf      = build_transform(image_size)
    tiles   = []
    for r in range(rows):
        for c in range(cols):
            box = (c * image_size, r * image_size,
                   (c + 1) * image_size, (r + 1) * image_size)
            tiles.append(tf(resized.crop(box)))
    tiles.append(tf(image.resize((image_size, image_size), Image.BICUBIC)))
    return torch.stack(tiles)


def load_model():
    print(f"[INFO] Loading {MODEL_NAME} in bfloat16 ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_fast=False)

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()

    # Unify all float params/buffers to bfloat16
    # Prevents "Input BFloat16 vs bias Half" crash with accelerate
    with torch.no_grad():
        for p in model.parameters():
            if p.dtype in (torch.float16, torch.float32):
                p.data = p.data.to(torch.bfloat16)
        for b in model.buffers():
            if b.dtype in (torch.float16, torch.float32):
                b.data = b.data.to(torch.bfloat16)

    print("[INFO] Model ready.")
    return tokenizer, model


def answer_question(tokenizer, model, pil_image, question, options):
    opts_str = "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))
    prompt = (
        "You are an expert at reading geographic and satellite maps.\n"
        "Study the map image carefully — look for text labels, "
        "landmarks, water bodies, roads, and spatial relationships.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opts_str}\n\n"
        "Reply with ONLY a single digit: 1, 2, 3, or 4.\n"
        "If you are not confident, reply with 5.\n"
        "Do NOT explain. Just the number."
    )

    pixel_values = dynamic_preprocess(pil_image, max_num=12)
    device       = next(model.parameters()).device
    pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
    num_patches  = pixel_values.shape[0]

    gen_cfg     = dict(max_new_tokens=8, do_sample=False, num_beams=3)
    full_prompt = f"<image>\n{prompt}"

    response = model.chat(
        tokenizer, pixel_values, full_prompt, gen_cfg,
        num_patches_list=[num_patches],
        history=None, return_history=False,
    )

    response = response.strip()
    m = re.search(r"[1-5]", response)
    if m:
        return int(m.group())
    print(f"  [WARN] Unparseable response: '{response}' → 5 (skip)")
    return 5


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print(f"[INFO] test_dir  : {TEST_DIR}")
    print(f"[INFO] patch_dir : {PATCH_DIR}")
    print(f"[INFO] output    : {OUT_CSV.resolve()}")

    # ── Stitch ───────────────────────────────────────────────
    images, grays = load_patches(PATCH_DIR)
    t0      = time.time()
    map_img = stitch_patches(images, grays)
    print(f"[INFO] Stitching done in {time.time()-t0:.1f}s")
    cv2.imwrite(str(MAP_OUT), map_img)
    print(f"[INFO] Map saved → {MAP_OUT}")

    # ── Load questions ────────────────────────────────────────
    test_df = pd.read_csv(TEST_CSV)
    print(f"[INFO] {len(test_df)} questions.")

    # ── Load VLM ─────────────────────────────────────────────
    tokenizer, model = load_model()
    map_pil = Image.fromarray(cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB))

    # ── Answer ───────────────────────────────────────────────
    results = []
    for _, row in test_df.iterrows():
        qid   = row["id"]
        qtext = row["question"]
        opts  = [row["option_1"], row["option_2"],
                 row["option_3"], row["option_4"]]
        print(f"\n  [Q] {qid}: {qtext}")
        ans   = answer_question(tokenizer, model, map_pil, qtext, opts)
        label = opts[ans - 1] if ans <= 4 else "SKIP"
        print(f"      → {ans} ({label})")
        results.append({"id": qid, "question_num": qid, "option": ans})

    # ── Save submission.csv in CWD ────────────────────────────
    out_df = pd.DataFrame(results, columns=["id", "question_num", "option"])
    out_df.to_csv(str(OUT_CSV), index=False)
    print(f"\n[DONE] submission.csv saved → {OUT_CSV.resolve()}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
