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
#  STEP 2 — BOUNDARY ENFORCED BFS STITCHER  (working approach)
#
#  patch_0 = top-left anchor (guaranteed by problem statement)
#  Single-pass BFS with template matching on boundary strips.
#  Tries all 4 rotations per candidate patch per side.
#  Boundary rule: rejects placements going left/above patch_0.
# ══════════════════════════════════════════════════════════════
def stitch_patches(images, grays):
    num_patches = len(images)
    patch_size = images[0].shape[0]

    # Fingerprint/Search settings — scaled to patch size (works for any map)
    f_size = max(10, patch_size // 6)   # ~16% of patch: enough context, not too wide
    s_zone = max(20, patch_size // 3)   # ~33% of patch: search zone on candidate side

    # ans[patch_id] = (global_x, global_y, rotation_angle)
    # Patch 0 is Top-Left (0, 0)
    ans = {0: (0, 0, 0)}
    queue = deque([0])
    visited = {0}

    rotations = [(0, None), (90, cv2.ROTATE_90_COUNTERCLOCKWISE),
                 (180, cv2.ROTATE_180), (270, cv2.ROTATE_90_CLOCKWISE)]

    print("🧩 Growing Map with Top-Left Boundary Enforcement...")

    while queue:
        i = queue.popleft()
        curr_x, curr_y, _ = ans[i]

        # Current patch sides
        fingerprints_i = {
            'right':  grays[i][:, -f_size:],
            'left':   grays[i][:, :f_size],
            'bottom': grays[i][-f_size:, :],
            'top':    grays[i][:f_size, :]
        }

        unvisited_pool = [idx for idx in range(num_patches) if idx not in visited]

        for j in unvisited_pool:
            found_j = False
            # Check every rotation
            for angle, cv2_rot in rotations:
                rot_j = cv2.rotate(grays[j], cv2_rot) if cv2_rot else grays[j]

                # Possible ways J can connect to I
                configs = [
                    ('right',  rot_j[:, :s_zone],  1, 0),   # J is Right of I
                    ('bottom', rot_j[:s_zone, :],  0, 1),   # J is Below I
                    ('left',   rot_j[:, -s_zone:], -1, 0),  # J is Left of I
                    ('top',    rot_j[-s_zone:, :], 0, -1)   # J is Above I
                ]

                for side_i, search_area_j, dx_dir, dy_dir in configs:
                    if np.std(fingerprints_i[side_i]) < 5: continue

                    res = cv2.matchTemplate(search_area_j, fingerprints_i[side_i], cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)

                    if max_val > 0.82: # Strict confidence
                        mx, my = max_loc

                        # Calculate exact relative shift
                        if dx_dir == 1:  tx, ty = (patch_size - f_size - mx), -my
                        if dx_dir == -1: tx, ty = (-patch_size + s_zone - mx), -my
                        if dy_dir == 1:  tx, ty = -mx, (patch_size - f_size - my)
                        if dy_dir == -1: tx, ty = -mx, (-patch_size + s_zone - my)

                        # Calculate Global Coordinates
                        gx, gy = curr_x + tx, curr_y + ty

                        # --- THE BOUNDARY RULE ---
                        if gx < -20 or gy < -20:
                            continue

                        ans[j] = (gx, gy, angle)
                        visited.add(j)
                        queue.append(j)
                        found_j = True
                        print(f"✅ Placed {j} at ({int(gx)}, {int(gy)}) | Score: {max_val:.2f}")
                        break
                if found_j: break

    print(f"\nFinal count: {len(visited)} patches anchored out of {num_patches}.")

    # --- Painting (Coordinate-Aware) ---
    min_x = min(v[0] for v in ans.values())
    min_y = min(v[1] for v in ans.values())
    max_x = max(v[0] for v in ans.values()) + patch_size
    max_y = max(v[1] for v in ans.values()) + patch_size

    canvas_w, canvas_h = int(max_x - min_x), int(max_y - min_y)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for p_id, (gx, gy, angle) in ans.items():
        p_img = images[p_id]
        if angle == 90:  p_img = cv2.rotate(p_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if angle == 180: p_img = cv2.rotate(p_img, cv2.ROTATE_180)
        if angle == 270: p_img = cv2.rotate(p_img, cv2.ROTATE_90_CLOCKWISE)

        y, x = int(gy - min_y), int(gx - min_x)
        canvas[y:y+patch_size, x:x+patch_size] = p_img

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
