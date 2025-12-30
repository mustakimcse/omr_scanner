#omr_runner.py

import cv2, numpy as np
from sklearn.cluster import KMeans

# -----------------------------
# Tunables (needed for both sheets)
# -----------------------------
MIN_FILL_GATE  = 0.33   # <33% filled হলে খালি ধরো (false positive কাটে)
STRONG_FILL_OK = 0.58   # খুব গাঢ় হলে সরাসরি accept
TH_FLOOR       = 0.36   # adaptive threshold-এর ন্যূনতম সীমা
TH_CEIL       = 0.62    # adaptive threshold-এর সর্বোচ্চ সীমা
MARGIN_MIN     = 0.06   # top-second ন্যূনতম পার্থক্য

# -----------------------------
# Light preprocessing
# -----------------------------
# def maybe_prepare_gray_for_threshold(img, force=False):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.Laplacian(gray, cv2.CV_64F).var()
#     mean = gray.mean()
#     if not force and (blur >= 70 and 80 <= mean <= 185):
#         return gray
#     # a bit stronger for low light / shadow
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     gray2 = clahe.apply(gray)
#     gray2 = cv2.fastNlMeansDenoising(gray2, None, 5, 7, 21)
#     return gray2

# -----------------------------
# Safe 1D KMeans (no crash)
# -----------------------------
def safe_kmeans_1d(values, k, n_init=10, random_state=0):
    arr = np.array(values, dtype=np.float32).reshape(-1,1)
    n = len(values)
    if n == 0:
        return [], {}, np.array([], dtype=int)
    k = max(1, min(int(k), n))
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(arr)
    cents = [c[0] for c in km.cluster_centers_]
    order = np.argsort(cents)
    label_to_rank = {int(lbl): int(rk) for rk, lbl in enumerate(order)}
    cents_sorted = sorted(cents)
    return cents_sorted, label_to_rank, km.labels_

# -----------------------------
# Roll reader (robust enough)
# -----------------------------
def read_roll_number_auto(bin_img, dbg, rect_xywh, all_cands,
                          cols=6, total_rows=12, header_rows=2):
    x, y, w, h = rect_xywh
    dots = [(cx, cy, r) for (cx, cy, r) in all_cands if x <= cx <= x+w and y <= cy <= y+h]
    if not dots:
        return "------"

    roi = bin_img[y:y+h, x:x+w]
    proj = cv2.reduce(roi, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).flatten().astype(np.float32)
    k = max(5, (h // 60) | 1)
    proj_s = cv2.blur(proj.reshape(-1,1), (1, k)).flatten()

    raw_peaks = []
    for i in range(1, len(proj_s)-1):
        if proj_s[i] > proj_s[i-1] and proj_s[i] > proj_s[i+1]:
            raw_peaks.append((proj_s[i], i))
    raw_peaks.sort(reverse=True)
    min_dist = max(6, h // (total_rows+2))
    selected = []
    for val, idx in raw_peaks:
        if all(abs(idx - s) >= min_dist for s in selected):
            selected.append(idx)
        if len(selected) == total_rows:
            break
    if len(selected) < total_rows:
        selected = [int((i+0.55) * h / float(total_rows)) for i in range(total_rows)]
    row_centers = [y + rc for rc in sorted(selected)]

    # Columns
    groups = [[] for _ in range(cols)]
    if len(dots) >= 2:
        xs = [cx for (cx,_,_) in dots]
        col_cents, _, labels = safe_kmeans_1d(xs, cols)
        if len(col_cents) >= 1:
            tmp = [[] for _ in range(len(col_cents))]
            for pt, lb in zip(dots, labels):
                tmp[lb].append(pt)
            means = [np.mean([p[0] for p in g]) if g else 1e9 for g in tmp]
            order = np.argsort(means)
            groups = [tmp[i] for i in order]
    if not any(groups):
        col_w = w / float(cols)
        for (cx, cy, r) in dots:
            cidx = int((cx - x) // col_w)
            cidx = min(max(cidx, 0), cols-1)
            groups[cidx].append((cx, cy, r))

    # y-offset refine
    valid_indices = list(range(header_rows, total_rows))
    offsets = []
    for g in groups:
        if not g: continue
        cx, cy, r = max(g, key=lambda p: p[2])
        ridx = int(np.argmin([abs(cy - rc) for rc in row_centers]))
        if ridx in valid_indices:
            offsets.append(cy - row_centers[ridx])
    if len(offsets) >= 2:
        off = float(np.median(offsets))
        row_centers = [rc + off for rc in row_centers]

    # choose best row per column
    digits = []
    for g in groups:
        if not g: digits.append("-"); continue
        scores = np.zeros(total_rows, dtype=float)
        for (cx, cy, r) in g:
            diffs = [abs(cy - rc) for rc in row_centers]
            ridx = int(np.argmin(diffs))
            scores[ridx] += (r*r) / (diffs[ridx] + 1e-3)
        ridx = max(valid_indices, key=lambda i: scores[i])
        digit = ridx - header_rows
        digit = int(np.clip(digit, 0, 9))
        digits.append(str(digit))
        cx_mean = int(np.mean([p[0] for p in g]))
        cv2.circle(dbg, (cx_mean, int(row_centers[ridx])), 10, (0,255,0), 2)
    return "".join(digits)

# -----------------------------
# Bubble scoring (return score + fill_ratio)
# -----------------------------
def robust_fill_score(bin_img, gray_img, cx, cy, r):
    # inner একটু বড় রাখি — fill বেশি কভার করে
    r_in  = max(3, int(r * 0.68))
    r_out = max(r_in+2, int(r * 1.00))

    H, W = bin_img.shape[:2]
    yy, xx = np.ogrid[-cy:H-cy, -cx:W-cx]
    mask_in  = (xx*xx + yy*yy) <= (r_in*r_in)
    mask_out = (xx*xx + yy*yy) <= (r_out*r_out)

    bin_in = bin_img[mask_in]
    g_in   = gray_img[mask_in]
    g_ring = gray_img[mask_out & (~mask_in)]

    fill_ratio = np.count_nonzero(bin_in) / (bin_in.size + 1e-6)
    ring_m = float(np.mean(g_ring)) + 1e-6
    gray_contrast = max(0.0, min(1.0, (ring_m - float(np.mean(g_in))) / ring_m))

    score = 0.65 * fill_ratio + 0.35 * gray_contrast
    return float(np.clip(score, 0, 1)), float(fill_ratio)

# -----------------------------
# Columns via KMeans (A,B,C,D)
# -----------------------------
def column_bins_by_kmeans(rect_xywh, bubs, n_cols=4):
    x, y, w, h = rect_xywh
    if not bubs:
        edges = np.linspace(x, x + w, n_cols + 1)
        centers = [(edges[i] + edges[i+1]) / 2.0 for i in range(n_cols)]
        return np.array(edges, dtype=float), centers
    xs = [cx for (cx,_,_) in bubs]
    centers = sorted(safe_kmeans_1d(xs, n_cols)[0])
    edges = [x]
    for i in range(len(centers)-1):
        edges.append(0.5*(centers[i] + centers[i+1]))
    edges.append(x + w)
    edges = [max(x, min(x+w, e)) for e in edges]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = min(x+w, edges[i-1] + 1)
    return np.array(edges, dtype=float), centers

# -----------------------------
# Build grid with fixed rows + KM columns
# -----------------------------
def build_grid_with_fixed_rows(rect_xywh, bubs, row_centers, n_cols=4):
    x, y, w, h = rect_xywh
    col_edges, col_centers = column_bins_by_kmeans(rect_xywh, bubs, n_cols=n_cols)
    R = len(row_centers)
    grid = [[None]*n_cols for _ in range(R)]
    for (cx, cy, r) in bubs:
        c_idx = int(np.clip(np.searchsorted(col_edges, cx) - 1, 0, n_cols-1))
        diffs = [abs(cy - rc) for rc in row_centers]
        r_idx = int(np.argmin(diffs))
        # top-row tolerance (bubble slightly above)
        if r_idx == 0 and cy < row_centers[0]:
            if abs(cy - row_centers[0]) < (h / (R * 2)):
                r_idx = 0
        prev = grid[r_idx][c_idx]
        d = abs(cy - row_centers[r_idx]) + abs(cx - col_centers[c_idx])
        if prev is None or d < prev[4]:
            grid[r_idx][c_idx] = (cx, cy, r, 0.0, d)
    return row_centers, col_centers, grid

# -----------------------------
# MSQ rects under roll
# -----------------------------
def find_msq_rects_from_blocks(blocks, img_shape, roll_rect=None):
    H, W = img_shape[:2]
    rx, ry, rw, rh = roll_rect if roll_rect else (0,0,0,0)
    r_bottom = ry + rh
    rects = []
    for b in blocks:
        x,y,w,h = b["rect"]
        area = w*h
        aspect = h / float(max(1,w))
        if r_bottom and y < r_bottom - int(0.02*H):
            continue
        if w < int(0.08*W) or w > int(0.45*W):
            continue
        if h < int(0.10*H):
            continue
        rects.append((x,y,w,h,area,aspect))
    if not rects: return []
    tall = [r for r in rects if r[5] >= 1.25 and r[3] >= 0.18*H]
    if len(tall) < 2:
        tall = sorted(rects, key=lambda t: t[3], reverse=True)[:4]
    tall_top = sorted(sorted(tall, key=lambda t: t[3], reverse=True)[:3], key=lambda t: t[0])
    main_two = tall_top[:2] if len(tall_top) >= 2 else sorted(tall, key=lambda t:(t[0],-t[3]))[:2]
    picked = [m[:4] for m in sorted(main_two, key=lambda r: r[0])]
    if len(picked) >= 2:
        left, mid = picked[0], picked[1]
        mid_x = mid[0]
        avg_w = np.mean([left[2], mid[2]])
        right_cands = []
        for r in rects:
            x,y,w,h,area,aspect = r
            if x <= mid_x: continue
            if aspect < 0.90: continue
            if not (0.55*avg_w <= w <= 1.5*avg_w): continue
            width_sim = 1.0 - abs(w-avg_w)/(avg_w+1e-6)
            score = 2.0*width_sim + 0.001*h + (x/W)
            right_cands.append((score, r))
        if right_cands:
            picked.append(max(right_cands, key=lambda z: z[0])[1][:4])
    return sorted(picked, key=lambda r: r[0])[:3]

def filter_msq_bubbles(all_cands, msq_rects):
    msq_cands = []
    for (cx,cy,r) in all_cands:
        for (x,y,w,h) in msq_rects:
            if x <= cx <= x+w and y <= cy <= y+h:
                msq_cands.append((cx,cy,r)); break
    return msq_cands

LETTER_MAP = ['A','B','C','D']

# -----------------------------
# Row centers from bubble vertical span
# -----------------------------
def row_centers_from_bubble_span(rect_xywh, bubs, total_rows):
    x,y,w,h = rect_xywh
    if not bubs:
        return [y + int((i+0.55) * h / float(total_rows)) for i in range(total_rows)]
    ys = np.array([cy for (_,cy,_) in bubs], dtype=float)
    lo = np.percentile(ys, 3)
    hi = np.percentile(ys, 97)
    pad = 0.02*(hi-lo) if hi>lo else 4.0
    lo -= pad; hi += pad
    centers = np.linspace(lo, hi, total_rows)
    return [float(c) for c in centers]

# -----------------------------
# Infer MSQ answers for one rect (with min-fill gate + adaptive threshold)
# -----------------------------
def infer_msq_answers_for_rect(rect_xywh, all_msq_cands, bin_img, dbg_img, start_q, force_rows=None):
    x,y,w,h = rect_xywh
    bubs = [(cx,cy,r) for (cx,cy,r) in all_msq_cands if x<=cx<=x+w and y<=cy<=y+h]
    total_rows = int(force_rows) if force_rows else 20

    row_centers = row_centers_from_bubble_span(rect_xywh, bubs, total_rows)

    # small upward bias for top rows
    r_guess = np.median([br for (*_, br) in bubs]) if bubs else 6
    bias_curve = np.linspace(-r_guess*0.5, 0, len(row_centers))
    row_centers = [rc + bias_curve[i] for i, rc in enumerate(row_centers)]

    # grid
    row_cs, col_cs, grid = build_grid_with_fixed_rows(rect_xywh, bubs, row_centers, n_cols=4)

    answers, q = [], start_q
    gray_local = cv2.cvtColor(dbg_img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold (soft) + floor/ceil
    all_scores = []
    for (cx,cy,r) in bubs:
        s, _ = robust_fill_score(bin_img, gray_local, cx, cy, r)
        all_scores.append(s)
    if len(all_scores) == 0:
        base_th = TH_FLOOR
    else:
        base_th = np.percentile(all_scores, 55) + 0.04
        base_th = float(np.clip(base_th, TH_FLOOR, TH_CEIL))
    # print("th=", round(base_th,3))  # debug if needed

    for r_idx in range(len(row_cs)):
        scores = [0.0, 0.0, 0.0, 0.0]
        fills  = [0.0, 0.0, 0.0, 0.0]
        row_has_any = False

        for c_idx in range(4):
            cell = grid[r_idx][c_idx]
            if cell is None: continue
            row_has_any = True
            cx, cy, r, _, d = cell
            s, f = robust_fill_score(bin_img, gray_local, cx, cy, r)
            grid[r_idx][c_idx] = (cx, cy, r, s, d)
            scores[c_idx] = s
            fills[c_idx]  = f

        if not row_has_any:
            answers.append((q, '-', scores)); q += 1; continue

        order = sorted(range(4), key=lambda i: scores[i], reverse=True)
        b1, b2 = order[0], order[1]
        top, second = scores[b1], scores[b2]
        margin = top - second

        # ---- accept rule with MIN_FILL gate ----
        accept = (
            (top >= base_th and margin >= MARGIN_MIN and fills[b1] >= MIN_FILL_GATE) or
            (top >= STRONG_FILL_OK)  # very strong fill always ok
        )

        if accept:
            best_letter = LETTER_MAP[b1]
            answers.append((q, best_letter, scores))
            cv2.putText(dbg_img, f"Q{q}:{best_letter}",
                        (int(col_cs[b1]) - 20, int(row_cs[r_idx]) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            answers.append((q, '-', scores))
        q += 1

    return answers, q

# -----------------------------
# Tick counter (for blocks/roll)
# -----------------------------
def count_left_ticks(bin_img, rect_xywh, band_w_ratio=0.035, dbg=None):
    Hh, Ww = bin_img.shape[:2]
    x, y, w, h = rect_xywh
    band_w = max(8, int(band_w_ratio * Ww))
    x1 = max(0, x - band_w); x2 = x
    y1 = max(0, y);          y2 = min(Hh, y + h)
    strip = bin_img[y1:y2, x1:x2]
    if strip.size == 0: return 0
    num, labels, stats, _ = cv2.connectedComponentsWithStats(strip, connectivity=8)
    area_min = max(15, int(0.00008 * Hh * Ww))
    area_max = int(0.004 * Hh * Ww)
    count = 0
    for i in range(1, num):
        sx, sy, sw, sh, area = stats[i]
        if not (area_min <= area <= area_max): continue
        ar = sw / float(sh) if sh else 0
        if not (0.5 <= ar <= 1.8): continue
        mask = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        hull = cv2.convexHull(cnts[0])
        ha = cv2.contourArea(hull) or 1.0
        if area / ha >= 0.85: count += 1
    if dbg is not None:
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (128,128,255), 1)
        cv2.putText(dbg, f"ticks={count}", (x1, max(12, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,255), 1)
    return count









# -----------------------------
# MAIN
# -----------------------------
def run_omr(image_path, debug=False):
    IMG_PATH = image_path

    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")

    # resize normalize (scale-invariant)
    h0, w0 = img.shape[:2]
    target_w = 1200
    scale = target_w / float(w0)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # binarize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
    bin_ = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, k, iterations=2)
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, k, iterations=2)

    # bubble candidates
    num, labels, stats, cents = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    minA = int(120 * (scale**2))
    maxA = int(1800 * (scale**2))

    cands = []
    dbg = img.copy()

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < minA or area > maxA:
            continue

        ar = w / float(h)
        if ar < 0.80 or ar > 1.20:
            continue

        mask = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        cnt = cnts[0]
        per = cv2.arcLength(cnt, True)
        if per == 0:
            continue

        circularity = 4 * np.pi * area / (per * per)
        if circularity < 0.78:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) or 1.0
        solidity = area / hull_area
        if solidity < 0.88:
            continue

        cx, cy = map(int, cents[i])
        r = int((w + h) / 4)
        if r < int(7 * scale):
            continue

        cands.append((cx, cy, r))

    # large blocks
    edges = cv2.Canny(gray, 60, 180)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

    blocks = []
    H, W = img.shape[:2]
    IMG_AREA = H * W
    MIN_AREA = IMG_AREA * 0.008

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for con in contours:
        peri = cv2.arcLength(con, True)
        if peri <= 0:
            continue

        approx = cv2.approxPolyDP(con, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if w * h < MIN_AREA or w < 60 or h < 60:
            continue

        area_poly = cv2.contourArea(approx)
        if area_poly < MIN_AREA:
            continue

        tick_cnt = count_left_ticks(bin_, (x, y, w, h), dbg=dbg)
        blocks.append({
            "rect": (x, y, w, h),
            "area": area_poly,
            "aspect": w / float(h),
            "ticks": tick_cnt
        })

    # roll detect (EXACT SAME LOGIC AS __main__)
    roll_candidate, best_diff = None, 1e9
    for b in blocks:
        t = b["ticks"]
        diff10 = abs(t - 10)
        diff20 = abs(t - 20)

        if diff10 < diff20 and diff10 <= best_diff:
            best_diff = diff10
            roll_candidate = b

    roll = None
    if roll_candidate is not None:
        roll = read_roll_number_auto(
            bin_,
            dbg,
            roll_candidate["rect"],
            cands,
            cols=6,
            total_rows=12,
            header_rows=2
        )

    # MSQ rects + bubbles
    msq_rects = find_msq_rects_from_blocks(
        blocks,
        img.shape,
        roll_candidate["rect"] if roll_candidate else None
    )

    msq_cands = filter_msq_bubbles(cands, msq_rects)

    answers = {}
    scores = {}
    q_start = 1

    for i, rect in enumerate(msq_rects):
        force = 20 if i in (0, 1) else 5
        ans, q_start = infer_msq_answers_for_rect(
            rect,
            msq_cands,
            bin_,
            dbg,
            q_start,
            force_rows=force
        )
        for q, letter, sc in ans:
            answers[q] = letter
            scores[q] = sc

    return {
        "roll": roll,
        "answers": answers,
        "scores": scores,
        "debug_image": dbg if debug else None
    }


