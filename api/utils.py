# # yourapp/utils.py
# import cv2
# import numpy as np
# from PIL import Image, ImageEnhance
# from io import BytesIO
# from django.core.files.uploadedfile import InMemoryUploadedFile
# import sys, os


# import cv2

# # Validate input_img against ref_img using ORB feature matching, histogram comparison, and aspect ratio check.
# # def validate_against_reference_opencv(input_img, ref_img):
# #     Hr, Wr = ref_img.shape[:2]
# #     inp = cv2.resize(input_img, (Wr, Hr))

# #     g_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
# #     g_inp = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)

# #     orb = cv2.ORB_create(3000)
# #     kp1, des1 = orb.detectAndCompute(g_ref, None)
# #     kp2, des2 = orb.detectAndCompute(g_inp, None)

# #     if des1 is None or des2 is None:
# #         return False, "Feature extraction failed"

# #     bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# #     matches = bf.knnMatch(des1, des2, k=2)

# #     good = []
# #     for m, n in matches:
# #         if m.distance < 0.75 * n.distance:
# #             good.append(m)

# #     match_ratio = len(good) / max(len(kp1), 1)

# #     hist1 = cv2.calcHist([g_ref], [0], None, [64], [0,256])
# #     hist2 = cv2.calcHist([g_inp], [0], None, [64], [0,256])
# #     hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# #     asp_ref = Hr / Wr
# #     asp_inp = input_img.shape[0] / input_img.shape[1]
# #     aspect_diff = abs(asp_ref - asp_inp)

# #     if match_ratio < 0.07:
# #         return False, "ORB mismatch"

# #     if hist_score < 0.75:
# #         return False, "Histogram mismatch"

# #     if aspect_diff > 0.08:
# #         return False, "Aspect ratio mismatch"

# #     return True, "Template matched"




# def _read_image_from_uploadedfile(uploaded_file):
#     """Return OpenCV BGR image from Django UploadedFile."""
#     data = uploaded_file.read()
#     arr = np.frombuffer(data, np.uint8)
#     img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
#     return img

# def _four_point_transform(image, pts):
#     # pts in order tl, tr, br, bl
#     (tl, tr, br, bl) = pts
#     # compute widths and heights
#     widthA = np.linalg.norm(br - bl)
#     widthB = np.linalg.norm(tr - tl)
#     maxWidth = max(int(widthA), int(widthB))
#     heightA = np.linalg.norm(tr - br)
#     heightB = np.linalg.norm(tl - bl)
#     maxHeight = max(int(heightA), int(heightB))

#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth - 1, maxHeight - 1],
#         [0, maxHeight - 1]], dtype="float32")

#     M = cv2.getPerspectiveTransform(pts, dst)
#     warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
#     return warped

# def detect_document_and_crop(img_bgr, debug_save=False, debug_prefix="dbg"):
#     """
#     Improved document detection:
#       - use grayscale + bilateral/blur + adaptive threshold + morphology
#       - find largest contour by area and ensure it's roughly rectangular
#       - approximate polygon with smaller eps (0.01) for tighter fit
#       - validate resulting warped dimensions (min width/height threshold)
#       - if detection fails or warp too small → return None (caller should fallback)
#     Parameters:
#       - img_bgr: OpenCV BGR image (numpy array)
#       - debug_save: if True, saves intermediate debug images to MEDIA_ROOT/debug_<prefix>_*.jpg
#       - debug_prefix: prefix for debug filenames
#     """
#     orig = img_bgr.copy()
#     H, W = orig.shape[:2]

#     # for speed, work on a resized copy but remember ratio
#     target_max = 1000.0  # work size (bigger helps accuracy; reduce if very slow)
#     ratio = 1.0
#     if max(H, W) > target_max:
#         ratio = target_max / max(H, W)
#         small = cv2.resize(orig, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
#     else:
#         small = orig.copy()

#     gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

#     # noise reduction but keep edges
#     gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
#     # enhance edges
#     # use adaptive threshold sometimes better for documents
#     th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                cv2.THRESH_BINARY, 11, 2)

#     # combine Canny and threshold to get stronger edges
#     edges = cv2.Canny(gray, 50, 150)
#     # morphological closing to join edge gaps
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
#     closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
#     # also combine with threshold (bitwise)
#     combined = cv2.bitwise_or(closed, th)


#     contours, _ = cv2.findContours(combined.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None

#     # sort by area and consider top few
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

#     doc_cnt = None
#     for c in contours:
#         area = cv2.contourArea(c)
#         # ignore tiny contours
#         if area < 0.02 * (small.shape[0] * small.shape[1]):
#             continue
#         peri = cv2.arcLength(c, True)
#         # try a tighter approximation epsilon
#         approx = cv2.approxPolyDP(c, 0.01 * peri, True)
#         if len(approx) == 4:
#             doc_cnt = approx
#             break

#     if doc_cnt is None:
#         # try relaxing approx (in case 4-point not found)
#         for c in contours:
#             peri = cv2.arcLength(c, True)
#             approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#             if len(approx) == 4:
#                 doc_cnt = approx
#                 break

#     if doc_cnt is None:
#         return None

#     # scale contour back to original coordinates
#     doc_pts = doc_cnt.reshape(4, 2) / ratio
#     rect = order_points(np.array(doc_pts, dtype="float32"))

#     # compute target size but enforce min size and reasonable max
#     (tl, tr, br, bl) = rect
#     widthA = np.linalg.norm(br - bl)
#     widthB = np.linalg.norm(tr - tl)
#     maxWidth = max(int(widthA), int(widthB))

#     heightA = np.linalg.norm(tr - br)
#     heightB = np.linalg.norm(tl - bl)
#     maxHeight = max(int(heightA), int(heightB))

#     # Safety checks — if dims tiny or NaN, abort
#     if not np.isfinite(maxWidth) or not np.isfinite(maxHeight):
#         return None
#     if maxWidth < 200 or maxHeight < 200:
#         # too small to be a sensible document — abort
#         return None

#     # optional: expand margin a bit to avoid cropped edges (in original scale)
#     margin_px = int(min(maxWidth, maxHeight) * 0.02)  # 2% margin
#     # shift rect points outward by tiny margin along normal directions
#     # (simple approach: expand bounding rect)
#     dst_width = maxWidth + margin_px * 2
#     dst_height = maxHeight + margin_px * 2

#     dst = np.array([
#         [0, 0],
#         [dst_width - 1, 0],
#         [dst_width - 1, dst_height - 1],
#         [0, dst_height - 1]
#     ], dtype="float32")

#     try:
#         M = cv2.getPerspectiveTransform(rect, dst)
#         warped = cv2.warpPerspective(orig, M, (dst_width, dst_height))
#     except Exception:
#         return None

#     # final safety: if warped is very narrow strip or mostly one color -> discard
#     wh = warped.shape[:2]
#     if wh[0] < 200 or wh[1] < 200:
#         return None

#     # check variance — if image almost solid color then probably wrong
#     if np.std(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)) < 10:
#         # low contrast -> likely wrong; abort
#         return None

#     return warped




# def order_points(pts):
#     # initial ordering: tl, tr, br, bl
#     rect = np.zeros((4, 2), dtype = "float32")
#     s = pts.sum(axis = 1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]

#     diff = np.diff(pts, axis = 1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]
#     return rect

# def enhance_image_for_upload(bgr_img, upscale=True, upscale_factor=2):
#     """
#     Enhance image: convert to RGB PIL, adjust sharpness/contrast, optionally upscale.
#     Return JPEG bytes.
#     """
#     # Convert BGR to RGB
#     rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
#     img_pil = Image.fromarray(rgb)

#     # Upscale using Pillow (better to use cv2.INTER_CUBIC if using OpenCV)
#     if upscale:
#         new_size = (int(img_pil.width * upscale_factor), int(img_pil.height * upscale_factor))
#         img_pil = img_pil.resize(new_size, Image.BICUBIC)

#     # Enhance: sharpness, contrast, brightness as needed
#     enhancer = ImageEnhance.Sharpness(img_pil)
#     img_pil = enhancer.enhance(1.3)   # increase sharpness

#     enhancer = ImageEnhance.Contrast(img_pil)
#     img_pil = enhancer.enhance(1.1)   # slight contrast

#     enhancer = ImageEnhance.Brightness(img_pil)
#     img_pil = enhancer.enhance(1.02)  # slight brighten

#     # Save to bytes
#     out_io = BytesIO()
#     img_pil.save(out_io, format='JPEG', quality=92, optimize=True)
#     out_io.seek(0)
#     return out_io

# def process_document_image(uploaded_file, request=None, try_enhance_if_small=True):
#     """
#     Main helper:
#      - read uploaded_file
#      - detect document and crop/warp; if fails, fallback to original
#      - if image dimension small or low-quality -> enhance/upscale
#      - return InMemoryUploadedFile ready for Django to save
#     """
#     try:
#         img = _read_image_from_uploadedfile(uploaded_file)
#         if img is None:
#             raise ValueError("cannot decode image")

#         cropped = detect_document_and_crop(img)
#         if cropped is None:
#             # fallback: use original image
#             target = img
#         else:
#             target = cropped

#         h, w = target.shape[:2]
#         need_enhance = False
#         # heuristics: if either dimension small -> upscale
#         if try_enhance_if_small and (h < 800 or w < 600):
#             need_enhance = True

#         # enhance and get bytes
#         out_io = enhance_image_for_upload(target, upscale=need_enhance, upscale_factor=2 if need_enhance else 1)
#         # assemble InMemoryUploadedFile
#         filename = uploaded_file.name
#         if not filename.lower().endswith('.jpg') and not filename.lower().endswith('.jpeg'):
#             # normalize to .jpg
#             base, _ = os.path.splitext(filename)
#             filename = base + '.jpg'

#         size = out_io.getbuffer().nbytes
#         django_file = InMemoryUploadedFile(
#             file=out_io,
#             field_name='image',
#             name=filename,
#             content_type='image/jpeg',
#             size=size,
#             charset=None
#         )
#         return django_file
#     except Exception as e:
#         # on any failure, return original uploaded_file (rewound)
#         try:
#             uploaded_file.seek(0)
#         except Exception:
#             pass
#         return uploaded_file
