"""Small ROI quality check shared by isolated YOLO and the capture adapter."""


def image_quality(frame, normalized_box=None):
    import cv2
    import math
    try:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return dict(ok=False, sharpness=0.)
        h, w = frame.shape[:2]
        if h < 64 or w < 64:
            return dict(ok=False, sharpness=0.)
        if normalized_box is None:
            roi = frame
        else:
            x1, y1, x2, y2 = normalized_box
            if not all(math.isfinite(v) for v in normalized_box):
                return dict(ok=False, sharpness=0.)
            roi = frame[max(0, int(y1 * h)):min(h, int(y2 * h)),
                        max(0, int(x1 * w)):min(w, int(x2 * w))]
        if min(roi.shape[:2]) < 8:
            return dict(ok=False, sharpness=0.)
        # Fixed upper size bounds cost and makes the threshold less resolution-dependent.
        scale = min(1., 256 / max(roi.shape[:2]))
        if scale < 1:
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean, contrast = float(gray.mean()), float(gray.std())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        ok = all(math.isfinite(v) for v in (mean, contrast, sharpness)) and 10 < mean < 245 and contrast >= 8
        return dict(ok=ok, sharpness=sharpness if ok else 0., mean=mean, contrast=contrast)
    except (ValueError, TypeError, cv2.error):
        return dict(ok=False, sharpness=0.)
