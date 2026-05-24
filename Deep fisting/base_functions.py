import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import find_contours
from scipy.spatial import ConvexHull, distance_matrix, QhullError

def conv(x, kernel, pad):
    H = x.shape[0]
    W = x.shape[1]

    Kh = kernel.shape[0]
    Kw = kernel.shape[1]

    H1 = 1+(H+2*pad-Kh)
    W1 = 1+(W+2*pad-Kw)

    res = np.zeros(shape = (H1,W1))

    ph = np.zeros(shape=(H,pad))
    pw = np.zeros(shape = (pad,W+2*pad))

    x=np.hstack((ph,x,ph))
    x=np.vstack((pw,x,pw))

    for i in range(H1):
      for j in range(W1):
        sum = 0
        for k in range(Kh):
          for f in range(Kw):
            sum+=x[i+k][j+f]*kernel[k][f]
        res[i][j]=sum
    return res


def applySobel(H):
    kernelx = np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
    kernely = np.array([[-1, 0, 1],[-2,0,2],[-1,0,1]])
    
    imx = conv(H,kernelx,pad=1)
    imy = conv(H,kernely,pad=1)

    image_sobel = np.sqrt(imx*imx+imy*imy)

    max_value = image_sobel.max()
    
    return image_sobel, max_value   

def fit_circle(points):

    y = points[:, 0]
    x = points[:, 1]

    A = np.column_stack((2*x, 2*y, np.ones_like(x)))
    b = x**2 + y**2

    c, *_ = np.linalg.lstsq(A, b, rcond=None)

    center_x = c[0]
    center_y = c[1]
    radius = np.sqrt(c[2] + center_x**2 + center_y**2)

    return center_y, center_x, radius 

def make_circle(H, dx=1.0, threshold=0.2, use_contour=True):

    mask_raw = H > threshold
    mask_area = binary_fill_holes(mask_raw)

    if not np.any(mask_area):
        return None, None, np.nan, np.nan, np.nan, mask_area

    if use_contour:
        contours = find_contours(mask_area.astype(float), level=0.5)

        if len(contours) == 0:
            return None, None, np.nan, np.nan, np.nan, mask_area

        contour = max(contours, key=len)

        y = contour[:, 0] * dx
        x = contour[:, 1] * dx

    else:
        rows, cols = np.where(mask_area)

        y = rows * dx
        x = cols * dx

    A = np.column_stack((2*x, 2*y, np.ones_like(x)))
    b = x**2 + y**2

    c, *_ = np.linalg.lstsq(A, b, rcond=None)

    cx = c[0]
    cy = c[1]
    r = np.sqrt(c[2] + cx**2 + cy**2)

    theta = np.linspace(0, 2*np.pi, 500)
    circle_x = cx + r * np.cos(theta)
    circle_y = cy + r * np.sin(theta)

    return circle_x, circle_y, r, cx, cy, mask_area

def get_spheroid_mask(H, threshold=0.2):
    """
    Defines the projected spheroid body.
    Internal holes, including necrotic cores, are filled.
    """
    mask_raw = H > threshold
    mask_area = binary_fill_holes(mask_raw)

    return mask_area

def area_parameters(H, dx, threshold=0.2):
    mask_area = get_spheroid_mask(H, threshold)

    area_binary = np.sum(mask_area) * dx**2
    area_effective_total = np.sum(H) * dx**2
    area_effective_inside = np.sum(H[mask_area]) * dx**2

    return area_binary, area_effective_total, area_effective_inside

def gray_statistics(H, bins=256, threshold = 0.2):
    mask_area = get_spheroid_mask(H, threshold)
    values = H[mask_area]

    if values.size == 0:
        return {
            "std_gray": np.nan,
            "modal_gray": np.nan,
            "min_gray": np.nan,
            "max_gray": np.nan,
        }

    std_gray = np.std(values)
    min_gray = np.min(values)
    max_gray = np.max(values)

    counts, edges = np.histogram(values, bins=bins)
    mode_bin = np.argmax(counts)
    modal_gray = 0.5 * (edges[mode_bin] + edges[mode_bin + 1])

    return {
        "std_gray": std_gray,
        "modal_gray": modal_gray,
        "min_gray": min_gray,
        "max_gray": max_gray,
    }

def centroid_physical(H, x, y, threshold = 0.2):
    mask_area = get_spheroid_mask(H, threshold)
    Y, X = np.meshgrid(y, x, indexing="ij")

    if not np.any(mask_area):
        return np.nan, np.nan

    x_c = np.mean(X[mask_area])
    y_c = np.mean(Y[mask_area])

    return x_c, y_c


def center_of_mass_physical(H, x, y, threshold = 0.2):
    mask_area = get_spheroid_mask(H, threshold)
    Y, X = np.meshgrid(y, x, indexing="ij")

    weights = H[mask_area]
    total_weight = np.sum(weights)

    if not np.any(mask_area) or total_weight == 0:
        return np.nan, np.nan

    x_m = np.sum(X[mask_area] * weights) / total_weight
    y_m = np.sum(Y[mask_area] * weights) / total_weight

    return x_m, y_m

def perimeter_contour(H, dx=1.0, threshold=0.2):
    mask_area = get_spheroid_mask(H, threshold)
    return perimeter_from_mask(mask_area, dx)

def bounding_rectangle_physical(H, x, y, threshold=0.2):
    mask_area = get_spheroid_mask(H, threshold)

    if not np.any(mask_area):
        return {
            "x_min": np.nan,
            "y_min": np.nan,
            "x_max": np.nan,
            "y_max": np.nan,
            "width": np.nan,
            "height": np.nan,
        }

    rows, cols = np.where(mask_area)

    row_min = rows.min()
    row_max = rows.max()
    col_min = cols.min()
    col_max = cols.max()

    x_min = x[col_min]
    x_max = x[col_max]
    y_min = y[row_min]
    y_max = y[row_max]

    width = x_max - x_min
    height = y_max - y_min

    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "width": width,
        "height": height,
    }

def fit_ellipse_from_mask(mask_area, dx=1.0):
    """
    Fits an equivalent ellipse to a binary mask using second central moments.

    Returns:
        x0, y0       - ellipse center
        major_axis   - full length of major axis
        minor_axis   - full length of minor axis
        angle        - orientation angle in radians
    """

    rows, cols = np.where(mask_area)

    if rows.size == 0:
        return {
            "x0": np.nan,
            "y0": np.nan,
            "major_axis": np.nan,
            "minor_axis": np.nan,
            "angle": np.nan,
        }

    x = cols * dx
    y = rows * dx

    x0 = np.mean(x)
    y0 = np.mean(y)

    x_c = x - x0
    y_c = y - y0

    mu20 = np.mean(x_c**2)
    mu02 = np.mean(y_c**2)
    mu11 = np.mean(x_c * y_c)

    cov = np.array([
        [mu20, mu11],
        [mu11, mu02]
    ])

    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # For a filled ellipse:
    # variance along semi-major axis = a^2 / 4
    # variance along semi-minor axis = b^2 / 4
    semi_major = 2 * np.sqrt(eigvals[0])
    semi_minor = 2 * np.sqrt(eigvals[1])

    major_axis = 2 * semi_major
    minor_axis = 2 * semi_minor

    major_vector = eigvecs[:, 0]
    angle = np.arctan2(major_vector[1], major_vector[0])

    return {
        "x0": x0,
        "y0": y0,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "angle": angle,
    }

def area_from_mask(mask_area, dx=1.0):
    return np.sum(mask_area) * dx**2


def perimeter_from_mask(mask_area, dx=1.0):
    if not np.any(mask_area):
        return np.nan

    contours = find_contours(mask_area.astype(float), level=0.5)

    if len(contours) == 0:
        return np.nan

    contour = max(contours, key=len)

    diffs = np.diff(contour, axis=0)
    segment_lengths = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)

    perimeter = np.sum(segment_lengths)


    if np.linalg.norm(contour[0] - contour[-1]) > 1e-12:
        perimeter += np.linalg.norm(contour[0] - contour[-1])

    return perimeter * dx

def circularity(H, dx=1.0, threshold=0.2):
    mask_area = get_spheroid_mask(H, threshold)

    area,_ = area_from_mask(mask_area, dx)
    perimeter = perimeter_from_mask(mask_area, dx)

    if perimeter == 0 or np.isnan(perimeter):
        return np.nan

    circ = 4 * np.pi * area / perimeter**2

    return circ

def feret_diameter_from_mask(mask_area, dx=1.0):
    if not np.any(mask_area):
        return np.nan

    contours = find_contours(mask_area.astype(float), level=0.5)

    if len(contours) == 0:
        return np.nan

    contour = max(contours, key=len)
    points = np.column_stack((contour[:, 1], contour[:, 0])) * dx

    if len(points) < 2:
        return np.nan

    if len(points) == 2:
        return np.linalg.norm(points[1] - points[0])

    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    except QhullError:
        D = distance_matrix(points, points)
        return np.max(D)

    D = distance_matrix(hull_points, hull_points)
    return np.max(D)

def integrated_density(H, mask_area, dx=1.0):
    values = H[mask_area]

    if values.size == 0:
        return {
            "RawIntDen": np.nan,
            "IntDen": np.nan,
            "PhysicalIntDen": np.nan,
        }

    raw_int_den = np.sum(values)

    area_pixels = np.sum(mask_area)
    mean_gray = np.mean(values)

    int_den = area_pixels * mean_gray

    physical_int_den = raw_int_den * dx**2

    return {
        "RawIntDen": raw_int_den,
        "IntDen": int_den,
        "PhysicalIntDen": physical_int_den,
    }
