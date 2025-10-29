import numpy as np
from scipy.spatial.distance import cosine

def cosine_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculates the cosine distance (1 - cosine similarity) between two vectors.
    Handles potential zero vectors by returning the maximum distance (1.0).

    Args:
        point1 (np.ndarray): First 1D vector.
        point2 (np.ndarray): Second 1D vector.

    Returns:
        float: The cosine distance value between 0.0 and 1.0.
    """
    epsilon = 1e-9
    norm1 = np.linalg.norm(point1)
    norm2 = np.linalg.norm(point2)

    if norm1 < epsilon or norm2 < epsilon:
        return 1.0

    dist = cosine(point1.astype(np.float64), point2.astype(np.float64))
    
    return float(np.clip(dist, 0.0, 1.0))


def magnitude_cosine_distance(point1: np.ndarray, point2: np.ndarray, alpha: float = 0.5) -> float:
    """
    Calculates a distance combining magnitude difference and cosine distance.
    d = sqrt( alpha * (||v1|| - ||v2||)^2 + (1-alpha) * (1 - cos(v1, v2)) )

    Args:
        point1 (np.ndarray): First 1D vector.
        point2 (np.ndarray): Second 1D vector.
        alpha (float, optional): Weight for the magnitude difference term. Defaults to 0.5.

    Returns:
        float: The combined distance value.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0.0 and 1.0")

    epsilon = 1e-9
    norm1 = np.linalg.norm(point1)
    norm2 = np.linalg.norm(point2)
    beta = 1.0 - alpha 

    # Term 1: Squared Magnitude Difference
    mag_diff_sq = (norm1 - norm2)**2

    # Term 2: Cosine Distance (1 - similarity)
    cos_dist = cosine_distance(point1, point2)

    combined_dist_sq = alpha * mag_diff_sq + beta * cos_dist
    
    return float(np.sqrt(max(0.0, combined_dist_sq)))