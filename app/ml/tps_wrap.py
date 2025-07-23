import numpy as np
import cv2

class TPSWarp:
    def warp_image(self, image, src_points, dst_points, output_size):
        """
        Warps an image using Thin Plate Spline (TPS) given control points.

        Args:
            image (np.ndarray): Source image.
            src_points (np.ndarray): Source control points.
            dst_points (np.ndarray): Target control points.
            output_size (tuple): (width, height) of output

        Returns:
            np.ndarray: Warped image using TPS
        """
        assert src_points.shape == dst_points.shape, "Mismatch in control points"

        # Create TPS transformer
        tps = cv2.createThinPlateSplineShapeTransformer()
        matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]

        tps.estimateTransformation(dst_points.reshape(-1, 1, 2), src_points.reshape(-1, 1, 2), matches)
        warped = tps.warpImage(image)
        return warped
