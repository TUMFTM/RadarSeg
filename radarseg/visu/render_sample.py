# Standard Libraries
import math

# 3rd Party Libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from PIL import Image

# Local imports
from radarseg.visu import utils


def get_annotation_corners(annotation, wlh_factor: float = 1.0):
    """
    Returns the bounding box corners.

    Reference:
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py

    Arguments:
        annotation: NuScenes annotation object (box), <Box>.
        wlh_factor: Multiply w, l, h by a factor to scale the box, <float>.

    Returns:
        corners: First four corners are the ones facing forward.
                 The last four are the ones facing backwards (3, 8), <np.float>.
    """
    orientation = Quaternion(annotation.orientation)
    wlh = np.array(annotation.wlh)
    w, l, h = wlh * wlh_factor

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    corners = np.dot(orientation.rotation_matrix, corners)

    # Translate
    x, y, z = annotation.center
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners


def render_ego_centric_map(map_,
                           pose,
                           axes_limit: float = 80,
                           ax=None):
    """
    Renders a map centered around the associated ego pose.

    Reference:
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py

    Arguments:
        map_: NuScenes map object, <dict>.
        pose: Ego pose of th vehicle, <dict>.
        axes_limit: Axes limit measured in meters, <float>.
        ax: Axes onto which to render, <matplotlib.axes.Axes>.
    """
    def crop_image(image: np.array,
                   x_px: int,
                   y_px: int,
                   axes_limit_px: int) -> np.array:
        x_min = int(x_px - axes_limit_px)
        x_max = int(x_px + axes_limit_px)
        y_min = int(y_px - axes_limit_px)
        y_max = int(y_px + axes_limit_px)

        cropped_image = image[y_min:y_max, x_min:x_max]

        return cropped_image

    map_mask = map_['mask']

    # Retrieve and crop mask.
    pixel_coords = map_mask.to_pixel_coords(pose['translation'][0], pose['translation'][1])
    scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
    mask_raster = map_mask.mask()
    cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

    # Rotate image.
    ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
    yaw_deg = -math.degrees(ypr_rad[0])
    rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))

    # Cop image.
    ego_centric_map = crop_image(rotated_cropped, rotated_cropped.shape[1] / 2,
                                 rotated_cropped.shape[0] / 2,
                                 scaled_limit_px)

    # Init axes and show image.
    # Set background to white and foreground (semantic prior) to gray.
    if ax is None:
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(1, 1, 1)

    foreground_color = int(utils.rgb2gray(utils.get_colors('secondary')[3]) * 255)
    ego_centric_map[ego_centric_map == map_mask.foreground] = foreground_color
    ego_centric_map[ego_centric_map == map_mask.background] = 255
    ax.imshow(ego_centric_map, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit], cmap='gray', vmin=0, vmax=255)


def render_annotation(annotation,
                      pose,
                      color=None,
                      view=None,
                      ax=None,
                      linewidth: float = 1.0,
                      alpha: float = 0.7):
    """
    Renders the given annotation.

    Reference:
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py

    Arguments:
        annotation: NuScenes annotation object (bounding box), <Box>.
        color: Color of the bounding box, normalized rgb array(1, 3), <np.ndarray>.
        view: LIDAR view point, <np.ndarray>.
        ax: LIDAR view point, <matplotlib.axes.Axes>.
    """
    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            ax.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth, alpha=alpha)
            prev = corner

    # Initialize
    color = color if color is not None else np.array([0, 0, 0])
    view = view if view is not None else np.eye(4)

    if ax is None:
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(1, 1, 1)

    # Get annotation corners
    corners = get_annotation_corners(annotation)

    # Transform to reference frame
    translation = np.expand_dims(np.array(pose['translation']), axis=1)
    rotation = np.linalg.inv(Quaternion(pose['rotation']).rotation_matrix)
    corners = np.dot(rotation, corners - translation)

    # Draw the sides
    for i in range(4):
        ax.plot([corners.T[i][0], corners.T[i + 4][0]],
                [corners.T[i][1], corners.T[i + 4][1]],
                color=color, linewidth=linewidth, alpha=alpha)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], color)
    draw_rect(corners.T[4:], color)

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    ax.plot([center_bottom[0], center_bottom_forward[0]],
            [center_bottom[1], center_bottom_forward[1]],
            color=color, linewidth=linewidth, alpha=alpha)


def render_sample(points,
                  labels=None,
                  annotations=None,
                  score=None,
                  velocities=None,
                  map_=None,
                  pose=None,
                  colors=None,
                  ax=None,
                  axes_limit: float = 80,
                  point_scale: float = 2.0,
                  linewidth: float = 0.7,
                  out_path: str = None,
                  out_format: str = None):
    """
    Renders the given sample data onto an axis.

    Reference:
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py

    Arguments:
        points: Points to render (3, n), <np.float>.
        labels: Class labels of the points (1, n), <np.int>.
        annotations: Annotations to render (list of nuScenes boxes), <List>.
        score: Class score of the points between 0 and 1 (1, n), <np.float>.
        velocities: Velocity values of the points (2, n), <np.float>.
        map_: When provided, Radar data is plotted onto the map, <dict>.
        pose: Ego pose of the vehicle, <dict>.
        colors: Colormap to highlight the individual classes (n, 3), <np.array>.
        ax: Axes onto which to render, <matplotlib.axes.Axes>.
        axes_limit: Axes limit for lidar and radar (measured in meters), <float>.
        point_scale: Size of the points, <float>.
        linewidth: Linewidth of the bounding boxes, <float>.
        out_path: Optional path to save the rendered figure to disk, <str>.
        out_format: Optional image format of the saved figure, <str>.

    Returns:
        fig: Rendered figure, <plt.figure>.
    """
    if ax is None:
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(1, 1, 1)

    # Defiitions
    cmap = matplotlib.colors.ListedColormap(colors)

    # Show map
    if map_ is not None:
        try:
            render_ego_centric_map(map_=map_, pose=pose, axes_limit=axes_limit, ax=ax)
        except IndexError:
            # This is required since some nuScenes maps are corrupted
            pass

    # Show boxes
    if annotations is not None:
        for annotation in annotations:
            render_annotation(annotation, pose=pose, color=cmap(annotation.label), ax=ax, linewidth=linewidth)

    # Scale points according to score
    if score is not None:
        point_scale = np.multiply(score, point_scale)

    # Show point cloud
    if points.ndim > 1:
        ax.scatter(points[0, :], points[1, :], c=cmap(labels), s=point_scale)
    else:
        ax.scatter(points[0], points[1], c=cmap(labels), s=point_scale)

    # Show velocities
    if velocities is not None:
        if points.ndim > 1:
            ax.quiver(points[0, :], points[1, :], velocities[0, :], velocities[1, :], color=cmap(labels), scale=300.0, headwidth=2.0,
                      headlength=3.0, headaxislength=3.0, minlength=0.5)
        else:
            ax.quiver(points[0], points[1], velocities[0], velocities[1], color=cmap(labels), scale=300.0, headwidth=2.0,
                      headlength=3.0, headaxislength=3.0, minlength=0.5)

    # Show ego vehicle
    ax.plot(0, 0, 'x', color='red')

    # Limit visible range
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    # Format axes
    ax.axis('off')
    ax.set_aspect('equal')

    # Export figure
    if out_path is not None:
        plt.savefig(out_path, transparent=True, format=out_format)

    plt.close(fig)
    return fig
