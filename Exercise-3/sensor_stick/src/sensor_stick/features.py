import matplotlib.colors
import numpy as np
from pcl_helper import float_to_rgb
import sensor_msgs.point_cloud2 as pc2


def rgb_to_hsv(rgb_list):
    rgb_normalized = [1. * rgb_list[x] / 255 for x in xrange(3)]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized


def compute_histogram(point_list, color_range=(0, 256)):
    channel_histogram = [None] * 3
    for channel in range(3):
        channel_histogram[channel] = np.histogram(
            [pixel[channel] for pixel in point_list],
            bins=32,
            range=color_range)

    # Concatenate and normalize the histograms
    hist_features = np.concatenate(
        [channel[0] for channel in channel_histogram]).astype(np.float64)
    normalized_features = hist_features / np.sum(hist_features)

    return normalized_features


def compute_color_histograms(cloud, using_hsv=False):

    # Compute histograms for the clusters
    point_colors_list = []

    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    return compute_histogram(point_colors_list)


def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    return compute_histogram((norm_x_vals, norm_y_vals, norm_z_vals), color_range=(-1, 1))
