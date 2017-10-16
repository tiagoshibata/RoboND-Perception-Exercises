#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

from sensor_msgs.msg import PointCloud2

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


pcl_objects_pub = None
pcl_table_pub = None


def voxel_grid_filter(cloud):
    vox = cloud.make_voxel_grid_filter()
    vox.set_leaf_size(.01, .01, .01)
    return vox.filter()


def pass_through_filter(cloud):
    passthrough = cloud.make_passthrough_filter()
    passthrough.set_filter_field_name('z')
    passthrough.set_filter_limits(.6, 1.1)
    return passthrough.filter()


def ransac(cloud):
    segmenter = cloud.make_segmenter()
    segmenter.set_model_type(pcl.SACMODEL_PLANE)
    segmenter.set_method_type(pcl.SAC_RANSAC)
    segmenter.set_distance_threshold(.02)

    inliers, coefficients = segmenter.segment()
    return inliers, coefficients


def euclidean_clustering(point_cloud):
    tree = point_cloud.make_kdtree()
    ec = point_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(50)
    ec.set_MaxClusterSize(2500)
    ec.set_SearchMethod(tree)
    return ec.Extract()


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    cloud = ros_to_pcl(pcl_msg)
    cloud = voxel_grid_filter(cloud)
    cloud = pass_through_filter(cloud)

    table_inliers, coefficients = ransac(cloud)
    # table_cloud can be extracted with cloud.extract(table_inliers)
    objects_cloud = cloud.extract(table_inliers, negative=True)
    point_cloud = XYZRGB_to_XYZ(objects_cloud)
    clusters = euclidean_clustering(point_cloud)

    cluster_color = get_color_list(len(clusters))
    color_cluster_point_list = []
    for j, indices in enumerate(clusters):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([
                point_cloud[indice][0],
                point_cloud[indice][1],
                point_cloud[indice][2],
                rgb_to_float(cluster_color[j]),
            ])

    # Create a new cloud with an unique color per cluster
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # table_message, objects_message = pcl_to_ros(table_inliers), pcl_to_ros(cluster_cloud)
    objects_message = pcl_to_ros(cluster_cloud)
    pcl_objects_pub.publish(objects_message)

    detected_objects_labels = []
    detected_objects = []
    for index, pts_list in enumerate(clusters):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = cloud_objects.extract(pts_list)
        # TODO: convert the cluster from pcl to ROS

        # Extract histogram features
        # TODO: complete this step based on capture_features.py

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    pcl_objects_pub.publish(detected_objects)

# Exercise-3 TODOs: 

    # Classify the clusters! (loop through each detected cluster one at a time)

        # Grab the points for the cluster

        # Compute the associated feature vector

        # Make the prediction

        # Publish a label into RViz

        # Add the detected object to the list of detected objects.

    # Publish the list of detected objects

if __name__ == '__main__':
    rospy.init_node('recognition', anonymous=True)
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", PointCloud2, pcl_callback, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)


    # TODO: Load Model From disk

    # Initialize color_list
    get_color_list.color_list = []

    print('Spinning...')
    rospy.spin()
    print('Done!')