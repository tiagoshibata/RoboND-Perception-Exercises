#!/usr/bin/env python
import pcl
from pcl_helper import get_color_list, pcl_to_ros, rgb_to_float, ros_to_pcl, XYZRGB_to_XYZ
import rospy
from sensor_msgs.msg import PointCloud2

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
    segmenter.set_distance_threshold(.01)

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

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(objects_message)
    print('Pub')
    # pcl_table_pub.publish(table_message)


if __name__ == '__main__':
    rospy.init_node('clustering', anonymous=True)
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", PointCloud2, pcl_callback, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)

    print('Spinning...')
    rospy.spin()
    print('Done!')
