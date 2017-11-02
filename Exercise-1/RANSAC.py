import pcl

cloud = pcl.load_XYZRGB('tabletop.pcd')


# Voxel Grid filter
vox = cloud.make_voxel_grid_filter()
vox.set_leaf_size(.01, .01, .01)
cloud_filtered = vox.filter()
pcl.save(cloud_filtered, 'voxel_downsampled.pcd')


# Pass Through filter
passthrough = cloud_filtered.make_passthrough_filter()
passthrough.set_filter_field_name('z')
passthrough.set_filter_limits(.6, 1.1)
cloud_filtered = passthrough.filter()
pcl.save(cloud_filtered, 'pass_through_filtered.pcd')


# RANSAC plane segmentation
segmenter = cloud_filtered.make_segmenter()
segmenter.set_model_type(pcl.SACMODEL_PLANE)
segmenter.set_method_type(pcl.SAC_RANSAC)
segmenter.set_distance_threshold(.01)

# Extract inliers
inliers, coefficients = segmenter.segment()
inlier_points = cloud_filtered.extract(inliers)
pcl.save(inlier_points, 'extracted_inliers.pcd')


# Extract outliers
inliers, coefficients = segmenter.segment()
outlier_points = cloud_filtered.extract(inliers, negative=True)
pcl.save(outlier_points, 'extracted_outliers.pcd')

# Save pcd for tabletop objects
