import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time

# Define the file path for the image directory
filepath = 'dataset/sequences/00/image_0/'

# List all files in the directory and display the first five
left_images = os.listdir(filepath)
print(f"Files in directory: {left_images[:5]}")

# Check if the directory contains any files
if not left_images:
    print("The directory is empty or the path is incorrect.")
else:
    # Attempt to read the first image in grayscale mode
    img_path = os.path.join(filepath, left_images[0])
    img = cv2.imread(img_path, 0)

    # Check if the image was loaded successfully
    if img is not None:
        print(f"Image loaded successfully: {img_path}")
        plt.imshow(img, cmap='gray')
        plt.title(f"Displaying: {left_images[0]}")
        plt.axis('off')  # Hide the axis for a cleaner look
        plt.show()
    else:
        print(f"Failed to load the image: {img_path}")

# Load the first image and display its shape
first_image = cv2.imread(filepath + left_images[0], 0)
print(f"shape of first_image: {first_image.shape}")

# Load the point cloud data and reshape it
filepath1 = 'dataset/sequences/00/'
velodyne_files = os.listdir(filepath1 + 'velodyne')
pointcloud = np.fromfile(filepath1 + 'velodyne/' + velodyne_files[0], dtype=np.float32)
pointcloud = pointcloud.reshape((-1, 4))
print(f"shape of first_image: {pointcloud.shape}")

# Visualize the point cloud using Matplotlib
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

xs = pointcloud[:, 0]
ys = pointcloud[:, 1]
zs = pointcloud[:, 2]

ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
ax.scatter(xs, ys, zs, s=0.01)
ax.grid(False)
ax.axis('off')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.view_init(elev=40, azim=180)
plt.show()

# Load calibration data and reshape the transformation matrix
calib = pd.read_csv('dataset/sequences/00/calib.txt', delimiter=' ', header=None, index_col=0)
Tr = np.array(calib.iloc[4]).reshape((3, 4))
print("Tr matrix:", Tr.round(4))

import progressbar

# Define a class to handle the dataset, including images and point clouds
class Dataset_handler():

    def __init__(self, sequence, lidar=True, progress_bar=True, low_memory=True):
        self.lidar = lidar
        self.low_memory = low_memory

        self.seq_dir = 'dataset/sequences/{}/'.format(sequence)
        self.poses_dir = 'dataset/poses/{}.txt'.format(sequence)

        self.left_image_files = os.listdir(self.seq_dir + 'image_0')
        self.right_image_files = os.listdir(self.seq_dir + 'image_1')
        self.velodyne_files = os.listdir(self.seq_dir + 'velodyne')
        self.num_frames = len(self.left_image_files)
        self.lidar_path = self.seq_dir + 'velodyne/'

        # Load ground truth poses
        poses = pd.read_csv(self.poses_dir, delimiter=' ', header=None)
        self.gt = np.zeros((self.num_frames, 3, 4))

        for i in range(len(poses)):
            self.gt[i] = np.array(poses.iloc[i]).reshape(3, 4)

        # Load calibration parameters
        calib = pd.read_csv(self.seq_dir + 'calib.txt', delimiter=' ', header=None, index_col=0)
        self.P0 = np.array(calib.loc['P0:']).reshape((3, 4))
        self.P1 = np.array(calib.loc['P1:']).reshape((3, 4))
        self.P2 = np.array(calib.loc['P2:']).reshape((3, 4))
        self.P3 = np.array(calib.loc['P3:']).reshape((3, 4))
        self.Tr = np.array(calib.loc['Tr:']).reshape((3, 4))

        # If in low memory mode, reset frames and load initial images and point clouds
        if low_memory:
            self.reset_frames()
            self.first_image_left = cv2.imread(self.seq_dir + 'image_0/' + self.left_image_files[0], 0)
            self.first_image_right = cv2.imread(self.seq_dir + 'image_1/' + self.right_image_files[0], 0)
            self.second_image_left = cv2.imread(self.seq_dir + 'image_0/' + self.left_image_files[1], 0)

            if lidar:
                self.first_pointcloud = np.fromfile(self.lidar_path + self.velodyne_files[0], dtype=np.float32).reshape((-1, 4))

            self.imheight = self.first_image_left.shape[0]
            self.imwidth = self.first_image_left.shape[1]

        else:
            # If not in low memory mode, load all images and point clouds
            self.images_left = []
            self.images_right = []
            self.pointclouds = []

            if progress_bar:
                bar = progressbar.ProgressBar(max_value=self.num_frames)
            for i, name_left in enumerate(self.left_image_files):
                name_right = self.right_image_files[i]
                self.images_left.append(cv2.imread(self.seq_dir + 'image_0/' + name_left))
                self.images_right.append(cv2.imread(self.seq_dir + 'image_1/' + name_right))

                if lidar:
                    pointcloud = np.fromfile(self.lidar_path + self.velodyne_files[i], dtype=np.float32).reshape((-1, 4))
                    self.pointclouds.append(pointcloud)

                if progress_bar:
                    bar.update(i + 1)
                self.imheight = self.images_left[0].shape[0]
                self.imwidth = self.images_right[0].shape[1]

    def reset_frames(self):
        # Reset frames to access images from the generator
        self.first_image_left = (cv2.imread(self.seq_dir + 'image_0/' + name_left, 0)
                                 for name_left in self.left_image_files)
        self.first_image_right = (cv2.imread(self.seq_dir + 'image_1/' + name_right, 0)
                                  for name_right in self.right_image_files)
        self.pointclouds = (np.fromfile(self.lidar_path + velodyne_file, dtype=np.float32).reshape((-1, 4))
                            for velodyne_file in self.velodyne_files)

# Main execution code

start_time = time.time()

# Initialize the dataset handler
handler = Dataset_handler('00', low_memory=True)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

# Compute disparity map using Stereo SGBM
def compute_left_disparity_map(img_left, img_right, matcher='bm', rgb=False, verbose=False):

    sad_window = 6
    num_disparities = max(16, sad_window * 16)
    block_size = 11
    matcher_name = matcher

    if matcher_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

    elif matcher_name == 'sgbm':
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities, minDisparity=0, blockSize=block_size,
                                        P1=8 * 1 * block_size ** 2,
                                        P2=32 * 1 * block_size ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    if rgb:
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    start = datetime.datetime.now()
    disp_left = matcher.compute(img_left, img_right).astype(np.float32) / 16
    end = datetime.datetime.now()

    if verbose:
        print(f'Time to compute disparity map using Stereo {matcher_name.upper()}: {end - start}')

    return disp_left

# Display the computed disparity map
disp = compute_left_disparity_map(handler.first_image_left, handler.first_image_right, matcher='sgbm', verbose=True)

plt.figure(figsize=(11, 7))
plt.imshow(disp)
plt.show()

# Function to decompose the projection matrix into intrinsic and extrinsic parameters
def decompose_projection_matrix(p):
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]
    return k, r, t

# Function to calculate the depth map from disparity
def calc_depth_map(disp_left, k_left, t_left, t_right, rectified=True):

    if rectified:
        b = t_right[0] - t_left[0]
    else:
        b = t_left[0] - t_right[0]

    f = k_left[0][0]

    disp_left[disp_left== 0.0] = 0.1
    disp_left[disp_left == -1.0] = 0.1

    depth_map = np.ones(disp_left.shape)
    depth_map = f * b / disp_left

    return depth_map

# Decompose the projection matrices for the left and right cameras
k_left, r_left, t_left = decompose_projection_matrix(handler.P0)
k_right, r_right, t_right = decompose_projection_matrix(handler.P1)

# Calculate and display the depth map using the stereo pair
depth = calc_depth_map(disp, k_left, t_left, t_right)
plt.figure(figsize=(11, 7))
plt.imshow(depth)
plt.show()

# Function to compute the depth map using stereo images
def stereo_2_depth(img_left, img_right, P0, P1, matcher='bm', rgb=False, verbose=True, rectified=True):

    # Compute the disparity map
    disp = compute_left_disparity_map(img_left, img_right, matcher=matcher, rgb=rgb, verbose=True)

    # Decompose the projection matrices to get intrinsic and extrinsic parameters
    k_left, r_left, t_left = decompose_projection_matrix(P0)
    k_right, r_right, t_right = decompose_projection_matrix(P1)

    # Calculate the depth map from the disparity map
    depth = calc_depth_map(disp, k_left, t_left, t_right)

    return depth

# Compute and display the stereo depth map
depth_stereo = stereo_2_depth(handler.first_image_left, handler.first_image_right, handler.P0, handler.P1, matcher='sgbm', rgb=False, verbose=True)

plt.grid = False
plt.imshow(depth_stereo)
plt.show()

# Function to project the point cloud into the image frame
def pointcloud2image(pointcloud, imheight, imwidth, Tr, P0):
    pointcloud = pointcloud[pointcloud[:, 0] > 0]
    reflectance = pointcloud[:, 3]

    # Make the point cloud homogeneous
    pointcloud = np.hstack([pointcloud[:, :3], np.ones(pointcloud.shape[0]).reshape((-1, 1))])

    # Transform the points into the 3D coordinate frame of the camera
    cam_xyz = Tr.dot(pointcloud.T)

    # Clip off the negative z values
    cam_xyz = cam_xyz[:, cam_xyz[2] > 0]

    depth = cam_xyz[2].copy()

    cam_xyz /= cam_xyz[2]
    cam_xyz = np.vstack([cam_xyz, np.ones(cam_xyz.shape[1])])
    projection = P0.dot(cam_xyz)
    pixel_coordinates = np.round(projection.T, 0)[:, :2].astype('int')

    # Filter out points that are outside the image bounds
    indices = np.where((pixel_coordinates[:, 0] < imwidth)
                       & (pixel_coordinates[:, 0] >= 0)
                       & (pixel_coordinates[:, 1] < imheight)
                       & (pixel_coordinates[:, 1] >= 0))

    pixel_coordinates = pixel_coordinates[indices]

    depth = depth[indices]
    reflectance = reflectance[indices]

    # Create an empty render and fill it with depth information
    render = np.zeros((imheight, imwidth))
    for j, (u, v) in enumerate(pixel_coordinates):
        if u >= imwidth or u < 0:
            continue
        if v >= imheight or v < 0:
            continue
        render[v, u] = depth[j]

    return render

# Reset the handler frames and load the first point cloud
render = pointcloud2image(handler.first_pointcloud, handler.imheight, handler.imwidth, handler.Tr, handler.P0)

# Create a generator for the poses and point clouds
handler.reset_frames()
poses = (gt for gt in handler.gt)
pcloud_frames = (pointcloud2image(next(handler.first_pointcloud), handler.imheight, handler.imwidth, handler.Tr, handler.P0)
                 for i in range(handler.num_frames))

# Visualize the ground truth trajectory and stereo data
xs = []
ys = []
zs = []
compute_times = []
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=-20, azim=270)
ax.plot(handler.gt[:, 1, 3], handler.gt[:, 2, 3], c='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Loop through frames and visualize the data
for i in range(handler.num_frames // 50):
    img_l = cv2.imread(handler.seq_dir + 'image_0/' + handler.left_image_files[i], 0)
    img_r = cv2.imread(handler.seq_dir + 'image_1/' + handler.right_image_files[i], 0)
    start = datetime.datetime.now()
    disp = compute_left_disparity_map(img_l, img_r, matcher='sgbm')
    disp /= disp.max()
    disp = (disp * 255).astype('uint8')
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_RAINBOW)
    pcloud = np.fromfile(handler.lidar_path + handler.velodyne_files[i], dtype=np.float32).reshape(-1, 4)
    pcloud_img = (pointcloud2image(pcloud, handler.imheight, handler.imwidth, handler.Tr, handler.P0) * 255).astype('uint8')

    gt = next(poses)
    xs.append(gt[0, 3])
    ys.append(gt[1, 3])
    zs.append(gt[2, 3])
    plt.plot(xs, ys, zs, c='chartreuse')
    plt.pause(0.000000000000000001)
    cv2.imshow('camera', img_l)
    cv2.imshow('disparity', disp)
    cv2.imshow('lidar', pcloud_img)
    cv2.waitKey(1)

    end = datetime.datetime.now()
    compute_times.append(end - start)

plt.close()
cv2.destroyAllWindows()

# Define functions to extract and match features between images
mask = None

def extract_features(image, detector='sift', mask=None):
    if detector == 'sift':
        det = cv2.SIFT_create()
    elif detector == 'orb':
        det = cv2.ORB_create()

    kp, des = det.detectAndCompute(image, mask)
    return kp, des

def match_features(des1, des2, matching='BF', detector='sift', sort=False, k=2):
    if matching == 'BF':
        if detector == 'sift':
            matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        elif detector == 'orb':
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=False)
    elif matching == 'FLANN':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(alorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    matches = matcher.knnMatch(des1, des2, k=k)

    if sort:
        matches = sorted(matches, key=lambda x: x[0].distance)

    return matches

def visualize_matches(image1, kp1, image2, kp2, matches, output_file=None):
    # Draw matches between the images
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Convert BGR to RGB for Matplotlib
    image_matches_rgb = cv2.cvtColor(image_matches, cv2.COLOR_BGR2RGB)

    # Display using Matplotlib
    plt.figure(figsize=(16, 6))
    plt.imshow(image_matches_rgb)
    plt.axis('off')  # Hide axis

    # Save the image if output_file is provided
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')

    # Show the plot
    plt.show()

# Function to filter matches based on distance threshold
def filter_matches_distance(matches, dist_threshold=0.5):
    filtered_matches = []
    for m, n in matches:
        if m.distance <= dist_threshold * n.distance:
            filtered_matches.append(m)

    return filtered_matches

# Reset frames and get images from the handler
handler.reset_frames()
image_left = next(handler.first_image_left)
image_right = next(handler.first_image_right)
image_plus1 = next(handler.first_image_left)  # Access the next image in the sequence

# Ensure images are loaded properly
if image_left is None or image_right is None or image_plus1 is None:
    print("Error: One of the images could not be loaded.")
    exit()

# Print the image type and shape for debugging
print(f"image_left type: {type(image_left)}, shape: {image_left.shape if image_left is not None else 'None'}")
print(f"image_right type: {type(image_right)}, shape: {image_right.shape if image_right is not None else 'None'}")
print(f"image_plus1 type: {type(image_plus1)}, shape: {image_plus1.shape if image_plus1 is not None else 'None'}")

# Optionally convert to grayscale if not already

if len(image_left.shape) == 3:
    image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
if len(image_right.shape) == 3:
    image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
if len(image_plus1.shape) == 3:
    image_plus1 = cv2.cvtColor(image_plus1, cv2.COLOR_BGR2GRAY)

# Ensure correct data type
if image_left.dtype != np.uint8:
    image_left = image_left.astype(np.uint8)
if image_right.dtype != np.uint8:
    image_right = image_right.astype(np.uint8)
if image_plus1.dtype != np.uint8:
    image_plus1 = image_plus1.astype(np.uint8)

# Start the feature extraction and matching process
start = datetime.datetime.now()
kp0, des0 = extract_features(image_left, 'sift', None)
kp1, des1 = extract_features(image_plus1, 'sift', None)

# Match features between the two frames and filter them
matches = match_features(des0, des1, matching='BF', detector='sift', sort=False)
print('Number of matches before filtering:', len(matches))
matches = filter_matches_distance(matches, 0.5)
print('Number of matches after filtering:', len(matches))
end = datetime.datetime.now()
print('Time to match and filter:', end-start)

# Visualize the matches
visualize_matches(image_left, kp0, image_plus1, kp1, matches, output_file='matches_visualization_sift.png')

# Now using ORB feature extraction and matching
start = datetime.datetime.now()
kp0, des0 = extract_features(image_left, 'orb', None)
kp1, des1 = extract_features(image_plus1, 'orb', None)

matches = match_features(des0, des1, matching='BF', detector='orb', sort=False)
print('Number of matches before filtering:', len(matches))
matches = filter_matches_distance(matches, 0.3)
print('Number of matches after filtering:', len(matches))
end = datetime.datetime.now()
print('Time to match and filter:', end-start)

visualize_matches(image_left, kp0, image_plus1, kp1, matches, output_file='matches_visualization_orb.png')

# Function to estimate motion based on matches and depth
def estimate_motion(matches, kp1, kp2, k, depth1, max_depth=3000):
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))

    image1_points = np.float32([kp1[m.queryIdx].pt for m in matches])
    image2_points = np.float32([kp2[m.trainIdx].pt for m in matches])

    cx = k[0, 2]
    cy = k[1, 2]
    fx = k[0, 0]
    fy = k[1, 1]

    object_points = np.zeros((0, 3))
    delete = []

    for i, (u, v) in enumerate(image1_points):
        z = depth1[int(round(v)), int(round(u))]

        if z > max_depth:
            delete.append(i)
            continue

        x = z * (u - cx) / fx
        y = z * (v - cy) / fy
        object_points = np.vstack([object_points, np.array([x, y, z])])

    image1_points = np.delete(image1_points, delete, 0)
    image2_points = np.delete(image2_points, delete, 0)

    _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k, None)
    rmat = cv2.Rodrigues(rvec)[0]

    return rmat, tvec, image1_points, image2_points

k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(handler.P0)

# Estimate motion using the matches and depth map
rmat, tvec, image1_points, image2_points = estimate_motion(matches, kp0, kp1, k, depth_stereo)

print('Rotation Matrix:')
print(rmat)
print('Translation Matrix:')
print(tvec)

transformation_matrix = np.hstack([rmat, tvec])
print('Transformation Matrix:')
print(transformation_matrix.round(4))
print('Ground Truth Matrix:')
print(handler.gt[1].round(4))

# Function to perform visual odometry
def visual_odometry(handler, detector='sift', matching='BF', filter_match_distance=None, stereo_matcher='sgbm', mask=None, subset=None, plot=False):
    lidar = handler.lidar
    print('Generating disparities with Stereo{}'.format(str.upper(stereo_matcher)))
    print('Detecting features with {} and matching with {}'.format(str.upper(detector), matching))

    if filter_match_distance is not None:
        print('Filtering feature matches at threshold of {}*distance'.format(filter_match_distance))
    if lidar:
        print('Improving stereo depth estimation with lidar data')
    if subset is not None:
        num_frames = subset
    else:
        num_frames = handler.num_frames

    if plot:
        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=20, azim=270)
        xs = handler.gt[:, 0, 3]
        ys = handler.gt[:, 1, 3]
        zs = handler.gt[:, 2, 3]
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.plot(xs, ys, zs, c='k', label='Ground Truth')
        ax.legend()

    # Initialize transformation matrix
    T_tot = np.eye(4)
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]
    imheight = handler.imheight
    imwidth = handler.imwidth

    k_left, r_left, t_left = decompose_projection_matrix(handler.P0)

    if handler.low_memory:
        handler.reset_frames()
        image_plus1 = next(handler.first_image_left)

    # Iterate through all frames
    for i in range(num_frames - 1):
        start = datetime.datetime.now()

        if handler.low_memory:
            image_left = image_plus1
            image_plus1 = next(handler.first_image_left)
            image_right = next(handler.first_image_right)
        else:
            image_left = handler.images_left[i]
            image_plus1 = handler.images_left[i + 1]
            image_right = handler.images_right[i]

        depth = stereo_2_depth(image_left, image_right, P0=handler.P0, P1=handler.P1, matcher=stereo_matcher)

        if lidar:
            if handler.low_memory:
                pointcloud = next(handler.pointclouds)
            else:
                pointcloud = handler.pointclouds[i]

            lidar_depth = pointcloud2image(pointcloud, imheight=imheight, imwidth=imwidth, Tr=handler.Tr, P0=handler.P0)

            indices = np.where(lidar_depth > 0)
            depth[indices] = lidar_depth[indices]

        # Get keypoints and descriptors for the left camera images of two sequential frames
        kp0, des0 = extract_features(image_left, detector, mask)
        kp1, des1 = extract_features(image_plus1, detector, mask)

        # Get matches between features detected in two sequential frames
        matches_unfilt = match_features(des0, des1, matching=matching, detector=detector)

        # Filter matches if a distance threshold is provided by the user
        if filter_match_distance is not None:
            matches = filter_matches_distance(matches_unfilt, filter_match_distance)
        else:
            matches = matches_unfilt

        # Estimate motion between sequential images of the left camera
        rmat, tvec, img1_points, img2_points = estimate_motion(matches, kp0, kp1, k_left, depth)

        # Create a transformation matrix
        Tmat = np.eye(4)
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T

        T_tot = T_tot.dot(np.linalg.inv(Tmat))
        trajectory[i + 1, :, :] = T_tot[:3, :]

        end = datetime.datetime.now()

        print(f'Time to compute frame {i + 1}: {end - start}')

        if plot:
            xs = trajectory[:i + 2, 0, 3]
            ys = trajectory[:i + 2, 1, 3]
            zs = trajectory[:i + 2, 2, 3]
            ax.plot(xs, ys, zs, c='chartreuse', label='Estimated Trajectory' if i == 0 else "")
            plt.pause(1e-32)

    if plot:
        ax.legend()
        plt.show()

    return trajectory

# Perform visual odometry and plot the results
trajectory_test = visual_odometry(handler, detector='sift', matching='BF', filter_match_distance=0.3,
                                  stereo_matcher='sgbm', mask=mask, subset=50, plot=True)
