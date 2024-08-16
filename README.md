## Visual Odometry Project
This project demonstrates the process of implementing visual odometry using the KITTI dataset, which includes stereo images and LIDAR point clouds. The project covers reading and processing data, feature extraction, feature matching, depth map computation, motion estimation, and visualizing the estimated trajectory.

## Dataset
The project uses the KITTI odometry dataset, which includes:

- Stereo images (image_0 and image_1)
- LIDAR point clouds (velodyne)
- Calibration files (calib.txt)
- Ground truth poses (poses/*.txt)

## Installation
To run this project, you need to install the required Python packages:
- numpy
- opencv-python
- matplotlib
 -pandas
- progressbar2
- Download the KITTI dataset and place it in the dataset/ directory.

## Usage
Run the main script to execute the visual odometry pipeline:
- Visual_Odometry.py
The script will:
- Load the KITTI dataset.
- Compute stereo disparity maps.
- Estimate depth maps from the disparity.
- Extract and match features between sequential frames.
- Estimate motion between frames.
- Visualize the estimated trajectory and compare it with the ground truth.

## Results
The script outputs several visualizations:

- Disparity Map: Displays the computed disparity map for the stereo image pair.
- Depth Map: Shows the depth map calculated from the disparity map.
- Feature Matches: Visualizes the matches between features in sequential frames.
- Trajectory Plot: Plots the estimated trajectory against the ground truth trajectory in a 3D space.

## References
- KITTI Vision Benchmark Suite
- OpenCV Documentation
- Matplotlib Documentation
