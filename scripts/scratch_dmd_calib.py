import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/asitiger')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/evomachine_repo')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/de-lta-rt')
from evomachine.dmd_socket import DMDControl

filename = '/home/idris/workspace_python/conda_evomachine3.9/evomachine_repo/evomachine/dmd_calibration_data.pkl'
with open(filename, 'rb') as f:
    calib_data = pkl.load(f)


fig, axs = plt.subplots(1, 2)
for i, ((r_dmd, c_dmd), (r_cam, c_cam), _) in enumerate(calib_data):
    marker = str(i)
    _ = axs[0].scatter(c_dmd, r_dmd, marker='$' + marker + '$')
    _ = axs[0].set_title('DMD Points')
    _ = axs[0].set_xlabel('Column')
    _ = axs[0].set_ylabel('Row')
    _ = axs[1].scatter(c_cam, r_cam, marker='$' + marker + '$')
    _ = axs[1].set_title('Camera Points')
    _ = axs[1].set_xlabel('Column')
    _ = axs[1].set_ylabel('Row')

plt.show()

dmd_points = np.array([(c_dmd, r_dmd) for ((r_dmd, c_dmd), (r_cam, c_cam), _) in calib_data])
cam_points = np.array([(c_cam, r_cam) for ((r_dmd, c_dmd), (r_cam, c_cam), _) in calib_data])
H, _ = cv2.findHomography(srcPoints=cam_points, dstPoints=dmd_points)

row_cam_point = 0
col_cam_point = 0
point_cam = np.array([(col_cam_point, row_cam_point)], dtype=np.float32)
point_dmd = cv2.perspectiveTransform(point_cam.reshape(-1, 1, 2), H)
col_dmd, row_dmd = point_dmd[0][0]

dmd = DMDControl()
dmd._load_calibration_data()
img = dmd._make_text("Hello", 1)
img_dmd = dmd.warp_image_to_dmd(img)

fig, axs = plt.subplots(1, 2)
_ = axs[0].imshow(img)
_ = axs[0].set_title('Camera Image')
_ = axs[1].imshow(img_dmd)
_ = axs[1].set_title('DMD Image')

# src_points = np.array([(x_cam_1, y_cam_1), (x_cam_2, y_cam_2), ..., (x_cam_N, y_cam_N)])
#
# # Define the destination points (x_dmd, y_dmd)
# dst_points = np.array([(x_dmd_1, y_dmd_1), (x_dmd_2, y_dmd_2), ..., (x_dmd_N, y_dmd_N)])
#
# # Calculate the homography matrix
# H, _ = cv2.findHomography(src_points, dst_points)
#
# # Now you can use this homography matrix to map points from (x_cam, y_cam) to (x_dmd, y_dmd)
# # Define a point in the camera coordinate system
# point_cam = np.array([[x_cam, y_cam]], dtype=np.float32)
#
# # Use the homography to transform the point to the DMD coordinate system
# point_dmd = cv2.perspectiveTransform(point_cam, H)
#
# # Extract the transformed coordinates
# x_dmd_transformed, y_dmd_transformed = point_dmd[0][0]