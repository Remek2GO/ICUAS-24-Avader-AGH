from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import open3d as o3d
from transformations import euler_matrix
from scipy.spatial.transform import Rotation as R
import tkinter as tk
from PIL import ImageTk, Image
import math

def generate_image(ROLL, PITCH, YAW, X, Y, Z, SCALE): 
    pcd = o3d.io.read_point_cloud("/root/sim_ws/src/icuas24_competition/plycal/lidar.pcd")
    image = cv2.imread("/root/sim_ws/src/icuas24_competition/plycal/camera_rgb.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    points_lidar = np.asarray(pcd.points)

    # Filter the lidar data in X-axis
    front_lidar_data = points_lidar[
        np.logical_and(
            points_lidar[:, 0] > 0, points_lidar[:, 0] < np.inf
        )
    ]
    # Filter the lidar data in Y-axis
    front_lidar_data = front_lidar_data[
        np.logical_and(front_lidar_data[:, 1] > -20, front_lidar_data[:, 1] < 20)
    ]

    # calculate distance at point y = 0 and z = 0 (in the lidar frame)
    distance_x = front_lidar_data[
        np.logical_and(front_lidar_data[:, 1] > -0.1, front_lidar_data[:, 1] < 0.1)
    ][:, 0]

    # calculate the distance of the lidar data and
    # calculate half size of image
    distance_lidar = np.mean(distance_x) / 1
    image_lidar_data = front_lidar_data[
        np.logical_and(
            front_lidar_data[:, 1] < distance_lidar,
            front_lidar_data[:, 1] > -1 * distance_lidar,
        )
    ]

    points_lidar = image_lidar_data
    lidar_camera_matrix = euler_matrix(np.pi, np.pi / 18, 0)[:3, :3]
    lidar_points_cam = np.dot(lidar_camera_matrix, np.array(points_lidar)[:, :3].T).T



    # _lidar_pose = R.from_rotvec([math.radians(ROLL),math.radians(PITCH),math.radians(YAW)]).as_matrix()  
    _lidar_pose = R.from_euler('xyz',[math.radians(ROLL),math.radians(PITCH), math.radians(YAW)], degrees=False).as_matrix()
    _lidar_translation = np.array([X, Y, Z]) 

    trans_matrix = np.hstack((_lidar_pose,_lidar_translation.reshape(3,1)))
    trans_matrix = np.vstack((trans_matrix,np.array([0, 0, 0, 1])))

    
    #     # + np.dot(self._lidar_pose, self._lidar_translation)
    points_lidar = lidar_points_cam

    

    # print(np.hstack((points_lidar[:,:3],np.array([1]))).T)
    _lidar_points = np.dot(_lidar_pose, np.array(points_lidar)[:, :3].T).T + np.dot(_lidar_pose, _lidar_translation)
    # _lidar_points = np.dot(trans_matrix, np.array(np.c_[points_lidar[:,:3],np.ones(points_lidar.shape[0])].T)).T 
    # _lidar_points = lidar_points_cam

    # lidar_camera_matrix2 = euler_matrix(math.radians(-140), math.radians(-156),math.radians(126))[:3, :3]
    # _lidar_points = np.dot(lidar_camera_matrix2, np.array(_lidar_points)[:, :3].T).T

    # -140.0 -156.0 126.0 -0.489 2.12 0.87 1.4

    # print(_lidar_points)

    # _lidar_points = np.delete(_lidar_points ,3,1)

    # # Update global map
    # if self._current_rpy is not None:
    #     rot_matrix = euler_matrix(
    #         self._current_rpy[0], self._current_rpy[1], self._current_rpy[2]
    #     )[:3, :3]
    #     self._global_map = np.dot(rot_matrix, self._lidar_points.T).T


    # # Filter the lidar data in X-axis 
    # # self._lidar_points --> [x, y, z]  

    # # Filter the lidar data in X-axis
    # front_lidar_data = self._lidar_points[
    #     np.logical_and(self._lidar_points[:, 0] > 0, self._lidar_points[:, 0] < np.inf)
    # ]
    # # Filter the lidar data in Y-axis
    # front_lidar_data = front_lidar_data[
    #     np.logical_and(front_lidar_data[:, 1] > -15, front_lidar_data[:, 1] < 15)
    # ]

    # # calculate distance at point y = 0 and z = 0 (in the lidar frame)
    # distance_x = front_lidar_data[
    #     np.logical_and(front_lidar_data[:, 1] > -0.1, front_lidar_data[:, 1] < 0.1)
    # ][:, 0]

    # # calculate the distance of the lidar data and
    # # calculate half size of image
    # distance_lidar = np.mean(distance_x) / 2
    # image_lidar_data = front_lidar_data[
    #     np.logical_and(
    #         front_lidar_data[:, 1] < distance_lidar,
    #         front_lidar_data[:, 1] > -1 * distance_lidar,
    #     )
    # ]
    # self._lidar_image_points  = image_lidar_data

    K = np.array([[556.4514483892963, 0.0, 319.2977064997255],
                    [0.0, 555.4048909642523, 223.04611185085423],
                    [0.0, 0.0, 1.0]])

    # image = np.zeros((480, 640, 3), dtype=np.uint8)

    point_on_2D = np.zeros((_lidar_points.shape[0], 3))
    
    for count, point in enumerate(_lidar_points[:]):
        point_on_2D[count] = np.dot(K, point.T).T/5


    for point in point_on_2D[:]:
        image = cv2.circle(image, (int(point[1]*SCALE),int(point[0]*SCALE)), 1, (0, 0, 255), -1)

    # #save the image
    # print("Saving image...")
    # cv2.imwrite('/home/remek2go/Desktop/lidar_image.jpg', image)
    return image, point_on_2D*SCALE, _lidar_points

def dense_map(Pts, n, m, grid):
    ng = 2 * grid + 1
    
    mX = np.zeros((m,n)) + np.cfloat("inf")
    mY = np.zeros((m,n)) + np.cfloat("inf")
    mD = np.zeros((m,n))
    mX[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    mY[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    mD[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[2]
    
    KmX = np.zeros((ng, ng, m - ng, n - ng))
    KmY = np.zeros((ng, ng, m - ng, n - ng))
    KmD = np.zeros((ng, ng, m - ng, n - ng))
    
    for i in range(ng):
        for j in range(ng):
            KmX[i,j] = mX[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmY[i,j] = mY[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmD[i,j] = mD[i : (m - ng + i), j : (n - ng + j)]
    S = np.zeros_like(KmD[0,0])
    Y = np.zeros_like(KmD[0,0])
    
    for i in range(ng):
        for j in range(ng):
            s = 1/np.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
            Y = Y + s * KmD[i,j]
            S = S + s
    
    S[S == 0] = 1
    out = np.zeros((m,n))
    out[grid + 1 : -grid, grid + 1 : -grid] = Y/S
    return out


def update_image(*args):
    # Generate a new image
    image, point_on_2D, _lidar_points = generate_image(ROLL.get(), PITCH.get(), YAW.get(), X.get(), Y.get(), Z.get(), SCALE.get())
    # print(ROLL.get(), PITCH.get(), YAW.get(), X.get(), Y.get(), Z.get(), SCALE.get())

    # img_width = 640
    # img_height = 480
    # mask = (point_on_2D[:,1] >= 0) & (point_on_2D[:,1] <= img_width) & (point_on_2D[:,0] >= 0) & (point_on_2D[:,0] <= img_height)
    # point_on_2D = point_on_2D[mask,0:2]

    # # change columns order
    # point_on_2D = np.flip(point_on_2D, 1)

    # # # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
    # point_on_2D = np.concatenate((point_on_2D, _lidar_points[mask,2].reshape(-1,1)), 1)


    # out = dense_map(point_on_2D.T, 640, 480, 10)
    # print(out.shape)

    # out = (out - out.min()) / (out.max() - out.min()) * 255
    # cv2.imwrite('/root/sim_ws/src/icuas24_competition/plycal/depth.png', out)
    # cv2.imshow("Depth Map", out)
    # cv2.waitKey(0)
    
    # Check if an image was returned
    if image is None:
        print("No image was returned by generate_image")
        return

    # Convert the image to a PIL Image object if it's not already one
    if not isinstance(image, Image.Image):
        print("Converting image to PIL Image object")
        image = Image.fromarray(image)

    # Update the image in the label
    try:
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo  # Keep a reference to the image to prevent it from being garbage collected
    except Exception as e:
        print(f"Failed to update image in label: {e}")
        
window = tk.Tk()
window.title("Variable Control GUI")

# Define the variables
ROLL = tk.DoubleVar()
PITCH = tk.DoubleVar()
YAW = tk.DoubleVar()
X = tk.DoubleVar()
Y = tk.DoubleVar()
Z = tk.DoubleVar()
SCALE = tk.DoubleVar()

ROLL.set(0)
PITCH.set(63)
YAW.set(8)
X.set(-0.761)
Y.set(1.196)
Z.set(5.87)
SCALE.set(1)

# ROLL.set(0)
# PITCH.set(0)
# YAW.set(0)
# X.set(0)
# Y.set(0)
# Z.set(0)
# SCALE.set(1)

# Create sliders
slider1 = tk.Scale(window, variable=ROLL, from_=-180, to=180, resolution=1, orient=tk.HORIZONTAL, length=400, command=update_image)
slider2 = tk.Scale(window, variable=PITCH, from_=-180, to=180, resolution=1, orient=tk.HORIZONTAL, length=400, command=update_image)
slider3 = tk.Scale(window, variable=YAW, from_=-180, to=180, resolution=1, orient=tk.HORIZONTAL, length=400, command=update_image)
slider4 = tk.Scale(window, variable=X, from_=-20, to=20, resolution=0.001, orient=tk.HORIZONTAL, length=400, command=update_image)
slider5 = tk.Scale(window, variable=Y, from_=-20, to=20, resolution=0.001, orient=tk.HORIZONTAL, length=400, command=update_image)
slider6 = tk.Scale(window, variable=Z, from_=-20, to=20, resolution=0.001, orient=tk.HORIZONTAL, length=400, command=update_image)
slider7 = tk.Scale(window, variable=SCALE, from_=0, to=5, resolution=0.1, orient=tk.HORIZONTAL, length=400, command=update_image)

reset_button1 = tk.Button(window, text="ROLL", command=lambda: [ROLL.set(0), update_image()])
reset_button2 = tk.Button(window, text="PITCH", command=lambda: [PITCH.set(0), update_image()])
reset_button3 = tk.Button(window, text="YAW", command=lambda: [YAW.set(0), update_image()])
reset_button4 = tk.Button(window, text="X", command=lambda: [X.set(0), update_image()])
reset_button5 = tk.Button(window, text="Y", command=lambda: [Y.set(0), update_image()])
reset_button6 = tk.Button(window, text="Z", command=lambda: [Z.set(0), update_image()])
reset_button7 = tk.Button(window, text="SCALE", command=lambda: [SCALE.set(1), update_image()])

# Place the sliders and reset buttons in the grid
slider1.grid(row=0, column=0)
reset_button1.grid(row=0, column=1)
slider2.grid(row=1, column=0)
reset_button2.grid(row=1, column=1)
slider3.grid(row=2, column=0)
reset_button3.grid(row=2, column=1)
slider4.grid(row=3, column=0)
reset_button4.grid(row=3, column=1)
slider5.grid(row=4, column=0)
reset_button5.grid(row=4, column=1)
slider6.grid(row=5, column=0)
reset_button6.grid(row=5, column=1)
slider7.grid(row=6, column=0)
reset_button7.grid(row=6, column=1)

# Load and display the image
image_path = "/root/sim_ws/src/icuas24_competition/plycal/camera_rgb.jpg"
image = Image.open(image_path)
image = image.resize((640, 480), Image.LANCZOS)
photo = ImageTk.PhotoImage(image)
label = tk.Label(window, image=photo)
label.grid(row=7, column=0, columnspan=2)  # Use grid instead of pack

# Start the GUI event loop
window.mainloop()


# -140.0 -156.0 126.0 -0.489 2.12 0.87 1.4