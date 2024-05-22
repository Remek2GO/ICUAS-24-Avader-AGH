"""Results plotter for comparison between Kalman and measurements."""

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import rosbag
import sys
import numpy as np
import utm

import pyproj

transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3035', always_xy=True)


# .bag files are stored in ~/.ros directory
if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("One mandatory argument required: path to .bag file.")
    #     exit(-1)

    # bag_path = sys.argv[1]
    root_path = "/root/sim_ws/src/icuas24_competition/"
    bag_name = "ICUAS_bag_1"
    bag_path = root_path + "bags/" + bag_name + ".bag"
    npy_path = root_path + "gps_data/" + bag_name + ".npy"


    bag = rosbag.Bag(bag_path, "r")

    recorded_data = {
        "stamp": [],
        "frame_id": [],
        "altitude": [],
        "latitude": [],
        "longitude": [],
    }

    topic: str
    for topic, msg, t in bag.read_messages():
        if topic == "/hawkblue/mavros/global_position/global":
            recorded_data["stamp"].append(msg.header.stamp.to_sec())
            recorded_data["frame_id"].append(msg.header.frame_id)
            recorded_data["altitude"].append(msg.altitude)
            recorded_data["latitude"].append(msg.latitude)
            recorded_data["longitude"].append(msg.longitude)

    np.save(npy_path, recorded_data)


    print(len(recorded_data["latitude"]))
    print(len(recorded_data["longitude"]))
    print(len(recorded_data["altitude"]))

    X = np.array(recorded_data["latitude"])
    Y = np.array(recorded_data["longitude"])
    Z = np.array(recorded_data["altitude"])

    # print(np.min(X[0]), np.min(Y[0]), Z[0])

    # X = (X - X[0])*100000
    # Y = (Y - Y[0])*100000
    EARTH_RADIUS = 6378000.0
    x = EARTH_RADIUS*np.cos(recorded_data["latitude"])*np.cos(recorded_data["longitude"])
    y = EARTH_RADIUS*np.cos(recorded_data["latitude"])*np.sin(recorded_data["longitude"])
    x = x - x[0]
    y = y - y[0]

    # normalizuj zmienne x i y do zakresu 0 1
    x = (x - np.min(x))/(np.max(x) - np.min(x))
    y = (y - np.min(y))/(np.max(y) - np.min(y))
    x2,y2 = utm.from_latlon(recorded_data["latitude"], recorded_data["longitude"])
    x2 = x2 - x2[0]
    y2 = y2 - y2[0]

    x2 = (x2 - np.min(x2))/(np.max(x2) - np.min(x2))
    y2 = (y2 - np.min(y2))/(np.max(y2) - np.min(y2))

    X, Y = transformer.transform(X, Y)
    X = X - X[0]
    Y = Y - Y[0]

    X = (X - np.min(X))/(np.max(X) - np.min(X))
    Y = (Y - np.min(Y))/(np.max(Y) - np.min(Y))


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter3D(X[0], Y[0], Z[0], c='g', marker='o')
    ax.scatter3D(X[-1], Y[-1], Z[-1], c='r', marker='o')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Altitude')
    ax.plot3D(X, Y, Z, 'gray')
    ax.plot3D(x, y, Z, 'red')
    ax.plot3D(x2, y2, Z, 'blue')

    # plt.savefig(root_path + "gps_data/" + bag_name + '_3D.png', dpi=300, bbox_inches='tight')

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(X, Y, 'gray')
    ax.plot(x,y, 'red')
    ax.plot(x2,y2, 'blue')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.scatter(X[0], Y[0], c='g', marker='o')
    ax.scatter(X[-1], Y[-1], c='r', marker='o')

    # plt.savefig(root_path + "gps_data/" + bag_name + '_2D.png', dpi=300, bbox_inches='tight')

    plt.show()

    

    bag.close()
