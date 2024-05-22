import numpy as np

data = np.load("sim_ws/src/icuas24_competition/gps_data/ICUAS_bag_4.npy", allow_pickle=True).item()
X = np.array(data["latitude"])
Y = np.array(data["longitude"])
Z = np.array(data["altitude"])

print(X,Y,Z)