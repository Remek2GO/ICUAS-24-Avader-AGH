uint8 bed_id # 1-27
uint8 bed_side # 0-1 (0: left (closer to origin), 1: right)

uint8 img_id # counter for the images taken from the same pose

string img_path_color
string img_path_depth

geometry_msgs/Pose pose