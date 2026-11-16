# What the safety gate decided in its last cycle, published by safety_node.
std_msgs/Header header

# ok | slow | protective_stop | estop | fault
string state
string reason
# Whose command it is: nav | dock | manual | none
string source

# Asked for, and passed to the drive. m/s and rad/s.
float64 linear_requested
float64 angular_requested
float64 linear
float64 angular

# Clear path ahead of the outline along the requested motion: metres for driving, radians
# for turning on the spot, negative when nothing is in reach.
float64 gap
bool estop_pressed

# The two fields in the base_footprint frame, for display.
geometry_msgs/Polygon protective_field
geometry_msgs/Polygon warning_field
