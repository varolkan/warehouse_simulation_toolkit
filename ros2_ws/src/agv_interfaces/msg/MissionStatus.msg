# Where the vehicle's work stands, published continuously by mission_node.
std_msgs/Header header

# idle | checking | to_pickup | docking | loading | to_dropoff | unloading | to_charger |
# complete | failed | canceled
string state
string detail
# Whether an order (or a trip to the charger the vehicle gave itself) is running.
bool active
string pickup
string dropoff
string station
bool carrying
string holding
float64 distance_remaining
float64 distance_driven
float64 elapsed

# The vehicle as the mission sees it.
geometry_msgs/Pose2D pose
bool localized
# 0..1, negative when unknown.
float64 battery
bool charging
# unknown | empty | loaded | moving | fault
string load
# ok | slow | protective_stop | estop | fault
string safety
