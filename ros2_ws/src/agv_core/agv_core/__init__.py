"""Decision-making of the warehouse AGV, as a plain Python library with no ROS in it.

The ROS nodes in agv_robot only translate between topics and these classes, so the logic
can be unit tested in milliseconds without a simulator.

Modules:
  geometry     2D poses, angles, polygons
  sites        the site file: stations and zones, validated on load
  grid         occupancy grids: map files, inflation, ray casting, A*
  masks        keep-out and speed-limit masks for Nav2's costmap filters
  vehicle      the vehicle's dimensions and limits
"""

__version__ = '1.0.0'
