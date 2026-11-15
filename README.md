# Warehouse AGV

A small automated guided vehicle that uses LiDAR for SLAM and navigation, built to move material around a simulated warehouse. ROS 2 Jazzy, Python, Gazebo Harmonic.

This repository is being rebuilt. It started as a fork of [warehouse_simulation_toolkit](https://github.com/wh200720041/warehouse_simulation_toolkit) by Wang Han, a ROS 1 Noetic package that drove a Pioneer 3-DX round the ARIAC warehouse with gmapping and move_base. The warehouse stays; the ROS 1 package has been retired and a ROS 2 stack is taking its place under `ros2_ws/src`.

What came from where, and under which terms, is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

BSD 3-Clause, see [LICENSE](LICENSE). The warehouse models are under the Apache License 2.0, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
