# Third-party notices

This repository started as a fork of an existing project and still ships some of its files. This page says which files those are, who made them, and under which terms they are passed on. Everything not listed here was written for this project and is covered by [LICENSE](LICENSE).

## warehouse_simulation_toolkit (the fork's origin)

- **Source**: <https://github.com/wh200720041/warehouse_simulation_toolkit>
- **Author**: Wang Han, Nanyang Technological University, Singapore
- **License**: BSD 3-Clause. The original copyright notice is kept, unchanged, at the top of [LICENSE](LICENSE).
- **What is still here**: the choice of warehouse (the ARIAC workcell below) and the row of six bins along the north side, taken from `worlds/warehouse.world`.
- **What was replaced**: the original was a ROS 1 Noetic package for Gazebo Classic: a Pioneer 3-DX with gmapping and move_base. The robot, its description, the launch files, the navigation parameters, the keyboard teleoperation node and the demo recordings were removed. The ROS 2 packages under `ros2_ws/src` are new. The original files remain in this repository's history and in the upstream repository.

## ARIAC / GEAR workcell models

- **Files**: `ros2_ws/src/agv_gazebo/models/workcell/` and `ros2_ws/src/agv_gazebo/models/workcell_bin/` (meshes, textures, `model.config`, `model.sdf`).
- **Source**: the GEAR simulation of the Agile Robotics for Industrial Automation Competition (ARIAC), by Open Source Robotics Foundation with NIST. They reached this repository through the fork above, which credits GEAR (<https://bitbucket.org/osrf/ariac>). The same models are published in the `nist_gear` package of <https://github.com/usnistgov/ARIAC>.
- **Authors** (from each `model.config`): Nate Koenig; Cole Biesemeyer.
- **License**: Apache License 2.0, as declared in the GEAR package manifest. A copy is in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
- **Changes made here**: the meshes and textures are unchanged. `mesh_backup.dae`, an unused copy of the workcell mesh, was removed. Changes to the `model.sdf` files are listed at the top of each file.
