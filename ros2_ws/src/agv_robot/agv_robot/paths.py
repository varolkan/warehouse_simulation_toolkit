"""Where a site's files are.

A site is a directory:

    site.yaml          stations and zones (agv_core.sites)
    map.yaml/.pgm      the occupancy map SLAM built
    keepout.*          the keep-out zones for the controller, with a soft rim
    keepout_planner.*  the keep-out zones for the planner, with a graded band round them
    speed.*            speed-limit mask

The three masks are generated from site.yaml (scripts/generate_masks.py); agv_core/masks.py
says why there are two keep-out masks.

Sites shipped with the package live in its share directory under sites/<name>; anything
else is given as a path.
"""

import os

from ament_index_python.packages import get_package_share_directory

SITE_FILE = 'site.yaml'
MAP_FILE = 'map.yaml'
KEEPOUT_MASK = 'keepout.yaml'
PLANNER_KEEPOUT_MASK = 'keepout_planner.yaml'
SPEED_MASK = 'speed.yaml'


def site_directory(site: str) -> str:
    """Resolve a site name or path to its directory."""
    if os.path.isdir(site):
        return os.path.abspath(site)
    shipped = os.path.join(get_package_share_directory('agv_robot'), 'sites', site)
    if os.path.isdir(shipped):
        return shipped
    raise FileNotFoundError(
        f"site '{site}' is neither a directory nor one of the sites shipped in "
        f'{os.path.dirname(shipped)}')


def site_file(site: str, name: str = SITE_FILE) -> str:
    return os.path.join(site_directory(site), name)
