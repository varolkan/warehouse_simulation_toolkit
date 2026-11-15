#!/usr/bin/env python3
"""Draw a site's keep-out and speed-limit masks, or check that they are up to date.

    generate_masks.py SITE_DIR            write keepout.* and speed.* next to site.yaml
    generate_masks.py --check SITE_DIR    exit 1 if they are not what would be written

SITE_DIR holds site.yaml and the map (map.yaml, map.pgm). The site is checked against the
map first: a station in a wall or a docking run through a rack stops here, with the
reason, before anything is written.
"""

import argparse
import filecmp
import os
import sys
import tempfile

from agv_core.grid import GridMap, MapError
from agv_core.masks import keepout_mask, site_map_problems, speed_mask
from agv_core.sites import Site, SiteError
from agv_core.vehicle import AGV

MASKS = (('keepout', keepout_mask, 'trinary'), ('speed', speed_mask, 'raw'))


def write_masks(site, grid, directory):
    for name, build, mode in MASKS:
        build(site, grid).save(os.path.join(directory, f'{name}.yaml'), mode=mode)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true',
                        help='compare instead of writing; exit 1 on a difference')
    parser.add_argument('site_dir', help='directory with site.yaml and map.yaml')
    args = parser.parse_args(argv)

    try:
        site = Site.load(os.path.join(args.site_dir, 'site.yaml'))
        grid = GridMap.load(os.path.join(args.site_dir, 'map.yaml'))
    except (SiteError, MapError) as error:
        print(error, file=sys.stderr)
        return 2
    problems = site.vehicle_problems(AGV.half_length, AGV.half_width)
    problems += site_map_problems(site, grid, AGV.half_length, AGV.half_width)
    if problems:
        print(f"site '{site.name}' does not fit its map:", file=sys.stderr)
        for problem in problems:
            print(f'  - {problem}', file=sys.stderr)
        return 2

    if not args.check:
        write_masks(site, grid, args.site_dir)
        print(f"wrote the keep-out and speed masks of '{site.name}' ({len(site.keepout_zones)} "
              f'keep-out zones, {len(site.speed_zones)} speed zones)')
        return 0

    with tempfile.TemporaryDirectory() as scratch:
        write_masks(site, grid, scratch)
        stale = [
            f'{name}{extension}' for name, _, _ in MASKS for extension in ('.yaml', '.pgm')
            if not os.path.isfile(os.path.join(args.site_dir, f'{name}{extension}'))
            or not filecmp.cmp(os.path.join(scratch, f'{name}{extension}'),
                               os.path.join(args.site_dir, f'{name}{extension}'), shallow=False)]
    if stale:
        print(f"{', '.join(stale)} in {args.site_dir} do not match site.yaml: run "
              'generate_masks.py again', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
