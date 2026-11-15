#!/usr/bin/env python3
"""Write the Gazebo world for a site, or check that the committed one is up to date.

    generate_world.py SITE.yaml WORLD.sdf            write the world
    generate_world.py --check SITE.yaml WORLD.sdf    exit 1 if WORLD.sdf is out of date

The world is the fixed hall plus what stands at each station of the site file.
"""

import argparse
import sys

from agv_core.sites import Site, SiteError
from agv_gazebo.world import world_sdf


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true',
                        help='compare instead of writing; exit 1 on a difference')
    parser.add_argument('site', help='site file (agv_robot/sites/<name>/site.yaml)')
    parser.add_argument('world', help='world file to write or check')
    args = parser.parse_args(argv)
    try:
        expected = world_sdf(Site.load(args.site))
    except SiteError as error:
        print(error, file=sys.stderr)
        return 2
    if args.check:
        try:
            with open(args.world) as fh:
                current = fh.read()
        except OSError as error:
            print(f'{args.world}: {error.strerror}', file=sys.stderr)
            return 1
        if current != expected:
            print(f'{args.world} does not match {args.site}: run generate_world.py again',
                  file=sys.stderr)
            return 1
        return 0
    with open(args.world, 'w') as fh:
        fh.write(expected)
    print(f'wrote {args.world}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
