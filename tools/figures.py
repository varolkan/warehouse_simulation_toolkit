#!/usr/bin/env python3
"""Draw the README's result figures from a Gazebo verification run.

    python3 tools/figures.py RESULTS_DIR [RESULTS_DIR ...] --slam MAPPING_TRACK.csv
                             [--out docs/images]

A RESULTS_DIR is what tools/scenarios.py and tools/recorder.py wrote: track.csv and
<scenario>_events.csv (and clock.csv, for a track stamped with wall-clock time only). A run
that was restarted part way is given as one directory per part; each scenario is drawn
from the part that has it. MAPPING_TRACK.csv is the true and estimated pose logged during
the SLAM mapping run. Every figure is drawn twice, for light and for dark pages
(name.png, name-dark.png), and the numbers the README quotes are printed.

Needs matplotlib and numpy, and agv_core on the Python path (a sourced workspace).
"""

import argparse
import csv
import math
import os
import sys

from agv_core.geometry import Pose2D
from agv_core.grid import GridMap, OCCUPIED, UNKNOWN
from agv_core.sites import Site
from agv_core.vehicle import AGV
import matplotlib
matplotlib.use('Agg')
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE_DIR = os.path.join(REPO, 'ros2_ws', 'src', 'agv_robot', 'sites', 'warehouse')

# The reference data-viz palette: surfaces, ink, categorical slots, the blue ramp for
# magnitude and the status colours, each stepped for its surface.
THEMES = {
    'light': {
        'surface': '#fcfcfb', 'ink': '#0b0b0b', 'ink2': '#52514e', 'muted': '#898781',
        'grid': '#e1e0d9', 'axis': '#c3c2b7', 'wall': '#52514e', 'unknown': '#f0efec',
        'series': ('#2a78d6', '#eb6834', '#1baf7a'),
        'ramp': ('#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0d366b'),
        'warning': '#fab219', 'critical': '#d03b3b', 'good': '#0ca30c'},
    'dark': {
        'surface': '#1a1a19', 'ink': '#f0efec', 'ink2': '#c3c2b7', 'muted': '#898781',
        'grid': '#2c2c2a', 'axis': '#383835', 'wall': '#c3c2b7', 'unknown': '#242423',
        'series': ('#3987e5', '#d95926', '#199e70'),
        'ramp': ('#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb'),
        'warning': '#fab219', 'critical': '#d03b3b', 'good': '#0ca30c'},
}
HL, HW = AGV.half_length, AGV.half_width


# -- data -------------------------------------------------------------------------------

def read_csv(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def num(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


class Runs:
    """A verification run in one or more parts (a simulator restarted in between)."""

    def __init__(self, directories):
        self.parts = [Run(d) for d in directories]

    def of(self, scenario):
        """Return the part that has a scenario's events."""
        for part in self.parts:
            if scenario in part.events:
                return part
        raise KeyError(f"no events for the scenario '{scenario}'")

    def dockings(self):
        return [d for part in self.parts for d in part.dockings()]


class Run:
    """One verification run: the track on simulated time, and the scenarios' events."""

    def __init__(self, results):
        self.rows = read_csv(os.path.join(results, 'track.csv'))
        if self.rows and 'sim' in self.rows[0]:
            for row in self.rows:
                row['sim'] = float(row['sim'])
        else:
            self._sim_time_from_clock_log(os.path.join(results, 'clock.csv'))
        self.events = {}
        for name in ('tour', 'crate', 'person', 'estop', 'blocked', 'cancel'):
            path = os.path.join(results, f'{name}_events.csv')
            if os.path.isfile(path):
                self.events[name] = read_csv(path)

    def _sim_time_from_clock_log(self, path):
        """For a track stamped with wall-clock time only (`t`): put it on simulated time."""
        clock = read_csv(path)
        wall = np.array([float(r['wall']) for r in clock])
        sim = np.array([float(r['sim']) for r in clock])
        # Outside the clock log, the real-time factor is taken to be 1.
        first, last = wall[0] - sim[0], wall[-1] - sim[-1]
        for row in self.rows:
            t = float(row['t'])
            if t < wall[0]:
                row['sim'] = t - first
            elif t > wall[-1]:
                row['sim'] = t - last
            else:
                row['sim'] = float(np.interp(t, wall, sim))

    def between(self, t0, t1):
        return [r for r in self.rows if t0 <= r['sim'] <= t1 and not math.isnan(num(r, 'gt_x'))]

    def event(self, scenario, what, nth=0):
        found = [e for e in self.events.get(scenario, []) if e['event'] == what]
        return found[nth] if len(found) > nth else None

    def results(self, scenario):
        return [e for e in self.events.get(scenario, []) if e['event'] == 'result']

    def scenario_span(self, scenario):
        """Return the simulated time from the scenario before's end to this one's last result."""
        names = list(self.events)
        i = names.index(scenario)
        start = float(self.results(names[i - 1])[-1]['sim_t']) if i else self.rows[0]['sim']
        return start, float(self.results(scenario)[-1]['sim_t'])

    def dockings(self):
        """(station, error) for every docking run: the true pose as the run ended.

        Only up to the last result of the scenarios this part has events for: what the
        recorder caught after them is not part of the run.
        """
        out, station = [], None
        site = Site.load(os.path.join(SITE_DIR, 'site.yaml'))
        previous = None
        end = max(float(e['sim_t']) for events in self.events.values() for e in events
                  if e['event'] == 'result')
        for row in self.rows:
            if row['sim'] > end:
                break
            if row['state'] == 'docking' and "'" in row['detail']:
                station = row['detail'].split("'")[1]
            if previous is not None and previous['state'] == 'docking' \
                    and row['state'] != 'docking' and station:
                truth = Pose2D(num(row, 'gt_x'), num(row, 'gt_y'), num(row, 'gt_yaw'))
                local = truth.relative_to(site.station(station).pose)
                out.append((station, local.x, local.y, math.degrees(local.yaw), row['sim']))
            previous = row
        return out


# -- drawing helpers --------------------------------------------------------------------

def style(theme):
    t = THEMES[theme]
    plt.rcParams.update({
        'figure.facecolor': t['surface'], 'axes.facecolor': t['surface'],
        'savefig.facecolor': t['surface'], 'axes.edgecolor': t['axis'],
        'axes.labelcolor': t['ink2'], 'xtick.color': t['muted'], 'ytick.color': t['muted'],
        'xtick.labelcolor': t['ink2'], 'ytick.labelcolor': t['ink2'],
        'text.color': t['ink'], 'axes.titlecolor': t['ink'], 'grid.color': t['grid'],
        'grid.linewidth': 0.6, 'axes.linewidth': 0.8, 'font.size': 10,
        'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'legend.frameon': False, 'legend.labelcolor': t['ink2'], 'lines.linewidth': 2.0,
        'lines.solid_capstyle': 'round', 'font.family': 'DejaVu Sans',
    })
    return t


def recessive(ax, grid_axis='both'):
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def draw_hall(ax, t, site, grid, labels=True, legend=True):
    """Draw the map, the zones and the stations; return legend handles for the zones."""
    rgb = np.empty(grid.data.shape + (3,))
    rgb[:] = to_rgb(t['surface'])
    rgb[grid.data == UNKNOWN] = to_rgb(t['unknown'])
    rgb[grid.data >= OCCUPIED] = to_rgb(t['wall'])
    x0, y0, x1, y1 = grid.bounds
    ax.imshow(rgb, origin='lower', extent=(x0, x1, y0, y1), interpolation='nearest', zorder=0)
    for zone in site.zones:
        if zone.kind == 'keepout':
            ax.add_patch(Polygon(zone.polygon, closed=True, fc=t['muted'], alpha=0.28,
                                 ec='none', zorder=1))
            ax.add_patch(Polygon(zone.polygon, closed=True, fill=False, ec=t['muted'],
                                 lw=0.8, zorder=1))
        else:
            ax.add_patch(Polygon(zone.polygon, closed=True, fc=t['series'][1], alpha=0.12,
                                 ec='none', zorder=1))
            ax.add_patch(Polygon(zone.polygon, closed=True, fill=False, ec=t['series'][1],
                                 lw=0.8, zorder=1))
    for station in site.stations.values():
        outline = [station.pose.to_parent(c) for c in ((HL, HW), (-HL, HW), (-HL, -HW), (HL, -HW))]
        ax.add_patch(Polygon(outline, closed=True, fill=False, ec=t['ink2'], lw=1.0, zorder=3))
        if labels:
            ahead = station.pose.advanced(-0.95)
            ax.text(ahead.x, ahead.y, station.name, ha='center', va='center', fontsize=9,
                    color=t['ink'], zorder=6, clip_on=True,
                    bbox={'boxstyle': 'round,pad=0.2', 'fc': t['surface'], 'ec': 'none',
                          'alpha': 0.85})
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if legend:
        return [Patch(fc=t['muted'], alpha=0.45, ec=t['muted'], label='keep-out zone'),
                Patch(fc=t['series'][1], alpha=0.25, ec=t['series'][1],
                      label='speed-limited zone'),
                Patch(fill=False, ec=t['ink2'], label='station (docked outline)')]
    return []


def ramp(t):
    return LinearSegmentedColormap.from_list('speed', t['ramp'])


def track_lines(ax, rows, t, cmap=None, color=None, lw=2.0, z=4):
    xy = np.array([[num(r, 'gt_x'), num(r, 'gt_y')] for r in rows])
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    if cmap is not None:
        speed = np.array([abs(num(r, 'v')) for r in rows[1:]])
        lines = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0.0, 1.0), lw=lw,
                               capstyle='round', zorder=z)
        lines.set_array(speed)
    else:
        lines = LineCollection(segments, colors=color, lw=lw, capstyle='round', zorder=z)
    ax.add_collection(lines)
    return lines


def marker(ax, t, x, y, color, z=7):
    ax.plot([x], [y], 'o', ms=8, mfc=color, mec=t['surface'], mew=2, zorder=z)


def note(ax, t, x, y, text, dx=0.0, dy=0.6, ha='center'):
    ax.annotate(text, (x, y), (x + dx, y + dy), ha=ha, va='center', fontsize=9, color=t['ink'],
                zorder=8, arrowprops={'arrowstyle': '-', 'color': t['ink2'], 'lw': 0.8},
                bbox={'boxstyle': 'round,pad=0.25', 'fc': t['surface'], 'ec': t['axis'],
                      'lw': 0.6})


def save(fig, out, name, theme):
    path = os.path.join(out, f"{name}{'' if theme == 'light' else '-dark'}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    return path


# -- figures ----------------------------------------------------------------------------

def figure_tour(runs, site, grid, theme, out):
    t = style(theme)
    run = runs.of('tour')
    start, end = run.scenario_span('tour')
    rows = run.between(start, end)
    fig, ax = plt.subplots(figsize=(10.5, 9.4))
    handles = draw_hall(ax, t, site, grid)
    lines = track_lines(ax, rows, t, cmap=ramp(t))
    ax.set_xlim(-10.2, 10.0)
    ax.set_ylim(-10.1, 8.3)
    bar = fig.colorbar(lines, ax=ax, orientation='horizontal', fraction=0.035, pad=0.02,
                       aspect=40)
    bar.set_label('speed of the vehicle along its track (m/s)', color=t['ink2'])
    bar.outline.set_visible(False)
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3,
              fontsize=9, handlelength=1.4, borderaxespad=0.2)
    results = run.results('tour')
    distance = sum(float(r['distance']) for r in results)
    duration = sum(float(r['duration']) for r in results)
    ax.set_title(f'Six transport orders and back to the charger: {distance:.0f} m in '
                 f'{duration / 60:.1f} min, Gazebo', pad=26)
    return save(fig, out, 'tour', theme), {'distance': distance, 'duration': duration}


def figure_docking(runs, site, theme, out):
    t = style(theme)
    dockings = runs.dockings()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2),
                                      gridspec_kw={'width_ratios': (1.25, 1)})
    tol = 2.0
    for ax in (left, right):
        recessive(ax)
    left.axvspan(-tol, tol, color=t['series'][0], alpha=0.08, lw=0)
    for kind, color, label in (('transfer', t['series'][0], 'transfer stations'),
                               ('charger', t['series'][1], 'charger')):
        pts = [(d[2] * 100, d[1] * 100) for d in dockings
               if (d[0] == 'charger') == (kind == 'charger')]
        if pts:
            xs, ys = zip(*pts)
            left.plot(xs, ys, 'o', ms=8, mfc=color, mec=t['surface'], mew=2, label=label,
                      zorder=3)
    left.axhline(0, color=t['axis'], lw=0.8)
    left.axvline(0, color=t['axis'], lw=0.8)
    lim = max(4.0, max(max(abs(d[1]), abs(d[2])) for d in dockings) * 100 + 0.8)
    left.set_xlim(-lim, lim)
    left.set_ylim(-lim, lim)
    left.set_aspect('equal')
    left.set_xlabel('across the station (cm)')
    left.set_ylabel('along the station (cm, + is further in)')
    left.legend(loc='upper left', fontsize=9)
    left.text(tol - 0.1, -lim + 0.4, "±2 cm: the docking\ncontroller's target",
              ha='right', va='bottom', fontsize=8, color=t['ink2'])
    left.set_title('Where the vehicle stopped, true position')
    names = list(dict.fromkeys(d[0] for d in dockings))
    heading_tol = 1.5
    right.axhspan(-heading_tol, heading_tol, color=t['series'][0], alpha=0.08, lw=0)
    for i, name in enumerate(names):
        values = [d[3] for d in dockings if d[0] == name]
        color = t['series'][1] if name == 'charger' else t['series'][0]
        right.plot([i] * len(values), values, 'o', ms=8, mfc=color, mec=t['surface'], mew=2,
                   zorder=3)
    right.axhline(0, color=t['axis'], lw=0.8)
    right.set_xticks(range(len(names)))
    right.set_xticklabels(names, rotation=30, ha='right')
    hmax = max(3.0, max(abs(d[3]) for d in dockings) + 0.5)
    right.set_ylim(-hmax, hmax)
    right.set_ylabel('heading error (degrees)')
    right.set_title('Heading, per station')
    right.text(len(names) - 0.6, heading_tol + 0.1, '±1.5°', ha='right', va='bottom', fontsize=8,
               color=t['ink2'])
    fig.suptitle(f'{len(dockings)} dockings in Gazebo, measured against the true pose',
                 x=0.01, ha='left', fontsize=12, fontweight='bold', color=t['ink'])
    fig.tight_layout()
    across = np.array([abs(d[2]) for d in dockings]) * 100
    along = np.array([abs(d[1]) for d in dockings]) * 100
    heading = np.array([abs(d[3]) for d in dockings])
    stats = {'n': len(dockings), 'across_max': across.max(), 'across_mean': across.mean(),
             'along_max': along.max(), 'heading_max': heading.max(),
             'heading_mean': heading.mean(),
             'within_2cm_2deg': int(((across <= 2.0) & (heading <= 2.0)).sum())}
    return save(fig, out, 'docking', theme), stats


def _segment_distance(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = dx * dx + dy * dy
    k = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length)) if length else 0.0
    return math.hypot(p[0] - a[0] - k * dx, p[1] - a[1] - k * dy)


def _inside(p, polygon):
    signs = {(b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 0
             for a, b in zip(polygon, polygon[1:] + polygon[:1])}
    return len(signs) == 1


def clearance(pose, obstacle):
    """Return the gap between the vehicle's outline at `pose` and an obstacle (0 if touching).

    `obstacle` is ('disc', (x, y), radius) or ('box', Pose2D, half_length, half_width).
    """
    outline = [pose.to_parent(c) for c in ((HL, HW), (-HL, HW), (-HL, -HW), (HL, -HW))]
    edges = list(zip(outline, outline[1:] + outline[:1]))
    if obstacle[0] == 'disc':
        centre, radius = obstacle[1], obstacle[2]
        if _inside(centre, outline):
            return 0.0
        return max(0.0, min(_segment_distance(centre, a, b) for a, b in edges) - radius)
    box_pose, hl, hw = obstacle[1], obstacle[2], obstacle[3]
    box = [box_pose.to_parent(c) for c in ((hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw))]
    if any(_inside(p, outline) for p in box) or any(_inside(p, box) for p in outline):
        return 0.0
    box_edges = list(zip(box, box[1:] + box[:1]))
    return min(min(_segment_distance(p, a, b) for p in outline for a, b in box_edges),
               min(_segment_distance(p, a, b) for p in box for a, b in edges))


SAFETY_LABELS = {'slow': 'slow', 'protective_stop': 'protective stop', 'estop': 'e-stop',
                 'fault': 'fault'}


def state_strip(ax, t, rows, t0, y0, y1):
    """Colour the time the gate spent in each state other than ok, and name it."""
    colors = {'slow': t['warning'], 'protective_stop': t['critical'], 'estop': t['critical'],
              'fault': t['critical']}
    spans, current, since = [], None, None
    for row in rows:
        state = row['safety']
        if state != current:
            if current in colors:
                spans.append((current, since, row['sim'] - t0))
            current, since = state, row['sim'] - t0
    if current in colors:
        spans.append((current, since, rows[-1]['sim'] - t0))
    named = set()
    for state, a, b in spans:
        ax.add_patch(Rectangle((a, y0), max(b - a, 0.05), y1 - y0, fc=colors[state], ec='none',
                               zorder=2, clip_on=False))
        if state not in named and b - a > 0.05:
            ax.text(a + 0.08, y1 + 0.03, SAFETY_LABELS[state], fontsize=8, color=t['ink2'],
                    va='bottom', ha='left')
            named.add(state)


def figure_stops(runs, theme, out):
    t = style(theme)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.0), sharex='col',
                             gridspec_kw={'height_ratios': (1.25, 1)}, layout='constrained')
    stats = {}
    columns = (
        ('crate', 'crate placed', 'A crate dropped 1.2 m ahead', 0.2),
        ('person', 'person placed', 'A person steps in 0.9 m ahead', 0.22),
        ('estop', 'e-stop pressed', 'E-stop pressed at full speed', None),
    )
    for col, (scenario, what, title, half) in enumerate(columns):
        top, bottom = axes[0][col], axes[1][col]
        run = runs.of(scenario)
        event = run.event(scenario, what)
        t0 = float(event['sim_t'])
        rows = run.between(t0 - 2.0, t0 + 8.0)
        x = np.array([r['sim'] - t0 for r in rows])
        asked = np.array([num(r, 'v_req') for r in rows])
        passed = np.array([num(r, 'v') for r in rows])
        for ax in (top, bottom):
            recessive(ax)
            ax.axvline(0, color=t['ink2'], lw=0.8)
        top.plot(x, asked, color=t['muted'], lw=2, label='asked for by navigation')
        top.plot(x, passed, color=t['series'][0], lw=2, label='let through by the safety gate')
        top.set_ylim(-0.05, 1.12)
        state_strip(top, t, rows, t0, 1.0, 1.04)
        top.set_title(title)
        if col == 0:
            top.set_ylabel('speed (m/s)')
        if half is not None:
            ox, oy = float(event['obstacle_x']), float(event['obstacle_y'])
            if scenario == 'person':
                obstacle = ('disc', (ox, oy), half)
            else:
                obstacle = ('box', Pose2D(ox, oy, float(event['yaw'])), half, half)
            gap = np.array([clearance(Pose2D(num(r, 'gt_x'), num(r, 'gt_y'), num(r, 'gt_yaw')),
                                      obstacle) for r in rows])
            gap[x < 0] = np.nan
            bottom.plot(x, gap, color=t['series'][0], lw=2)
            stop = run.event(scenario, 'stopped')
            final = float(stop['bumper_to_obstacle'])
            closest = float(np.nanmin(gap))
            stats[scenario] = {'stopped_short': final, 'closest': closest}
            bottom.axhline(final, color=t['axis'], lw=0.8)
            bottom.text(7.8, final + 0.05, f'stopped {final:.2f} m short', ha='right',
                        va='bottom', fontsize=9, color=t['ink'])
            bottom.set_ylim(0, max(1.6, np.nanmax(gap) + 0.1))
            if col == 0:
                bottom.set_ylabel('clearance to the obstacle (m)')
        else:
            pressed = min(rows, key=lambda r: abs(r['sim'] - t0))
            p0 = (num(pressed, 'gt_x'), num(pressed, 'gt_y'))
            moved = np.array([math.hypot(num(r, 'gt_x') - p0[0], num(r, 'gt_y') - p0[1])
                              for r in rows])
            moved[x < 0] = np.nan
            stop = run.event(scenario, 'stopped')
            final = float(stop['stopping_distance'])
            stats['estop'] = final
            bottom.plot(x, moved, color=t['series'][0], lw=2)
            bottom.axhline(final, color=t['axis'], lw=0.8)
            bottom.text(7.8, final + 0.02, f'stopped in {final:.2f} m', ha='right',
                        va='bottom', fontsize=9, color=t['ink'])
            bottom.set_ylim(0, max(0.3, np.nanmax(moved) + 0.05))
            bottom.set_ylabel('travelled since the press (m)')
        bottom.set_xlabel('seconds from the event')
        bottom.set_xlim(-2, 8)
    fig.suptitle('The safety gate, in Gazebo: what navigation asked for and what reached '
                 'the drive', x=0.01, ha='left', fontsize=12, fontweight='bold', color=t['ink'])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower left', ncol=2, fontsize=9)
    return save(fig, out, 'safety-stops', theme), stats


def figure_operations(runs, site, grid, theme, out):
    t = style(theme)
    run = runs.of('blocked')
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 6.4))
    stats = {}
    # Blocked aisle.
    draw_hall(left, t, site, grid, legend=False)
    placed = run.event('blocked', 'pallet placed in aisle A')
    removed = run.event('blocked', 'pallet removed')
    first, second = run.results('blocked')[:2]
    t_place, t_removed = float(placed['sim_t']), float(removed['sim_t'])
    rows_a = run.between(t_place, t_removed)
    rows_b = run.between(t_removed, float(second['sim_t']))
    track_lines(left, rows_a, t, color=t['series'][1])
    track_lines(left, rows_b, t, color=t['series'][0])
    ox, oy = float(placed['obstacle_x']), float(placed['obstacle_y'])
    left.add_patch(Rectangle((ox - 0.65, oy - 0.5), 1.3, 1.0, fc=t['ink2'], ec='none',
                             zorder=5))
    note(left, t, ox, oy - 0.5, 'pallet', dx=-1.6, dy=0.3)
    gave_up = rows_a[-1]
    marker(left, t, num(gave_up, 'gt_x'), num(gave_up, 'gt_y'), t['series'][1])
    left.set_xlim(-9.4, 9.8)
    left.set_ylim(-10.0, 4.9)
    left.set_title('Rack aisle A closed by a pallet')
    handles = [
        plt.Line2D([], [], color=t['series'][1], lw=2,
                   label=f"pallet in the aisle: {first['outcome']} after "
                         f"{float(first['duration']):.0f} s"),
        plt.Line2D([], [], color=t['series'][0], lw=2,
                   label=f"pallet removed: {second['outcome']} after "
                         f"{float(second['duration']):.0f} s")]
    left.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, -0.01), fontsize=9)
    stats['blocked'] = (first, second)
    # Cancel.
    draw_hall(right, t, site, grid, legend=False)
    run = runs.of('cancel')
    sent = run.event('cancel', 'cancel sent')
    results = run.results('cancel')
    start, _ = run.scenario_span('cancel')
    t_cancel = float(sent['sim_t'])
    rows_a = run.between(start, t_cancel)
    rows_b = run.between(t_cancel, float(results[0]['sim_t']))
    rows_c = run.between(float(results[0]['sim_t']), float(results[-1]['sim_t']))
    track_lines(right, rows_a, t, color=t['series'][0])
    track_lines(right, rows_b, t, color=t['series'][1])
    track_lines(right, rows_c, t, color=t['series'][2])
    c = rows_a[-1]
    marker(right, t, num(c, 'gt_x'), num(c, 'gt_y'), t['series'][1])
    right.set_xlim(-9.4, 9.8)
    right.set_ylim(-10.0, 4.9)
    right.set_title('An order cancelled with the load on board')
    handles = [
        plt.Line2D([], [], color=t['series'][0], lw=2,
                   label='rack_a to inbound, until the cancel'),
        plt.Line2D([], [], color=t['series'][1], lw=2,
                   label=f"to the charger with the load: {results[0]['outcome']}"),
        plt.Line2D([], [], color=t['series'][2], lw=2,
                   label=f"next order, no pickup: {results[-1]['outcome']}")]
    right.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, -0.01), fontsize=9)
    stats['cancel'] = results
    fig.tight_layout()
    return save(fig, out, 'blocked-and-cancel', theme), stats


def figure_slam(path, site, grid, theme, out):
    t = style(theme)
    rows = read_csv(path)
    gt = np.array([[float(r['gt_x']), float(r['gt_y'])] for r in rows])
    est = np.array([[float(r['est_x']), float(r['est_y'])] for r in rows])
    yaw_err = np.degrees(np.abs(np.angle(np.exp(1j * (
        np.array([float(r['est_yaw']) for r in rows])
        - np.array([float(r['gt_yaw']) for r in rows]))))))
    err = np.hypot(*(est - gt).T) * 100
    driven = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(gt, axis=0).T))])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.0),
                                      gridspec_kw={'width_ratios': (1.1, 1)})
    draw_hall(left, t, site, grid, labels=False, legend=False)
    segments = np.stack([gt[:-1], gt[1:]], axis=1)
    left.add_collection(LineCollection(segments, colors=t['series'][0], lw=2, zorder=4))
    marker(left, t, gt[0, 0], gt[0, 1], t['series'][0])
    left.set_xlim(-10.2, 10.0)
    left.set_ylim(-10.1, 8.3)
    left.set_title('The mapping run: Nav2 through 28 waypoints')
    recessive(right)
    right.plot(driven, err, color=t['series'][0], lw=1.5)
    p95 = float(np.percentile(err, 95))
    right.axhline(p95, color=t['axis'], lw=0.8)
    right.text(driven[-1], p95 + 0.15, f'95% of the time under {p95:.1f} cm', ha='right',
               va='bottom', fontsize=9, color=t['ink'])
    right.set_xlim(0, driven[-1])
    right.set_ylim(0, max(8.0, err.max() + 0.5))
    right.set_xlabel('distance driven (m)')
    right.set_ylabel('SLAM position error (cm)')
    right.set_title('slam_toolbox against the true pose')
    fig.tight_layout()
    stats = {'length': driven[-1], 'mean': err.mean(), 'p95': p95, 'max': err.max(),
             'yaw_mean': yaw_err.mean(), 'yaw_max': yaw_err.max()}
    return save(fig, out, 'slam', theme), stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('results', nargs='+', help='results directories, in run order')
    parser.add_argument('--slam', required=True, help='track of the mapping run (CSV)')
    parser.add_argument('--out', default=os.path.join(REPO, 'docs', 'images'))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    site = Site.load(os.path.join(SITE_DIR, 'site.yaml'))
    grid = GridMap.load(os.path.join(SITE_DIR, 'map.yaml'))
    runs = Runs(args.results)
    for theme in THEMES:
        report = [figure_tour(runs, site, grid, theme, args.out),
                  figure_docking(runs, site, theme, args.out),
                  figure_stops(runs, theme, args.out),
                  figure_operations(runs, site, grid, theme, args.out),
                  figure_slam(args.slam, site, grid, theme, args.out)]
    for path, stats in report:
        print(os.path.relpath(path, REPO), stats)
    return 0


if __name__ == '__main__':
    sys.exit(main())
