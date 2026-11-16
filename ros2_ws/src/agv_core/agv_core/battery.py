"""The battery: how it drains and charges, and how far what is left will go.

The model is three numbers an operator knows about a vehicle: how long it stands switched
on, how far it drives, and how long it charges. The mission's thresholds (when to refuse
an order, when to go and charge, when to stop) are in MissionConfig; this only keeps the
level and answers "how far".
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryConfig:
    """A battery described by what it does."""

    idle_hours: float = 12.0       # switched on and standing: full to empty
    range_km: float = 14.0         # driving, on top of standing: full to empty
    charge_hours: float = 1.2      # on the charger: empty to full

    @property
    def idle_drain(self) -> float:
        """Fraction of the battery used per second just by being on."""
        return 1.0 / (self.idle_hours * 3600.0)

    @property
    def drive_drain(self) -> float:
        """Fraction used per metre driven."""
        return 1.0 / (self.range_km * 1000.0)

    @property
    def charge_rate(self) -> float:
        """Fraction gained per second on the charger."""
        return 1.0 / (self.charge_hours * 3600.0)

    def range_m(self, level: float, reserve: float, speed: float) -> float:
        """Return how far the vehicle drives at `speed` before `level` drops to `reserve`."""
        usable = level - reserve
        if usable <= 0.0 or speed <= 0.0:
            return 0.0
        return usable / (self.drive_drain + self.idle_drain / speed)


class Battery:
    """State of charge between 0 and 1."""

    def __init__(self, config: BatteryConfig = BatteryConfig(), level: float = 1.0):
        self.config = config
        self.level = max(0.0, min(1.0, level))

    def step(self, dt: float, distance: float, charging: bool) -> float:
        """Advance by `dt` seconds in which the vehicle drove `distance` metres."""
        cfg = self.config
        change = -cfg.idle_drain * dt - cfg.drive_drain * abs(distance)
        if charging:
            change += cfg.charge_rate * dt
        self.level = max(0.0, min(1.0, self.level + change))
        return self.level

    @property
    def empty(self) -> bool:
        return self.level <= 0.0
