"""Interpretable mechanism scores for the PileGuard demonstration risk index."""

from __future__ import annotations

from dataclasses import asdict, dataclass

ALERT_LEVELS = ("normal", "watch", "warning", "critical")


@dataclass(frozen=True)
class RiskEvidence:
    density: float
    inflow: float
    proximity: float
    convergence: float
    directional_coherence: float
    corner: float
    context: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one, got {value}")


@dataclass(frozen=True)
class MechanismScores:
    social_attraction: float
    group_convergence: float
    external_context: float

    @property
    def maximum(self) -> float:
        return max(self.social_attraction, self.group_convergence, self.external_context)


def compute_mechanism_scores(evidence: RiskEvidence) -> MechanismScores:
    """Score three literature-derived mechanisms without claiming calibrated probability."""

    return MechanismScores(
        social_attraction=(
            0.35 * evidence.density
            + 0.30 * evidence.inflow
            + 0.25 * evidence.proximity
            + 0.10 * evidence.corner
        ),
        group_convergence=(
            0.25 * evidence.density
            + 0.15 * evidence.inflow
            + 0.35 * evidence.convergence
            + 0.25 * evidence.directional_coherence
        ),
        external_context=(
            0.20 * evidence.density
            + 0.15 * evidence.inflow
            + 0.15 * evidence.proximity
            + 0.20 * evidence.corner
            + 0.30 * evidence.context
        ),
    )


def risk_index(scores: MechanismScores) -> float:
    """Return a 0-100 demonstration index, not a calibrated event probability."""

    return 100 * scores.maximum


def alert_candidate(risk: float, thresholds: dict[str, float]) -> str:
    if risk >= thresholds["critical"]:
        return "critical"
    if risk >= thresholds["warning"]:
        return "warning"
    if risk >= thresholds["watch"]:
        return "watch"
    return "normal"


def next_alert_state(
    current: str,
    risk: float,
    *,
    thresholds: dict[str, float],
    release_margin: float,
) -> str:
    """Apply immediate escalation and hysteretic release to avoid alert flicker."""

    if current not in ALERT_LEVELS:
        raise ValueError(f"Unknown alert state: {current}")
    candidate = alert_candidate(risk, thresholds)
    current_level = ALERT_LEVELS.index(current)
    candidate_level = ALERT_LEVELS.index(candidate)
    if candidate_level >= current_level:
        return candidate
    if current == "normal":
        return "normal"
    release_threshold = thresholds[current] - release_margin
    return candidate if risk < release_threshold else current


def baseline_delta(
    value: float,
    *,
    median: float,
    interquartile_range: float,
    direction: float = 1.0,
    saturation_iqr: float = 3.0,
) -> float:
    """Normalize camera-relative change to 0-1 while keeping baseline at zero."""

    scale = max(interquartile_range * saturation_iqr, 1e-12)
    return min(max(direction * (value - median) / scale, 0.0), 1.0)
