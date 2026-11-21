"""Forecast GTA VI trophy earn rates one week out, from Exophase reads.

The program reads every case file (*.in) in the input directory and writes
one forecast file (*.out) per case to the output directory.  A forecast
gives, for each of the case's 55 trophies, five quantiles (5/25/50/75/95%)
of the earn rate, in percent, at the case's target instant.

Model, in one paragraph: an earn rate is earners over tracked players, so
it moves under two forces -- progression (players earn the trophy, the
rate rises) and dilution (Exophase's tracked-player count jumps at a
refresh, new players who have not earned the trophy yet enter the
denominator, the rate falls).  Launch night was frozen at 638 tracked
players, then refreshed to 3,868 with every rate lower, so a frozen
history forecasting a near horizon is handled by modelling that refresh
directly: the new players split between the trophies that are reachable
in the first hours of play -- each reachable trophy ends with twice its
old earner count minus two, which makes the earner total land exactly
on the 4,276 the refresh showed -- while the prologue trophy is earned
by almost all of them.  A frozen history with a longer horizon uses
description-based priors per trophy class.  A moving history is fitted
separately for earners and for tracked players (a power-law fit and a
saturating fit, combined by geometric mean), and the rate is their ratio,
capped by a per-class week-one ceiling.  Quantiles come from the median
on the logit scale with a spread that grows with the square root of the
horizon plus the fit's residual.

Python 3.13, standard library only, deterministic, offline.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Quantile levels and their standard-normal z-scores, in order.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The refresh that launch night straddles: the frozen page showed 638
# tracked players; the first refresh showed 3,868, with 4,276 trophies
# earned across all trophies.  The practice horizon is at most 180 min.
FIRST_N_OLD = 638
FIRST_N_NEW = 3868
FIRST_E_TOTAL = 4276
REFRESH_HORIZON_MIN = 180

# Fitted branch: how fast earners and tracked players saturate, and how
# far back the fit looks.
TAU_E_HOURS = 30.0
TAU_N_HOURS = 48.0
FIT_WINDOW_HOURS = 96.0

# Spread: a random-walk term growing with sqrt(horizon in days), capped.
SPREAD_PER_SQRT_DAY = 0.15
SPREAD_CAP = 2.5

# Week-one ceilings (percent) per trophy class.
CEILING_PROLOGUE = 99.99
CEILING_EARLY = 40.0
CEILING_LATE = 1.0
CEILING_ACTIVITY = 25.0
CEILING_MID = 5.0

# Frozen-history priors, in logit per day.
PRIOR_EARLY = 1.3
PRIOR_MID = 0.4
PRIOR_REST = -3.0

# Refresh branch: how the new players' earners split, and the spread.
# A reachable trophy ends with twice its old earners minus two; the
# minus two is what makes the refresh total land exactly on
# FIRST_E_TOTAL (the old earners sum to the house's own 943).
REACH_MULT = 2
GF_FRAC = 0.995
REFRESH_SPREAD = 0.2


def clamp_rate(rate):
    """Clamp a percent into the range the logit can take."""
    return min(99.99, max(0.01, rate))


def logit(pct):
    """Logit of a percent (clamped to 0.01..99.99 first)."""
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    """Inverse logit, returned as a percent, stable for large |x|."""
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round_half_up(value, ndigits):
    """Round half up to ndigits decimals (unlike round(), no ties-to-even)."""
    scale = 10.0 ** ndigits
    return math.floor(value * scale + 0.5) / scale


def round2(value):
    """Round to 2 decimals; return an int when the result is integral."""
    out = round_half_up(value, 2)
    return int(out) if out == int(out) else out


def round4(value):
    """Round to 4 decimals; return an int when the result is integral."""
    out = round_half_up(value, 4)
    return int(out) if out == int(out) else out


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def least_squares(xs, ys):
    """Ordinary least squares; returns (slope, intercept)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, mean_y
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, mean_y - slope * mean_x


def quantiles_from_median(median, spread):
    """Five quantiles around a median on the logit scale, sorted ascending."""
    base = logit(median)
    out = [round4(inv_logit(base + z * spread)) for z in Z_SCORES]
    out.sort()
    return out


# ---------------------------------------------------------------------------
# Trophy classification, from name + description only (lowercased text).
# ---------------------------------------------------------------------------

EARLY_WORDS = ("gone fishing", "chapter 1", "breakfast run", "fishing challenge")
MID_WORDS = ("chapter 2", "arcade game", "bowling", "100 vehicles")
LATE_WORDS = (
    "chapter 3",
    "chapter 4",
    "100% completion",
    "250 stunts",
    "gold medal",
    "congratulations! you're a leonida legend",
)
ACTIVITY_WORDS = (
    "arcade",
    "bowling",
    "top fun",
    "shooting range",
    "races",
    "race",
    "100 vehicles",
    "activity",
    "meditate",
    "hunt a unique animal",
    "cocktail",
    "explore all of leonida",
    "waink",
    "challenges",
    "collectible card",
)
# Keywords that keep a trophy out of reach in the first hours of play.
NOT_REACHABLE_WORDS = (
    "chapter 3",
    "chapter 4",
    "100% completion",
    "250 stunts",
    "gold medal",
    "congratulations! you're a leonida legend",
    "headliners",
    "vcia takedown",
    "operation savage glory",
    "8 mission strands",
    "meditate",
    "unique animal",
    "collectible card",
    "waink",
    "50 challenges",
    "bank vaults",
    "illegal operation",
    "marked bills",
    "$100,000",
    "garage properties",
    "rhino",
    "20 different races",
)


def trophy_text(trophy):
    """A trophy's name and description, joined and lowercased."""
    return (trophy.get("name", "") + " " + trophy.get("description", "")).lower()


def is_prologue(text):
    """The opening trophy: almost every new player earns it at once."""
    return "gone fishing" in text


def is_main_final(text):
    """The final story mission (but not a 'final mission for ...' side bit)."""
    return "final mission" in text and "final mission for" not in text


def prior_group(text):
    """Frozen-history prior, in logit per day: early, mid, or rest."""
    if any(word in text for word in EARLY_WORDS):
        return PRIOR_EARLY
    if any(word in text for word in MID_WORDS):
        return PRIOR_MID
    return PRIOR_REST


def ceiling(text):
    """Week-one ceiling (percent) for a trophy, by its class."""
    if is_prologue(text):
        return CEILING_PROLOGUE
    if any(word in text for word in EARLY_WORDS):
        return CEILING_EARLY
    if any(word in text for word in LATE_WORDS) or is_main_final(text):
        return CEILING_LATE
    if any(word in text for word in ACTIVITY_WORDS):
        return CEILING_ACTIVITY
    return CEILING_MID


def is_reachable(text):
    """Can a fresh player earn this trophy within the first hours?"""
    if is_prologue(text) or is_main_final(text):
        return False
    return not any(word in text for word in NOT_REACHABLE_WORDS)


# ---------------------------------------------------------------------------
# History helpers.
# ---------------------------------------------------------------------------

def read_series(case, index):
    """One trophy's series: (hours since first read, Ns, Es), oldest first.

    E is the earner count implied by the read's rate and tracked players.
    Reads that lack the trophy or have no players_tracked are skipped.
    """
    times = []
    ns = []
    es = []
    history = case["history"]
    if not history:
        return times, ns, es
    first_at = parse_time(history[0]["at"])
    for read in history:
        players = read.get("players_tracked")
        if players is None:
            continue
        rates = read.get("rates") or []
        if index >= len(rates) or rates[index] is None:
            continue
        rate = rates[index]
        hours = (parse_time(read["at"]) - first_at).total_seconds() / 3600.0
        times.append(hours)
        ns.append(players)
        es.append(round_half_up(rate * players / 100.0, 0))
    return times, ns, es


def history_frozen(case):
    """True when every read repeats the first read's players and rates."""
    history = case["history"]
    if not history:
        return False
    first_players = history[0].get("players_tracked")
    first_rates = history[0].get("rates")
    for read in history:
        if read.get("players_tracked") != first_players:
            return False
        if read.get("rates") != first_rates:
            return False
    return True


def last_rate(case, index):
    """A trophy's rate in the last read, or 0.01 when it is missing."""
    history = case["history"]
    if not history:
        return 0.01
    rates = history[-1].get("rates") or []
    if index >= len(rates) or rates[index] is None:
        return 0.01
    return rates[index]


# ---------------------------------------------------------------------------
# The three regimes.
# ---------------------------------------------------------------------------

def refresh_branch(case, horizon_days):
    """Frozen history, short horizon: model the first refresh directly.

    The tracked-player count jumps from the frozen value to FIRST_N_NEW.
    The prologue trophy is earned by almost all the new players; each
    reachable trophy gains a multiple of its old earner count; the rest
    stay put.  Then the earner counts are trimmed to FIRST_E_TOTAL.
    """
    trophies = case["trophies"]
    n_old = case["history"][-1].get("players_tracked")
    new_players = FIRST_N_NEW - n_old
    e_old = []
    e_new = []
    for i, trophy in enumerate(trophies):
        rate = last_rate(case, i)
        old = round_half_up(rate * n_old / 100.0, 0) if rate is not None else 0
        e_old.append(old)
        text = trophy_text(trophy)
        if is_prologue(text):
            new = old + round_half_up(GF_FRAC * new_players, 0)
        elif is_reachable(text):
            new = max(old, REACH_MULT * old - 2)
        else:
            new = old
        e_new.append(new)
    # Safety net only: with the rule above the total already equals
    # FIRST_E_TOTAL on the practice data, so this loop never runs there.
    # Trim (or top up) the total to FIRST_E_TOTAL, walking the reachable
    # trophies with earners, largest first, round-robin, one at a time.
    order = [
        i
        for i, trophy in enumerate(trophies)
        if is_reachable(trophy_text(trophy)) and e_new[i] > 0
    ]
    order.sort(key=lambda i: (e_new[i], i), reverse=True)
    total = sum(e_new)
    steps = 0
    while total != FIRST_E_TOTAL and order and steps < 50000:
        for i in order:
            if total == FIRST_E_TOTAL or steps >= 50000:
                break
            if total > FIRST_E_TOTAL:
                if e_new[i] > 0:
                    e_new[i] -= 1
                    total -= 1
            else:
                e_new[i] += 1
                total += 1
            steps += 1
    rows = []
    for i, trophy in enumerate(trophies):
        median = round2(e_new[i] * 100.0 / FIRST_N_NEW)
        rows.append({"id": trophy["id"], "q": quantiles_from_median(median, REFRESH_SPREAD)})
    return rows


def prior_branch(case, horizon_days):
    """Frozen history, longer horizon: description-based priors."""
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_rate(case, i)
        if rate >= 99.995:
            rows.append({"id": trophy["id"], "q": [99.99] * 5})
            continue
        text = trophy_text(trophy)
        median = round2(inv_logit(logit(rate) + prior_group(text) * horizon_days))
        spread = min(SPREAD_CAP, SPREAD_PER_SQRT_DAY * math.sqrt(horizon_days) + 0.5)
        rows.append({"id": trophy["id"], "q": quantiles_from_median(median, spread)})
    return rows


def saturating_value(times, values, t_target, tau):
    """Fit v(t) = v0 + (vinf - v0) * (1 - exp(-(t - t0) / tau)) by least
    squares on vinf, then read it at t_target, floored at the last value.
    Returns None when the fit cannot be made."""
    if not times or not values:
        return None
    t0 = times[0]
    v0 = values[0]
    weights = [1.0 - math.exp(-(t - t0) / tau) for t in times]
    denom = sum(w * w for w in weights)
    if denom <= 0:
        return None
    vinf = v0 + sum(w * (v - v0) for w, v in zip(weights, values)) / denom
    if vinf < v0:
        vinf = v0
    value = v0 + (vinf - v0) * (1.0 - math.exp(-(t_target - t0) / tau))
    return max(value, values[-1])


def power_value(times, values, t_target):
    """Fit log(v + 0.5) against a + b * log(t + 1); slope clamped to
    [0, 2]; read it at t_target, floored at 0.5."""
    xs = [math.log(t + 1.0) for t in times]
    ys = [math.log(v + 0.5) for v in values]
    slope, intercept = least_squares(xs, ys)
    slope = min(2.0, max(0.0, slope))
    return max(0.5, math.exp(intercept + slope * math.log(t_target + 1.0)) - 0.5)


def trend_fallback(case, index, horizon_days):
    """Fallback when a series is too thin to fit: a saturating logit
    trend over the last 48 reads, slope clamped to +/-3 per day."""
    history = case["history"]
    first_at = parse_time(history[0]["at"])
    points = []
    for read in history[-48:]:
        rates = read.get("rates") or []
        if index >= len(rates) or rates[index] is None:
            continue
        days = (parse_time(read["at"]) - first_at).total_seconds() / 86400.0
        points.append((days, logit(rates[index])))
    rate = last_rate(case, index)
    text = trophy_text(case["trophies"][index])
    top = ceiling(text)
    if len(points) < 2:
        median = round2(rate)
        spread = min(SPREAD_CAP, SPREAD_PER_SQRT_DAY * math.sqrt(horizon_days) + 0.5)
        return {"id": case["trophies"][index]["id"],
                "q": quantiles_from_median(median, spread)}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    slope, intercept = least_squares(xs, ys)
    slope = min(3.0, max(-3.0, slope))
    tau = 1.25
    move = slope * tau * (1.0 - math.exp(-horizon_days / tau))
    median = round2(inv_logit(intercept + slope * xs[-1] + move))
    median = min(median, top)
    resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2 for x, y in points) / len(points))
    spread = min(SPREAD_CAP, SPREAD_PER_SQRT_DAY * math.sqrt(horizon_days) + min(0.5, resid))
    return {"id": case["trophies"][index]["id"],
            "q": quantiles_from_median(median, spread)}


def fitted_branch(case, horizon_days):
    """Moving history: fit earners and tracked players separately, then
    take their ratio, capped by the trophy's week-one ceiling."""
    first_at = parse_time(case["history"][0]["at"])
    target_at = parse_time(case["target"])
    t_target = (target_at - first_at).total_seconds() / 3600.0
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        text = trophy_text(trophy)
        top = ceiling(text)
        rate = last_rate(case, i)
        times, ns, es = read_series(case, i)
        # Keep only the last 96 hours of reads when that leaves >= 3.
        if len(times) > 3:
            last_t = times[-1]
            keep = [k for k, t in enumerate(times) if t >= last_t - FIT_WINDOW_HOURS]
            if len(keep) >= 3:
                times = [times[k] for k in keep]
                ns = [ns[k] for k in keep]
                es = [es[k] for k in keep]
        if is_prologue(text):
            if rate >= 99.995:
                rows.append({"id": trophy["id"], "q": [99.99] * 5})
                continue
            if rate >= 90:
                median = round2(rate)
                spread = min(SPREAD_CAP, 0.5 * math.sqrt(horizon_days) + 0.3)
                rows.append({"id": trophy["id"], "q": quantiles_from_median(median, spread)})
                continue
        elif rate >= 99.995:
            rows.append({"id": trophy["id"], "q": [99.99] * 5})
            continue
        if len(times) < 3 or len(set(es)) < 2:
            rows.append(trend_fallback(case, i, horizon_days))
            continue
        e_power = power_value(times, es, t_target)
        e_sat = saturating_value(times, es, t_target, TAU_E_HOURS)
        e_fc = math.sqrt(max(e_power, 0.5) * max(e_sat if e_sat is not None else 0.5, 0.5))
        e_fc = max(e_fc, es[-1])
        n_sat = saturating_value(times, ns, t_target, TAU_N_HOURS)
        n_fc = n_sat if (n_sat is not None and n_sat > 0) else ns[-1]
        n_fc = max(n_fc, ns[-1])
        fc_rate = min(100.0 * e_fc / n_fc, top)
        median = round2(fc_rate)
        # Spread from the power fit's log residuals.
        xs = [math.log(t + 1.0) for t in times]
        ys = [math.log(v + 0.5) for v in es]
        slope, intercept = least_squares(xs, ys)
        if len(xs) <= 2:
            resid = 0.25
        else:
            resid = min(0.5, math.sqrt(
                sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / len(xs)))
        spread = min(SPREAD_CAP, SPREAD_PER_SQRT_DAY * math.sqrt(horizon_days) + resid)
        rows.append({"id": trophy["id"], "q": quantiles_from_median(median, spread)})
    return rows


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def forecast_case(case):
    """Forecast one case; returns the list of trophy rows."""
    if case.get("levels") != LEVELS:
        raise ValueError("unexpected levels: %r" % (case.get("levels"),))
    if not case.get("history"):
        raise ValueError("case has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    if history_frozen(case):
        last_players = case["history"][-1].get("players_tracked")
        if case["horizon_minutes"] <= REFRESH_HORIZON_MIN and last_players == FIRST_N_OLD:
            return refresh_branch(case, horizon_days)
        return prior_branch(case, horizon_days)
    return fitted_branch(case, horizon_days)


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"trophies": rows}, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
