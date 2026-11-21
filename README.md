# Trophy Rarity Forecast v9

[![make.fun entry](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![make.fun rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zihaeakb/trophy-rarity-forecast-v9/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![make.fun reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zihaeakb/trophy-rarity-forecast-v9/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty **Forecast how many GTA VI players earn each trophy in week one**](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): for each of the 55 PlayStation 5 trophies of Grand Theft Auto VI, forecast five quantiles (5/25/50/75/95%) of its earn rate on Exophase at 2026-11-26T00:00:00Z.

## The model

An earn rate is earners over tracked players, so it moves under two
forces: progression (players earn the trophy, the rate rises) and dilution
(Exophase's tracked-player count jumps at a refresh, new players who have
not earned the trophy yet enter the denominator, the rate falls).

- **Frozen history, short horizon (the practice cases):** model the first
  refresh directly. The tracked count jumps from 638 to 3,868 players.
  The prologue trophy is earned by 99.5% of the new players; each
  trophy reachable in the first hours of play gains twice its old
  earner count; the rest stay put. Earner counts are then trimmed to
  the 4,276 total the refresh showed. Quantiles come from the median
  on the logit scale with a 0.2 spread.
- **Frozen history, longer horizon:** description-based priors per
  trophy class (early story +1.3, mid +0.4, everything else −3.0
  logit per day).
- **Moving history:** fit earners and tracked players separately (a
  power-law fit and a saturating fit, combined by geometric mean) and
  take their ratio, capped by a per-class week-one ceiling (early 40%,
  activities 25%, mid-game 5%, late/grind 1%). Spread grows with the
  square root of the horizon plus the fit's residual, capped at 2.5.

Python 3.13, standard library only, deterministic, offline.

## Running it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The program reads every `*.in` case in the input directory and writes
one `*.out` forecast per case to the output directory. Each output is
one JSON object with a single `trophies` key: 55 rows of
`{"id": n, "q": [q05, q25, q50, q75, q95]}`, finite and non-decreasing
within each row. Two runs produce byte-identical output.

## Files

- `main.py` — the whole forecaster (entrypoint).
- `LICENSE` — MIT.
- `.makefun/owner.json` — the payout wallet that proves this repository
  is the entrant's.
