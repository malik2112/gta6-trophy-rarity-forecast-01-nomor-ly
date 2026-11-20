# Pooled logit trend in log-time

[![make.fun - gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/malik2112/gta6-trophy-rarity-forecast-01-nomor-ly/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/malik2112/gta6-trophy-rarity-forecast-01-nomor-ly/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Built for the [make.fun bounty "Forecast how many GTA VI players earn each trophy in week one"](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Model

The logit of each rate trends in log-time with a pooled, damped slope; dilution is already inside the rate.

Earn rate = earners / players tracked. Both counts are monotone and grow
roughly linearly in log-time since launch, then saturate. Slopes are capped
and the horizon step is damped so a steep launch ramp cannot explode; the
spread is logit-normal and widens with the square root of the horizon in
days. On the flat launch-night practice histories the forecast stays at
persistence, as it should.

## The program

`main.py` (Python 3.13, standard library only) reads each case in
`$MAKEFUN_INPUT_DIR` (default `/input`) and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (default `/output`). No network; two runs on the
same case write identical bytes.

## License

MIT - see `LICENSE`.
