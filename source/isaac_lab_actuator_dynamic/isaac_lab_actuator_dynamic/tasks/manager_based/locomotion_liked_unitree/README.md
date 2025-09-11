## Overview

## Train

walking rough terrain - rnn type actor

```bash
python scripts/customized_rsl_rl/train.py --task RoughWalkingRobot-v3 --num_envs 4096 --headless
```

## Play

walking rough terrain - rnn type actor

```bash
python scripts/customized_rsl_rl/play.py --task RoughWalkingRobot-Play-v3 --num_envs 1
python scripts/customized_rsl_rl/play.py --task RoughWalkingRobot-Play-v3 --num_envs 1 --load_run 2025-02-22_10-25-43 --checkpoint model_19999.pt
```
