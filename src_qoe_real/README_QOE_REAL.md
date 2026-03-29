# src_qoe_real: Real QoE Branch

This directory is a copy of `src` used for real-QoE implementation only.

## What changed

- Real QoE telemetry bridge added:
  - `POST /qoe/session/start`
  - `POST /qoe/segment`
- `VanetEnvironment.step()` now uses player telemetry (P.1203-like composite) as primary reward signal.
- Runtime CSV now includes QoE fields (`qoe_mos`, `qoe_cost`, `startup_sec`, `rebuffer_sec`, `switch_magnitude`, etc.).
- QEA objective switched to calibrated QoE surrogate; optional coefficients loaded from `results/qea_qoe_calibration.json`.
- Default demo alignment:
  - `qoe_focus_car_name='car1'` to keep requesting car synced with player telemetry
  - `rl_step_interval_s=2.0` to align RL step with DASH segment duration

## Demo workflow

1. Build DASH assets:
   - `qoe_demo/scripts/build_dash_assets.sh`
2. Run simulation:
   - `sudo python3 main_thesis.py`
3. Serve player:
   - `qoe_demo/scripts/start_player_server.sh 8090`
4. Open:
   - `http://127.0.0.1:8090/player/index.html?rest=http://127.0.0.1:8081&car=car1&mpd=/assets/manifest.mpd`

## Fit calibrated surrogate for QEA

After collecting telemetry in `results/qoe_runtime_events.csv`:

```bash
python3 results/fit_qoe_surrogate.py --input qoe_runtime_events.csv --output qea_qoe_calibration.json
```
