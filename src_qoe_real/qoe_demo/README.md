# QoE Real Demo (DASH.js + REST Telemetry)

This folder contains a practical pipeline to demo real video playback QoE and feed per-segment telemetry into `src_qoe_real` runtime.

## 1) Build DASH assets

```bash
cd /home/mec/DATN/src_qoe_real/qoe_demo/scripts
./build_dash_assets.sh /path/to/input.mp4 ../assets 2
```

Expected output: `../assets/manifest.mpd` + segment files.

## 2) Run simulation + REST env

In one terminal (normal thesis flow):

```bash
cd /home/mec/DATN/src_qoe_real
sudo python3 main_thesis.py
```

`main_thesis.py` exposes telemetry endpoints on `http://127.0.0.1:8081`:
- `POST /qoe/session/start`
- `POST /qoe/segment`

## 3) Serve the web player

In another terminal:

```bash
cd /home/mec/DATN/src_qoe_real/qoe_demo/scripts
./start_player_server.sh 8090
```

Open browser:

```text
http://127.0.0.1:8090/player/index.html?rest=http://127.0.0.1:8081&car=car1&mpd=/assets/manifest.mpd
```

Notes for accurate real-QoE mapping:
- Keep `car` query param equal to `qoe_focus_car_name` in `src_qoe_real/config.py` (default `car1`).
- Keep DASH segment duration aligned with `rl_step_interval_s` (default both `2.0s`).

## 4) Runtime artifacts

- Player telemetry CSV: `src_qoe_real/results/qoe_runtime_events.csv`
- Ryu runtime CSV: `src_qoe_real/results/ryu_deploy_training.csv` or `ryu_deploy_eval.csv`

## 5) Fit QEA calibrated surrogate

After collecting enough QoE runtime events:

```bash
cd /home/mec/DATN/src_qoe_real
python3 results/fit_qoe_surrogate.py --input qoe_runtime_events.csv --output qea_qoe_calibration.json
```

This writes `src_qoe_real/results/qea_qoe_calibration.json`, consumed by QEA objective.
