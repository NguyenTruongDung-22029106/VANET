(function () {
  const qp = new URLSearchParams(window.location.search);
  const restBase = qp.get("rest") || "http://127.0.0.1:8081";
  const carName = qp.get("car") || "car1";
  const mpdUrl = qp.get("mpd") || "/assets/manifest.mpd";
  const sessionId = qp.get("session") || `qoe-${Date.now()}`;

  const video = document.getElementById("videoPlayer");
  const logEl = document.getElementById("log");
  const sessionEl = document.getElementById("session");
  const carEl = document.getElementById("car");
  const qualityEl = document.getElementById("quality");
  const rebufferEl = document.getElementById("rebuffer");
  const segmentsEl = document.getElementById("segments");

  sessionEl.textContent = sessionId;
  carEl.textContent = carName;

  let segmentIdx = 0;
  let lastQuality = 0;
  let startupDone = false;
  let startupSec = 0;
  let startupBegin = performance.now();
  let waitingSince = null;
  let rebufferSecCurrent = 0;
  let rebufferCountCurrent = 0;
  let sentCount = 0;
  let lastSentSegmentIdx = -1;

  function log(msg, cls) {
    const line = document.createElement("div");
    line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    if (cls) line.className = cls;
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
  }

  async function postJson(path, payload) {
    const res = await fetch(`${restBase}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async function startSession() {
    try {
      await postJson("/qoe/session/start", {
        session_id: sessionId,
        car_name: carName,
        meta: { ua: navigator.userAgent, page: location.href },
      });
      log(`session started: ${sessionId}`, "ok");
    } catch (e) {
      log(`session start failed: ${e.message}`, "warn");
    }
  }

  function getBufferLevel() {
    try {
      if (!window.player) return 0;
      const dashMetrics = window.player.getDashMetrics();
      if (!dashMetrics) return 0;
      return dashMetrics.getCurrentBufferLevel("video") || 0;
    } catch (_) {
      return 0;
    }
  }

  async function sendSegmentEvent(downloadSec, explicitSegIdx) {
    const quality = window.player ? window.player.getQualityFor("video") : 0;
    const switchMag = Math.abs(quality - lastQuality);
    const segForPayload = Number.isFinite(explicitSegIdx) ? explicitSegIdx : segmentIdx;
    qualityEl.textContent = `${quality}`;
    rebufferEl.textContent = rebufferSecCurrent.toFixed(2);

    const payload = {
      session_id: sessionId,
      car_name: carName,
      segment_idx: segForPayload,
      startup_sec: startupDone ? 0 : startupSec,
      rebuffer_sec: rebufferSecCurrent,
      rebuffer_count: rebufferCountCurrent,
      quality_index: quality,
      switch_magnitude: switchMag,
      segment_download_sec: downloadSec,
      buffer_sec: getBufferLevel(),
    };

    try {
      await postJson("/qoe/segment", payload);
      sentCount += 1;
      segmentsEl.textContent = `${sentCount}`;
      log(`segment=${segForPayload} q=${quality} rebuf=${rebufferSecCurrent.toFixed(2)}s dl=${downloadSec.toFixed(2)}s`, "ok");
      lastSentSegmentIdx = segForPayload;
      segmentIdx = Math.max(segmentIdx, segForPayload + 1);
    } catch (e) {
      log(`segment send failed: ${e.message}`, "warn");
    }

    lastQuality = quality;
    startupDone = true;
    rebufferSecCurrent = 0;
    rebufferCountCurrent = 0;
  }

  async function init() {
    await startSession();

    const player = dashjs.MediaPlayer().create();
    window.player = player;
    player.initialize(video, mpdUrl, true);

    player.updateSettings({
      streaming: {
        abr: { autoSwitchBitrate: { video: true } },
        scheduleWhilePaused: false,
      },
    });

    player.on(dashjs.MediaPlayer.events.PLAYBACK_PLAYING, function () {
      if (!startupDone) {
        startupSec = Math.max((performance.now() - startupBegin) / 1000.0, 0);
      }
      if (waitingSince !== null) {
        rebufferSecCurrent += Math.max((performance.now() - waitingSince) / 1000.0, 0);
        waitingSince = null;
      }
    });

    player.on(dashjs.MediaPlayer.events.PLAYBACK_WAITING, function () {
      if (waitingSince === null) {
        waitingSince = performance.now();
        rebufferCountCurrent += 1;
      }
    });

    player.on(dashjs.MediaPlayer.events.FRAGMENT_LOADING_COMPLETED, function (e) {
      try {
        if (!e || !e.request || e.request.mediaType !== "video") return;
        const req = e.request;
        if (req.type && req.type !== "MediaSegment") return;

        let segIdx = segmentIdx;
        const idx = Number(req.index);
        if (Number.isFinite(idx) && idx >= 0) {
          if (idx <= lastSentSegmentIdx) return;
          segIdx = idx;
        }

        let downloadSec = 0.0;
        if (req.trequest && req.tfinish) {
          const t0 = new Date(req.trequest).getTime();
          const t1 = new Date(req.tfinish).getTime();
          if (Number.isFinite(t0) && Number.isFinite(t1) && t1 >= t0) {
            downloadSec = (t1 - t0) / 1000.0;
          }
        }
        if (downloadSec <= 0.0) {
          downloadSec = (video.buffered.length > 0 ? 0.1 : 0.5);
        }
        sendSegmentEvent(downloadSec, segIdx);
      } catch (err) {
        log(`segment parse error: ${err.message}`, "warn");
      }
    });

    player.on(dashjs.MediaPlayer.events.ERROR, function (e) {
      log(`dash error: ${JSON.stringify(e)}`, "warn");
    });

    log(`player ready, mpd=${mpdUrl}`);
  }

  init();
})();
