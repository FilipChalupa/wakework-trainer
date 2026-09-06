import { useCallback, useEffect, useRef, useState } from "react";
import { Alert, Box, Button, Card, CardContent, CardHeader, Chip, FormControlLabel, IconButton, LinearProgress, MenuItem, Slider, Stack, Switch, Table, TableBody, TableCell, TableHead, TableRow, TextField, Tooltip, Typography, useTheme } from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import DeleteIcon from "@mui/icons-material/Delete";
import AddCircleOutlineIcon from "@mui/icons-material/AddCircleOutline";
import FlagIcon from "@mui/icons-material/Flag";
import HearingIcon from "@mui/icons-material/Hearing";
import StopIcon from "@mui/icons-material/Stop";
import FactCheckIcon from "@mui/icons-material/FactCheck";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import { api, type Evaluation, type Job, type MonitorItem } from "../api";
import { errorText, useI18n, type TKey } from "../i18n";
import { Recorder } from "../lib/recorder";

type Props = { jobs: Job[]; wakeWord: string; disabled: boolean; onError: (message: string) => void; onInfo?: (message: string) => void };

type FrameMsg = { type: "frames"; t: number; frames: { p: number; avg: number; detected: boolean }[]; detections: number; max: number };
type DetectionMsg = { type: "detection"; item: MonitorItem; t: number };

const HISTORY = 240;

export function TestCard({ jobs, wakeWord, disabled, onError, onInfo }: Props) {
  const { t } = useI18n();
  const theme = useTheme();
  const models = jobs.filter((j) => j.status === "done" && j.model_url);
  const [jobId, setJobId] = useState<string>("");
  const currentTarget = models.find((m) => m.job_id === jobId)?.target ?? "esphome";
  const [cutoff, setCutoff] = useState(0.97);
  const [window, setWindow] = useState(5);
  const [listening, setListening] = useState<"off" | "connecting" | "on">("off");
  const [prob, setProb] = useState(0);
  const [avg, setAvg] = useState(0);
  const [max, setMax] = useState(0);
  const [level, setLevel] = useState(0);
  const [detections, setDetections] = useState<number[]>([]);
  const [flash, setFlash] = useState(false);
  const [history, setHistory] = useState<number[]>([]);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [evaluating, setEvaluating] = useState(false);
  const [monitor, setMonitor] = useState(false);
  const [saved, setSaved] = useState<MonitorItem[]>([]);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [flagged, setFlagged] = useState<number | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const loadSaved = useCallback(() => api.listMonitor().then((r) => setSaved(r.items)).catch(() => undefined), []);
  useEffect(() => {
    loadSaved();
  }, [loadSaved, jobId]);
  useEffect(() => {
    if (!startedAt) return;
    const timer = setInterval(() => setElapsed((Date.now() - startedAt) / 60000), 5000);
    return () => clearInterval(timer);
  }, [startedAt]);

  const playSaved = (item: MonitorItem) => {
    audioRef.current?.pause();
    const audio = new Audio(item.url);
    audioRef.current = audio;
    audio.play().catch(() => undefined);
  };
  const toNegative = async (item: MonitorItem) => {
    try {
      await api.monitorToNegative(item.id);
      await loadSaved();
    } catch (e) {
      onError(errorText(t, e));
    }
  };
  const removeSaved = async (item: MonitorItem) => {
    try {
      await api.deleteMonitor(item.id);
      await loadSaved();
    } catch (e) {
      onError(errorText(t, e));
    }
  };
  const clearSaved = async () => {
    try {
      await api.clearMonitor();
      await loadSaved();
    } catch (e) {
      onError(errorText(t, e));
    }
  };
  const retrainWithMonitor = async () => {
    try {
      const res = await api.monitorAdoptAll();
      await api.startTraining({}, t("monitor.retrainLabel"));
      await loadSaved();
      onInfo?.(t("monitor.retrained", { n: res.moved }));
    } catch (e) {
      onError(errorText(t, e));
    }
  };

  const flagOutliers = async () => {
    if (!evaluation) return;
    const outliers = evaluation.items.filter((i) => i.outlier);
    try {
      for (const item of outliers) await api.setReview(item.kind, item.id, true);
      setFlagged(outliers.length);
    } catch (e) {
      onError(errorText(t, e));
    }
  };
  const recorderRef = useRef(new Recorder());
  const socketRef = useRef<WebSocket | null>(null);
  const stopStreamRef = useRef<(() => void) | null>(null);
  const flashTimer = useRef<number | null>(null);

  useEffect(() => {
    if (!jobId && models.length) setJobId(models[0].job_id);
    if (jobId && !models.some((m) => m.job_id === jobId)) setJobId(models[0]?.job_id ?? "");
  }, [models, jobId]);

  useEffect(() => {
    if (!jobId) return;
    api
      .testInfo(jobId)
      .then((info) => {
        setCutoff(info.probability_cutoff);
        setWindow(info.sliding_window_size);
      })
      .catch(() => undefined);
    setEvaluation(null);
  }, [jobId]);

  const stop = useCallback(() => {
    stopStreamRef.current?.();
    stopStreamRef.current = null;
    socketRef.current?.close();
    socketRef.current = null;
    setListening("off");
    setLevel(0);
  }, []);

  useEffect(() => {
    const recorder = recorderRef.current;
    return () => {
      stop();
      recorder.close();
    };
  }, [stop]);

  const start = async () => {
    if (!jobId || listening !== "off") return;
    setListening("connecting");
    setHistory([]);
    setMax(0);
    setDetections([]);
    setStartedAt(Date.now());
    setElapsed(0);
    try {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const socket = new WebSocket(`${proto}://${location.host}/api/test/ws?job_id=${encodeURIComponent(jobId)}&cutoff=${cutoff}&window=${window}${monitor ? "&save=1" : ""}`);
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;
      socket.onmessage = (event) => {
        const msg = JSON.parse(event.data as string) as FrameMsg | DetectionMsg | { type: "ready" } | { type: "error"; message: string };
        if (msg.type === "detection") {
          setSaved((prev) => [msg.item, ...prev].slice(0, 200));
          return;
        }
        if (msg.type === "error") {
          onError(t("test.wsError", { err: msg.message }));
          stop();
          return;
        }
        if (msg.type === "ready") {
          setListening("on");
          return;
        }
        if (msg.type === "frames" && msg.frames.length) {
          const last = msg.frames[msg.frames.length - 1];
          setProb(last.p);
          setAvg(last.avg);
          setMax(msg.max);
          setHistory((h) => {
            const next = [...h, ...msg.frames.map((f) => f.avg)];
            return next.length > HISTORY ? next.slice(next.length - HISTORY) : next;
          });
          if (msg.frames.some((f) => f.detected)) {
            setDetections((d) => [Date.now(), ...d].slice(0, 10));
            setFlash(true);
            if (flashTimer.current) clearTimeout(flashTimer.current);
            flashTimer.current = window_setTimeout(() => setFlash(false), 900);
          }
        }
      };
      socket.onerror = () => {
        onError(t("test.wsError", { err: "WebSocket" }));
        stop();
      };
      socket.onclose = () => {
        if (socketRef.current === socket) stop();
      };
      await new Promise<void>((resolve, reject) => {
        socket.onopen = () => resolve();
        const prevError = socket.onerror;
        socket.onerror = (e) => {
          prevError?.call(socket, e);
          reject(new Error("WebSocket"));
        };
      });
      stopStreamRef.current = await recorderRef.current.startStream(
        (pcm) => {
          if (socket.readyState === WebSocket.OPEN) socket.send(pcm.buffer);
        },
        (rms) => setLevel(Math.min(1, rms * 6)),
      );
    } catch (e) {
      onError(errorText(t, e));
      stop();
    }
  };

  const resetMax = () => {
    setMax(0);
    setDetections([]);
    setHistory([]);
    socketRef.current?.send("reset");
  };

  const evaluate = async () => {
    if (!jobId) return;
    setEvaluating(true);
    try {
      setEvaluation(await api.evaluateJob(jobId, cutoff, window));
      setFlagged(null);
    } catch (e) {
      onError(errorText(t, e));
    } finally {
      setEvaluating(false);
    }
  };

  const active = listening !== "off";

  return (
    <Card sx={{ transition: "box-shadow 200ms", boxShadow: flash ? `0 0 0 4px ${theme.palette.success.main}` : undefined }}>
      <CardHeader avatar={<HearingIcon color="primary" />} title={t("test.title")} subheader={t("test.subtitle")} />
      <CardContent>
        {models.length === 0 ? (
          <Typography color="text.secondary">{t("test.noModel")}</Typography>
        ) : (
          <Stack spacing={2}>
            <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ md: "center" }}>
              <TextField select size="small" label={t("test.model")} value={jobId} onChange={(e) => setJobId(e.target.value)} sx={{ minWidth: 260 }} disabled={active}>
                {models.map((m) => (
                  <MenuItem key={m.job_id} value={m.job_id}>
                    {m.slug}.tflite · {m.target === "wyoming" ? "Wyoming" : "ESPHome"} · {new Date(m.created_at).toLocaleString()}
                  </MenuItem>
                ))}
              </TextField>
              <Box sx={{ flex: 1, px: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  {t("test.cutoff")}: {cutoff.toFixed(2)}
                </Typography>
                <Slider size="small" min={0.3} max={0.99} step={0.01} value={cutoff} onChange={(_, v) => setCutoff(v as number)} disabled={active} />
              </Box>
              <FormControlLabel control={<Switch checked={monitor} onChange={(e) => setMonitor(e.target.checked)} disabled={active} />} label={<Typography variant="body2">{t("monitor.switch")}</Typography>} />
              <TextField select size="small" label={currentTarget === "wyoming" ? t("test.trigger") : t("test.window")} value={window} onChange={(e) => setWindow(Number(e.target.value))} sx={{ minWidth: 170 }} disabled={active}>
                {(currentTarget === "wyoming" ? [1, 2, 3] : [1, 3, 5, 7, 10]).map((n) => (
                  <MenuItem key={n} value={n}>
                    {n}
                  </MenuItem>
                ))}
              </TextField>
            </Stack>

            <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" useFlexGap>
              {!active ? (
                <Button variant="contained" size="large" startIcon={<HearingIcon />} onClick={start} disabled={disabled || !jobId}>
                  {t("test.start")}
                </Button>
              ) : (
                <Button variant="outlined" color="error" size="large" startIcon={<StopIcon />} onClick={stop}>
                  {t("test.stop")}
                </Button>
              )}
              <Typography variant="subtitle1" sx={{ flex: 1 }}>
                {listening === "connecting" && t("test.connecting")}
                {listening === "on" && t("test.listening", { word: wakeWord })}
              </Typography>
              <Chip color={flash ? "success" : "default"} label={flash ? t("test.detected") : t("test.detections", { n: detections.length })} sx={{ fontWeight: 700, transform: flash ? "scale(1.15)" : "none", transition: "transform 150ms" }} />
              <Button size="small" startIcon={<RestartAltIcon />} onClick={resetMax} disabled={!active}>
                {t("test.reset")}
              </Button>
            </Stack>

            <Box>
              <LinearProgress variant="determinate" value={level * 100} color="success" sx={{ height: 6, borderRadius: 3, mb: 1, opacity: active ? 1 : 0.3 }} />
              <Typography variant="caption" color="text.secondary">
                {t("test.prob")}: {(prob * 100).toFixed(0)} %
              </Typography>
              <LinearProgress variant="determinate" value={prob * 100} sx={{ height: 10, borderRadius: 5, mb: 1 }} />
              <Typography variant="caption" color="text.secondary">
                {t("test.avg")}: {(avg * 100).toFixed(0)} % · {t("test.max", { v: (max * 100).toFixed(0) + " %" })}
              </Typography>
              <Box sx={{ position: "relative" }}>
                <LinearProgress variant="determinate" value={avg * 100} color={avg >= cutoff ? "success" : "primary"} sx={{ height: 14, borderRadius: 7 }} />
                <Box sx={{ position: "absolute", top: -3, bottom: -3, left: `${cutoff * 100}%`, width: 2, bgcolor: "error.main" }} />
              </Box>
              <Sparkline values={history} cutoff={cutoff} color={theme.palette.primary.main} cutoffColor={theme.palette.error.main} />
            </Box>

            {detections.length > 0 && (
              <Typography variant="body2" color="text.secondary">
                {t("test.lastDetections")}: {detections.map((d) => new Date(d).toLocaleTimeString()).join(", ")}
              </Typography>
            )}
            {active && startedAt && (
              <Typography variant="body2" color="text.secondary">
                {t("monitor.stats", { minutes: elapsed.toFixed(1), n: detections.length, rate: elapsed > 0.05 ? (detections.length / (elapsed / 60)).toFixed(1) : "–" })}
              </Typography>
            )}

            <Box sx={{ p: 1.5, border: 1, borderColor: "divider", borderRadius: 2 }}>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                <Typography variant="subtitle2" sx={{ flex: 1 }}>
                  {t("monitor.title")}
                </Typography>
                {saved.length > 0 && (
                  <>
                    <Button size="small" variant="contained" onClick={retrainWithMonitor} disabled={disabled || active}>
                      {t("monitor.retrain")}
                    </Button>
                    <Button size="small" color="error" onClick={clearSaved}>
                      {t("monitor.clear")}
                    </Button>
                  </>
                )}
              </Stack>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                {t("monitor.help")}
              </Typography>
              {saved.length === 0 ? (
                <Typography variant="caption" color="text.secondary">
                  {t("monitor.empty")}
                </Typography>
              ) : (
                <Stack spacing={0.5} sx={{ maxHeight: 240, overflow: "auto" }}>
                  {saved.map((item) => (
                    <Stack key={item.id} direction="row" spacing={1} alignItems="center">
                      <IconButton size="small" onClick={() => playSaved(item)}>
                        <PlayArrowIcon />
                      </IconButton>
                      <Typography variant="body2" sx={{ flex: 1 }} noWrap>
                        {new Date(item.created).toLocaleString()} · {item.duration.toFixed(1)} s
                      </Typography>
                      <Chip size="small" variant="outlined" label={t("monitor.source.browser")} />
                      <Tooltip title={t("monitor.toNegative")}>
                        <IconButton size="small" color="primary" onClick={() => toNegative(item)}>
                          <AddCircleOutlineIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title={t("monitor.delete")}>
                        <IconButton size="small" onClick={() => removeSaved(item)}>
                          <DeleteIcon />
                        </IconButton>
                      </Tooltip>
                    </Stack>
                  ))}
                </Stack>
              )}
            </Box>

            <Alert severity="info" variant="outlined">
              {t("test.hint")}
            </Alert>

            <Stack direction="row" spacing={2} alignItems="center">
              <Button variant="outlined" startIcon={<FactCheckIcon />} onClick={evaluate} disabled={disabled || evaluating || !jobId || active}>
                {evaluating ? t("test.evaluating") : t("test.evaluate")}
              </Button>
              {evaluating && <LinearProgress sx={{ flex: 1 }} />}
            </Stack>

            {evaluation && (
              <Box>
                <Typography variant="body2" gutterBottom>
                  {t("test.evalSummary", {
                    pd: evaluation.summary.positive_detected,
                    pt: evaluation.summary.positive_total,
                    nt: evaluation.summary.negative_triggered,
                    ntot: evaluation.summary.negative_total,
                    cutoff: evaluation.cutoff.toFixed(2),
                  })}
                </Typography>
                {evaluation.summary.outliers > 0 && (
                  <Alert
                    severity="warning"
                    variant="outlined"
                    sx={{ mb: 1 }}
                    action={
                      <Button color="inherit" size="small" startIcon={<FlagIcon />} onClick={flagOutliers} disabled={flagged !== null}>
                        {t("eval.flag")}
                      </Button>
                    }
                  >
                    {t("eval.outliers", { n: evaluation.summary.outliers })} – {t("eval.outliersHelp", { median: Math.round(evaluation.summary.median_positive * 100) })}
                    {flagged !== null && ` ${t("eval.flagged", { n: flagged })}`}
                  </Alert>
                )}
                {Object.keys(evaluation.summary.by_tag ?? {}).length > 1 && (
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mb: 1 }} alignItems="center">
                    <Typography variant="caption" color="text.secondary">
                      {t("test.byTag")}:
                    </Typography>
                    {Object.entries(evaluation.summary.by_tag).map(([tg, v]) => (
                      <Chip key={tg} size="small" variant="outlined" color={v.detected === v.total ? "success" : v.detected === 0 ? "error" : "warning"} label={`${t(`tag.${tg}` as TKey)}: ${v.detected} / ${v.total}`} />
                    ))}
                  </Stack>
                )}
                <Box sx={{ maxHeight: 320, overflow: "auto" }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>{t("test.t.recording")}</TableCell>
                        <TableCell>{t("test.t.kind")}</TableCell>
                        <TableCell>{t("test.t.max")}</TableCell>
                        <TableCell>{t("test.t.result")}</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {evaluation.items.map((item) => {
                        const ok = item.kind === "positive" ? item.detections > 0 : item.detections === 0;
                        const label = item.kind === "positive" ? (ok ? t("test.r.ok") : t("test.r.missed")) : ok ? t("test.r.ok") : t("test.r.false");
                        const p = item.max_probability ?? 0;
                        return (
                          <TableRow key={item.id} hover selected={item.outlier}>
                            <TableCell>
                              <Typography variant="body2" component="a" href={item.url} target="_blank" rel="noreferrer" sx={{ color: "inherit" }}>
                                {item.id}
                              </Typography>
                              {item.outlier && <Chip size="small" color="warning" label={t("rec.review")} sx={{ ml: 1, height: 18, fontSize: 11 }} />}
                            </TableCell>
                            <TableCell>
                              {t(item.kind === "positive" ? "test.kind.positive" : "test.kind.negative")}
                              {item.tag && item.tag !== "normal" ? ` · ${t(`tag.${item.tag}` as TKey)}` : ""}
                            </TableCell>
                            <TableCell sx={{ minWidth: 160 }}>
                              <Stack direction="row" spacing={1} alignItems="center">
                                <LinearProgress variant="determinate" value={p * 100} sx={{ flex: 1, height: 8, borderRadius: 4 }} color={p >= evaluation.cutoff ? "success" : "inherit"} />
                                <Typography variant="caption">{(p * 100).toFixed(0)} %</Typography>
                              </Stack>
                            </TableCell>
                            <TableCell>
                              <Chip size="small" label={item.error ?? label} color={item.error ? "default" : ok ? "success" : "error"} variant={ok ? "outlined" : "filled"} />
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </Box>
              </Box>
            )}
          </Stack>
        )}
      </CardContent>
    </Card>
  );
}

function window_setTimeout(fn: () => void, ms: number): number {
  return globalThis.setTimeout(fn, ms) as unknown as number;
}

function Sparkline({ values, cutoff, color, cutoffColor }: { values: number[]; cutoff: number; color: string; cutoffColor: string }) {
  const w = 600;
  const h = 60;
  const points = values.map((v, i) => `${(i / Math.max(1, HISTORY - 1)) * w},${h - v * h}`).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ width: "100%", height: 60, display: "block", marginTop: 8 }}>
      <line x1={0} x2={w} y1={h - cutoff * h} y2={h - cutoff * h} stroke={cutoffColor} strokeDasharray="4 4" strokeWidth={1} />
      {values.length > 1 && <polyline points={points} fill="none" stroke={color} strokeWidth={2} />}
    </svg>
  );
}
