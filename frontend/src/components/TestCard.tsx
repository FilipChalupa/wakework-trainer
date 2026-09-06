import { useCallback, useEffect, useRef, useState } from "react";
import { Alert, Box, Button, Card, CardContent, CardHeader, Chip, LinearProgress, MenuItem, Slider, Stack, Table, TableBody, TableCell, TableHead, TableRow, TextField, Typography, useTheme } from "@mui/material";
import HearingIcon from "@mui/icons-material/Hearing";
import StopIcon from "@mui/icons-material/Stop";
import FactCheckIcon from "@mui/icons-material/FactCheck";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import { api, type Evaluation, type Job } from "../api";
import { errorText, useI18n } from "../i18n";
import { Recorder } from "../lib/recorder";

type Props = { jobs: Job[]; wakeWord: string; disabled: boolean; onError: (message: string) => void };

type FrameMsg = { type: "frames"; t: number; frames: { p: number; avg: number; detected: boolean }[]; detections: number; max: number };

const HISTORY = 240;

export function TestCard({ jobs, wakeWord, disabled, onError }: Props) {
  const { t } = useI18n();
  const theme = useTheme();
  const models = jobs.filter((j) => j.status === "done" && j.model_url);
  const [jobId, setJobId] = useState<string>("");
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
    try {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const socket = new WebSocket(`${proto}://${location.host}/api/test/ws?job_id=${encodeURIComponent(jobId)}&cutoff=${cutoff}&window=${window}`);
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;
      socket.onmessage = (event) => {
        const msg = JSON.parse(event.data as string) as FrameMsg | { type: "ready" } | { type: "error"; message: string };
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
                    {m.slug}.tflite · {new Date(m.created_at).toLocaleString()}
                  </MenuItem>
                ))}
              </TextField>
              <Box sx={{ flex: 1, px: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  {t("test.cutoff")}: {cutoff.toFixed(2)}
                </Typography>
                <Slider size="small" min={0.3} max={0.99} step={0.01} value={cutoff} onChange={(_, v) => setCutoff(v as number)} disabled={active} />
              </Box>
              <TextField select size="small" label={t("test.window")} value={window} onChange={(e) => setWindow(Number(e.target.value))} sx={{ minWidth: 120 }} disabled={active}>
                {[1, 3, 5, 7, 10].map((n) => (
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
                          <TableRow key={item.id} hover>
                            <TableCell>
                              <Typography variant="body2" component="a" href={item.url} target="_blank" rel="noreferrer" sx={{ color: "inherit" }}>
                                {item.id}
                              </Typography>
                            </TableCell>
                            <TableCell>{t(item.kind === "positive" ? "test.kind.positive" : "test.kind.negative")}</TableCell>
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
