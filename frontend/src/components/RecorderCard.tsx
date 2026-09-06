import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Checkbox,
  Chip,
  CircularProgress,
  FormControlLabel,
  IconButton,
  LinearProgress,
  MenuItem,
  Snackbar,
  Stack,
  Switch,
  Tab,
  Tabs,
  TextField,
  Tooltip,
  Typography,
  useTheme,
} from "@mui/material";
import MicIcon from "@mui/icons-material/Mic";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import DeleteIcon from "@mui/icons-material/Delete";
import GraphicEqIcon from "@mui/icons-material/GraphicEq";
import PlaylistPlayIcon from "@mui/icons-material/PlaylistPlay";
import RepeatIcon from "@mui/icons-material/Repeat";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";
import { api, TAGS, type Recording, type RecordingsClient, type Tag } from "../api";
import { errorText, useI18n, type TKey } from "../i18n";
import { Recorder, waveformPeaks } from "../lib/recorder";

type Kind = "positive" | "negative";
const RECOMMENDED = 30;

type Props = {
  wakeWord: string;
  /** hard limit for one recording; it normally stops by itself after the word */
  maxSeconds: number;
  disabled: boolean;
  onCountsChange: (counts: Record<Kind, number>) => void;
  onError: (message: string) => void;
  /** API used for recordings – the contributor page passes a token based client. */
  client?: RecordingsClient;
  /** Contributor mode: no import, no bulk selection, simplified header. */
  compact?: boolean;
  title?: string;
};

type Phase = "idle" | "prepare" | "countdown" | "recording" | "uploading";

export function RecorderCard({ wakeWord, maxSeconds, disabled, onCountsChange, onError, client = api, compact = false, title }: Props) {
  const theme = useTheme();
  const { t } = useI18n();
  const fail = useCallback((e: unknown) => onError(e instanceof Error && e.message === "mic_unsupported" ? t("rec.micUnsupported") : errorText(t, e)), [onError, t]);
  const VARIATION_HINTS = t("rec.hints").split("|");
  const [kind, setKind] = useState<Kind>("positive");
  const [items, setItems] = useState<Record<Kind, Recording[]>>({ positive: [], negative: [] });
  const [phase, setPhase] = useState<Phase>("idle");
  const [countdown, setCountdown] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);
  const [lastPeaks, setLastPeaks] = useState<number[] | null>(null);
  const [playing, setPlaying] = useState<{ id: string; progress: number } | null>(null);
  const [playAll, setPlayAll] = useState(false);
  const [autoPlay, setAutoPlay] = useState(true);
  const [seriesSize, setSeriesSize] = useState(10);
  const [series, setSeries] = useState<{ done: number; total: number } | null>(null);
  const [devices, setDevices] = useState<{ deviceId: string; label: string }[]>([]);
  const [deviceId, setDeviceId] = useState<string>("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [undo, setUndo] = useState<{ kind: Kind; ids: string[] } | null>(null);
  const [importing, setImporting] = useState<{ done: number; total: number } | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [tag, setTag] = useState<Tag>("normal");

  const recorderRef = useRef(new Recorder());
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const stopSeriesRef = useRef(false);
  const playAllRef = useRef(false);
  const busyRef = useRef(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const stopRecordingRef = useRef<(() => void) | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [pos, neg] = await Promise.all([client.listRecordings("positive"), client.listRecordings("negative")]);
      setItems({ positive: pos.items, negative: neg.items });
      onCountsChange({ positive: pos.items.length, negative: neg.items.length });
    } catch (e) {
      fail(e);
    }
  }, [onCountsChange, fail, client]);

  useEffect(() => {
    refresh();
    Recorder.listDevices().then(setDevices).catch(() => undefined);
    const recorder = recorderRef.current;
    return () => recorder.close();
  }, [refresh]);

  // ----- playback ---------------------------------------------------------
  const stopPlayback = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.onended = null;
      audioRef.current.ontimeupdate = null;
      audioRef.current.pause();
      audioRef.current = null;
    }
    setPlaying(null);
  }, []);

  const playOne = useCallback(
    (rec: Recording) =>
      new Promise<void>((resolve) => {
        stopPlayback();
        const audio = new Audio(rec.url);
        audioRef.current = audio;
        setPlaying({ id: rec.id, progress: 0 });
        audio.ontimeupdate = () => setPlaying({ id: rec.id, progress: audio.duration ? audio.currentTime / audio.duration : 0 });
        audio.onended = () => {
          setPlaying(null);
          resolve();
        };
        audio.onerror = () => {
          setPlaying(null);
          resolve();
        };
        audio.play().catch(() => resolve());
      }),
    [stopPlayback],
  );

  const togglePlay = (rec: Recording) => {
    if (playing?.id === rec.id) {
      stopPlayback();
      playAllRef.current = false;
      setPlayAll(false);
      return;
    }
    playOne(rec);
  };

  const playEverything = async () => {
    if (playAll) {
      playAllRef.current = false;
      setPlayAll(false);
      stopPlayback();
      return;
    }
    setPlayAll(true);
    playAllRef.current = true;
    for (const rec of [...items[kind]]) {
      if (!playAllRef.current) break;
      await playOne(rec);
    }
    playAllRef.current = false;
    setPlayAll(false);
  };

  // ----- recording ----------------------------------------------------------
  const captureOne = useCallback(
    async (targetKind: Kind, doCountdown: boolean) => {
      setLastPeaks(null);
      setPhase("prepare");
      await recorderRef.current.init(deviceId || undefined);
      if (devices.length === 0) Recorder.listDevices().then(setDevices).catch(() => undefined);
      if (doCountdown) {
        setPhase("countdown");
        for (let n = 3; n > 0; n--) {
          if (stopSeriesRef.current) return null;
          setCountdown(n);
          await new Promise((r) => setTimeout(r, 550));
        }
      } else {
        await new Promise((r) => setTimeout(r, 300));
      }
      setPhase("recording");
      setElapsed(0);
      const { wav, samples } = await recorderRef.current.record(
        maxSeconds,
        ({ rms, elapsed }) => {
          setLevel(Math.min(1, rms * 6));
          setElapsed(elapsed);
        },
        { onStart: (stop) => (stopRecordingRef.current = stop) },
      );
      stopRecordingRef.current = null;
      setLevel(0);
      setLastPeaks(waveformPeaks(samples));
      setPhase("uploading");
      const saved = await client.uploadRecording(targetKind, wav, "sample.wav", tag === "normal" ? null : tag);
      await refresh();
      return saved;
    },
    [deviceId, devices.length, maxSeconds, refresh, client, tag],
  );

  const recordSingle = useCallback(async () => {
    if (busyRef.current || disabled) return;
    busyRef.current = true;
    stopPlayback();
    try {
      const saved = await captureOne(kind, false);
      setPhase("idle");
      if (saved && autoPlay) await playOne(saved);
    } catch (e) {
      fail(e);
    } finally {
      setPhase("idle");
      busyRef.current = false;
    }
  }, [autoPlay, captureOne, disabled, kind, fail, playOne, stopPlayback]);

  const recordSeries = async () => {
    if (busyRef.current || disabled) return;
    busyRef.current = true;
    stopSeriesRef.current = false;
    stopPlayback();
    let wakeLock: { release: () => Promise<void> } | null = null;
    try {
      wakeLock = (await (navigator as Navigator & { wakeLock?: { request: (t: string) => Promise<{ release: () => Promise<void> }> } }).wakeLock?.request("screen")) ?? null;
    } catch {
      wakeLock = null;
    }
    const total = Math.max(1, seriesSize);
    setSeries({ done: 0, total });
    try {
      for (let i = 0; i < total; i++) {
        if (stopSeriesRef.current) break;
        const saved = await captureOne(kind, true);
        if (!saved) break;
        setSeries({ done: i + 1, total });
        setPhase("idle");
        if (autoPlay) await playOne(saved);
        else await new Promise((r) => setTimeout(r, 400));
      }
    } catch (e) {
      fail(e);
    } finally {
      setSeries(null);
      setPhase("idle");
      busyRef.current = false;
      wakeLock?.release().catch(() => undefined);
    }
  };

  const stopSeries = () => {
    stopSeriesRef.current = true;
  };

  // keyboard shortcuts: Space / R = record, Escape = stop
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT" || target.isContentEditable)) return;
      if (e.code === "Space" || e.key.toLowerCase() === "r") {
        e.preventDefault();
        recordSingle();
      } else if (e.key === "Escape") {
        stopRecordingRef.current?.();
        stopSeriesRef.current = true;
        playAllRef.current = false;
        setPlayAll(false);
        stopPlayback();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [recordSingle, stopPlayback]);

  // ----- delete / restore ---------------------------------------------------
  const removeMany = async (targetKind: Kind, ids: string[]) => {
    if (!ids.length) return;
    try {
      if (playing && ids.includes(playing.id)) stopPlayback();
      for (const id of ids) await client.deleteRecording(targetKind, id);
      setSelected(new Set());
      setUndo({ kind: targetKind, ids });
      await refresh();
    } catch (e) {
      fail(e);
    }
  };

  const clearReview = async (rec: Recording) => {
    try {
      await api.setReview(rec.kind, rec.id, false);
      await refresh();
    } catch (e) {
      fail(e);
    }
  };

  const restore = async () => {
    if (!undo) return;
    try {
      for (const id of undo.ids) await client.restoreRecording(undo.kind, id);
      setUndo(null);
      await refresh();
    } catch (e) {
      fail(e);
    }
  };

  // ----- import -------------------------------------------------------------
  const importFiles = async (files: FileList | File[]) => {
    const list = Array.from(files).filter((f) => f.type.startsWith("audio/") || /\.(wav|mp3|ogg|webm|m4a|flac)$/i.test(f.name));
    if (!list.length) return;
    setImporting({ done: 0, total: list.length });
    let failed = 0;
    for (let i = 0; i < list.length; i++) {
      try {
        await client.uploadRecording(kind, list[i], list[i].name);
      } catch {
        failed += 1;
      }
      setImporting({ done: i + 1, total: list.length });
    }
    setImporting(null);
    if (failed) onError(t("rec.importFailed", { n: failed }));
    await refresh();
  };

  // ----- derived --------------------------------------------------------------
  const list = items[kind];
  const positiveCount = items.positive.length;
  const progress = Math.min(100, (positiveCount / RECOMMENDED) * 100);
  const problems = useMemo(() => list.filter((r) => r.quality.issues.length > 0).length, [list]);
  const hint = VARIATION_HINTS[(kind === "positive" ? positiveCount : items.negative.length) % VARIATION_HINTS.length];
  const busy = phase !== "idle";

  const toggleSelected = (id: string) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelected(next);
  };

  return (
    <Card
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        if (!disabled && !compact) importFiles(e.dataTransfer.files);
      }}
      sx={{ outline: dragOver ? `2px dashed ${theme.palette.primary.main}` : "none" }}
    >
      <CardHeader
        avatar={<GraphicEqIcon color="primary" />}
        title={title ?? t("rec.title")}
        subheader={t("rec.subtitle")}
        action={
          <Chip
            color={positiveCount >= 20 ? "success" : "default"}
            label={t("rec.chip", { count: positiveCount, recommended: RECOMMENDED })}
            variant={positiveCount >= 20 ? "filled" : "outlined"}
          />
        }
      />
      <CardContent>
        <Stack spacing={2}>
          <Box>
            <LinearProgress variant="determinate" value={progress} sx={{ height: 8, borderRadius: 4 }} />
            <Typography variant="caption" color="text.secondary">
              {t("rec.recommended")}
            </Typography>
          </Box>

          <Tabs value={kind} onChange={(_, v) => setKind(v)} variant="fullWidth">
            <Tab value="positive" label={t("rec.tabPositive", { word: wakeWord, n: items.positive.length })} />
            <Tab value="negative" label={t("rec.tabNegative", { n: items.negative.length })} />
          </Tabs>

          {kind === "negative" && (
            <Alert severity="info" variant="outlined">
              {t("rec.negativeInfo")}
            </Alert>
          )}

          <Stack direction={{ xs: "column", sm: "row" }} spacing={3} alignItems="center">
            <Box sx={{ position: "relative", display: "inline-flex", "& .MuiIconButton-root": { width: { xs: 160, sm: 120 }, height: { xs: 160, sm: 120 } }, "& > .MuiCircularProgress-root": { width: { xs: "160px !important", sm: "120px !important" }, height: { xs: "160px !important", sm: "120px !important" } } }}>
              <CircularProgress
                variant="determinate"
                value={phase === "recording" ? (elapsed / maxSeconds) * 100 : 0}
                size={120}
                thickness={3}
                sx={{ color: phase === "recording" ? theme.palette.error.main : theme.palette.divider, position: "absolute", inset: 0 }}
              />
              <IconButton
                onClick={phase === "recording" ? () => stopRecordingRef.current?.() : series ? stopSeries : recordSingle}
                disabled={disabled || (busy && !series && phase !== "recording")}
                sx={{
                  width: 120,
                  height: 120,
                  bgcolor: phase === "recording" ? "error.main" : "primary.main",
                  color: phase === "recording" ? "error.contrastText" : "primary.contrastText",
                  "&:hover": { bgcolor: phase === "recording" ? "error.dark" : "primary.dark" },
                  "&.Mui-disabled": { bgcolor: "action.disabledBackground" },
                  transform: phase === "recording" ? `scale(${1 + level * 0.08})` : "none",
                  transition: "transform 80ms linear",
                  fontSize: 44,
                  fontWeight: 700,
                }}
              >
                {phase === "countdown" ? countdown : phase === "uploading" ? <CircularProgress size={36} color="inherit" /> : phase === "recording" || series ? <StopIcon sx={{ fontSize: 48 }} /> : <MicIcon sx={{ fontSize: 52 }} />}
              </IconButton>
            </Box>
            <Box sx={{ flex: 1, width: "100%" }}>
              <Typography variant="h6">
                {phase === "idle" && (kind === "positive" ? t("rec.idlePositive", { word: wakeWord, hint: tag === "normal" ? hint : t(`tag.hint.${tag}` as TKey) }) : t("rec.idleNegative"))}
                {phase === "prepare" && t("rec.prepare")}
                {phase === "countdown" && t("rec.countdown", { n: countdown })}
                {phase === "recording" && t("rec.recording", { s: elapsed.toFixed(1) })}
                {phase === "uploading" && t("rec.uploading")}
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {series ? t("rec.seriesStatus", { done: series.done, total: series.total }) : t("rec.instructions", { s: maxSeconds.toFixed(0) })}
              </Typography>
              <LinearProgress variant="determinate" value={level * 100} color={level > 0.9 ? "error" : "success"} sx={{ height: 10, borderRadius: 5, mb: 1 }} />
              {lastPeaks && <Waveform peaks={lastPeaks} color={theme.palette.primary.main} height={40} />}
            </Box>
          </Stack>

          {kind === "positive" && (
            <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
              <Typography variant="caption" color="text.secondary">
                {t("tag.label")}:
              </Typography>
              {TAGS.map((tg) => (
                <Tooltip key={tg} title={t(`tag.hint.${tg}` as TKey)}>
                  <Chip size="small" label={t(`tag.${tg}` as TKey)} color={tag === tg ? "primary" : "default"} variant={tag === tg ? "filled" : "outlined"} onClick={() => setTag(tg)} disabled={busy} />
                </Tooltip>
              ))}
            </Stack>
          )}

          <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ md: "center" }} flexWrap="wrap" useFlexGap>
            <Stack direction="row" spacing={1} alignItems="center">
              <TextField
                select
                size="small"
                label={t("rec.series")}
                value={seriesSize}
                onChange={(e) => setSeriesSize(Number(e.target.value))}
                sx={{ minWidth: 110 }}
                disabled={busy}
              >
                {[5, 10, 20, 30].map((n) => (
                  <MenuItem key={n} value={n}>
                    {t("rec.seriesN", { n })}
                  </MenuItem>
                ))}
              </TextField>
              <Button variant="outlined" startIcon={<RepeatIcon />} onClick={recordSeries} disabled={disabled || busy}>
                {t("rec.seriesButton")}
              </Button>
            </Stack>
            <FormControlLabel control={<Switch checked={autoPlay} onChange={(e) => setAutoPlay(e.target.checked)} />} label={t("rec.autoplay")} />
            <TextField
              select
              size="small"
              label={t("rec.mic")}
              value={deviceId}
              onChange={(e) => setDeviceId(e.target.value)}
              sx={{ minWidth: 220 }}
              disabled={busy}
            >
              <MenuItem value="">{t("rec.micDefault")}</MenuItem>
              {devices.map((d) => (
                <MenuItem key={d.deviceId} value={d.deviceId}>
                  {d.label}
                </MenuItem>
              ))}
            </TextField>
            {!compact && <input ref={fileInputRef} type="file" accept="audio/*,.wav" multiple hidden onChange={(e) => e.target.files && importFiles(e.target.files)} />}
            {!compact && (
              <Button variant="text" startIcon={<UploadFileIcon />} onClick={() => fileInputRef.current?.click()} disabled={disabled || !!importing}>
                {importing ? t("rec.importing", { done: importing.done, total: importing.total }) : t("rec.import")}
              </Button>
            )}
          </Stack>

          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
            <Button size="small" startIcon={playAll ? <StopIcon /> : <PlaylistPlayIcon />} onClick={playEverything} disabled={list.length === 0}>
              {playAll ? t("rec.stop") : t("rec.playAll")}
            </Button>
            {!compact && (
              <Button size="small" color="error" startIcon={<DeleteSweepIcon />} onClick={() => removeMany(kind, [...selected])} disabled={disabled || selected.size === 0}>
                {t("rec.deleteSelected", { n: selected.size })}
              </Button>
            )}
            {problems > 0 && (
              <Tooltip title={t("rec.warningsTooltip")}>
                <Chip icon={<WarningAmberIcon />} color="warning" variant="outlined" size="small" label={t("rec.withWarnings", { n: problems })} onClick={() => setSelected(new Set(list.filter((r) => r.quality.issues.length).map((r) => r.id)))} />
              </Tooltip>
            )}
            {!compact && (
              <Typography variant="caption" color="text.secondary" sx={{ ml: "auto" }}>
                {t("rec.dragHint")}
              </Typography>
            )}
          </Stack>

          <Box sx={{ maxHeight: 360, overflow: "auto", border: 1, borderColor: "divider", borderRadius: 2 }}>
            {list.length === 0 && (
              <Typography sx={{ p: 2 }} color="text.secondary">
                {t("rec.empty")}
              </Typography>
            )}
            {[...list].reverse().map((rec, idx) => {
              const isPlaying = playing?.id === rec.id;
              return (
                <Stack
                  key={rec.id}
                  direction="row"
                  spacing={1}
                  alignItems="center"
                  sx={{
                    px: 1,
                    py: 0.5,
                    borderBottom: idx < list.length - 1 ? 1 : 0,
                    borderColor: "divider",
                    bgcolor: isPlaying ? "action.selected" : selected.has(rec.id) ? "action.hover" : "transparent",
                  }}
                >
                  {!compact && <Checkbox size="small" checked={selected.has(rec.id)} onChange={() => toggleSelected(rec.id)} disabled={disabled} />}
                  <IconButton size="small" onClick={() => togglePlay(rec)} color={isPlaying ? "primary" : "default"}>
                    {isPlaying ? <StopIcon /> : <PlayArrowIcon />}
                  </IconButton>
                  <Box sx={{ width: 140, cursor: "pointer" }} onClick={() => togglePlay(rec)}>
                    <Waveform peaks={rec.peaks} color={isPlaying ? theme.palette.primary.main : theme.palette.text.secondary} height={28} progress={isPlaying ? playing?.progress : undefined} />
                  </Box>
                  <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Typography variant="body2" noWrap>
                      #{list.length - idx} · {rec.duration.toFixed(2)} s · {new Date(rec.created).toLocaleTimeString()}
                      {rec.contributor && !compact && (
                        <Chip size="small" variant="outlined" label={t("rec.by", { name: rec.contributor })} sx={{ ml: 1, height: 18, fontSize: 11 }} />
                      )}
                      {rec.tag && rec.tag !== "normal" && (
                        <Chip size="small" color="secondary" variant="outlined" label={t(`tag.${rec.tag}` as TKey)} sx={{ ml: 1, height: 18, fontSize: 11 }} />
                      )}
                      {rec.review && !compact && (
                        <Tooltip title={t("rec.reviewHint")}>
                          <Chip size="small" color="warning" label={t("rec.review")} onClick={() => clearReview(rec)} sx={{ ml: 1, height: 18, fontSize: 11 }} />
                        </Tooltip>
                      )}
                    </Typography>
                    <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                      {rec.quality.issues.map((issue) => (
                        <Tooltip key={issue} title={t(`rec.issueHint.${issue}` as TKey)}>
                          <Chip size="small" color="warning" variant="outlined" label={t(`rec.issue.${issue}` as TKey)} sx={{ height: 20, fontSize: 11 }} />
                        </Tooltip>
                      ))}
                      {rec.quality.issues.length === 0 && (
                        <Typography variant="caption" color="text.secondary">
                          {t("rec.peakInfo", { peak: Math.round((rec.quality.peak ?? 0) * 100), db: rec.quality.rms_db ?? 0 })}
                        </Typography>
                      )}
                    </Stack>
                  </Box>
                  <Tooltip title={t("rec.deleteTooltip")}>
                    <span>
                      <IconButton size="small" onClick={() => removeMany(rec.kind, [rec.id])} disabled={disabled}>
                        <DeleteIcon />
                      </IconButton>
                    </span>
                  </Tooltip>
                </Stack>
              );
            })}
          </Box>
        </Stack>
      </CardContent>
      <Snackbar
        open={!!undo}
        autoHideDuration={8000}
        onClose={() => setUndo(null)}
        message={undo ? t("rec.deleted", { n: undo.ids.length }) : ""}
        action={
          <Button color="primary" size="small" onClick={restore}>
            {t("rec.undo")}
          </Button>
        }
      />
    </Card>
  );
}

function Waveform({ peaks, color, height, progress }: { peaks: number[]; color: string; height: number; progress?: number }) {
  const max = Math.max(0.05, ...peaks);
  return (
    <Box sx={{ position: "relative", display: "flex", alignItems: "center", gap: "1px", height }}>
      {peaks.map((p, i) => {
        const played = progress !== undefined && i / peaks.length <= progress;
        return <Box key={i} sx={{ flex: 1, height: `${Math.max(6, (p / max) * 100)}%`, bgcolor: color, borderRadius: 1, opacity: played ? 1 : progress !== undefined ? 0.35 : 0.75 }} />;
      })}
    </Box>
  );
}
