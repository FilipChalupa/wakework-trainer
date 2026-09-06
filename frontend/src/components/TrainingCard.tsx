import { useEffect, useRef, useState } from "react";
import { Alert, Box, Button, Card, CardContent, CardHeader, Chip, Collapse, LinearProgress, Stack, Table, TableBody, TableCell, TableHead, TableRow, Typography } from "@mui/material";
import ModelTrainingIcon from "@mui/icons-material/ModelTraining";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import DownloadIcon from "@mui/icons-material/Download";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import ArchiveIcon from "@mui/icons-material/Archive";
import { api, type TrainingState } from "../api";
import { errorText, useI18n, type TKey } from "../i18n";
import { TrainingCharts } from "./TrainingCharts";

type Props = {
  state: TrainingState;
  log: string[];
  connected: boolean;
  positiveCount: number;
  wakeWord: string;
  onError: (message: string) => void;
  onFinished: () => void;
};

const STATUS_COLOR: Record<TrainingState["status"], "default" | "info" | "success" | "error" | "warning"> = {
  idle: "default",
  downloading: "info",
  preparing: "info",
  training: "info",
  converting: "info",
  done: "success",
  failed: "error",
  cancelled: "warning",
  interrupted: "warning",
};

const STAGE_KEYS = new Set(["checking_datasets", "downloading_dataset", "extracting_dataset", "preparing", "training", "converting", "done", "failed", "cancelled", "interrupted"]);
const MSG_KEYS = new Set(["init_tf", "augment_positive", "hard_negatives", "speech_features", "noise_features", "ambient_features", "train_steps", "find_model", "model_ready"]);

function formatBytes(n: number) {
  if (n > 1 << 30) return `${(n / (1 << 30)).toFixed(2)} GB`;
  if (n > 1 << 20) return `${(n / (1 << 20)).toFixed(1)} MB`;
  return `${(n / 1024).toFixed(0)} kB`;
}

export function TrainingCard({ state, log, connected, positiveCount, wakeWord, onError, onFinished }: Props) {
  const { t } = useI18n();
  const [showLog, setShowLog] = useState(false);
  const [busy, setBusy] = useState(false);
  const logRef = useRef<HTMLDivElement | null>(null);
  const prevStatus = useRef(state.status);

  useEffect(() => {
    if (showLog && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log, showLog]);

  useEffect(() => {
    if (prevStatus.current !== state.status && ["done", "failed", "cancelled"].includes(state.status)) onFinished();
    prevStatus.current = state.status;
  }, [state.status, onFinished]);

  const running = ["downloading", "preparing", "training", "converting"].includes(state.status);

  const start = async () => {
    setBusy(true);
    try {
      await api.startTraining();
      setShowLog(true);
    } catch (e) {
      onError(errorText(t, e));
    } finally {
      setBusy(false);
    }
  };

  const resume = async () => {
    setBusy(true);
    try {
      await api.resumeTraining();
      setShowLog(true);
    } catch (e) {
      onError(errorText(t, e));
    } finally {
      setBusy(false);
    }
  };

  const cancel = async () => {
    setBusy(true);
    try {
      await api.cancelTraining();
    } catch (e) {
      onError(errorText(t, e));
    } finally {
      setBusy(false);
    }
  };

  const isTraining = state.status === "training";
  const total = isTraining ? state.total_steps : state.progress.total;
  const current = isTraining ? state.step : state.progress.current;
  const pct = total > 0 ? Math.min(100, (current / total) * 100) : 0;
  const lastValidation = state.validation[state.validation.length - 1];
  const params = state.message_params ?? {};
  const stageText = state.stage_key && STAGE_KEYS.has(state.stage_key) ? t(`train.stage.${state.stage_key}` as TKey, params) : state.stage;
  const messageText = state.message_key && MSG_KEYS.has(state.message_key) ? t(`train.msg.${state.message_key}` as TKey, params) : state.message;

  return (
    <Card>
      <CardHeader
        avatar={<ModelTrainingIcon color="primary" />}
        title={t("train.title")}
        subheader={t("train.subtitle")}
        action={
          <Stack direction="row" spacing={1} alignItems="center">
            {!connected && <Chip size="small" label={t("train.offline")} color="warning" variant="outlined" />}
            <Chip label={t(`train.status.${state.status}` as TKey)} color={STATUS_COLOR[state.status] ?? "default"} />
          </Stack>
        }
      />
      <CardContent>
        <Stack spacing={2}>
          {positiveCount < 20 && !running && (
            <Alert severity={positiveCount < 3 ? "error" : "warning"} variant="outlined">
              {positiveCount < 3 ? t("train.fewSamplesError") : t("train.fewSamplesWarn", { n: positiveCount })}
            </Alert>
          )}

          <Stack direction="row" spacing={2} alignItems="center">
            {!running ? (
              <Button variant="contained" size="large" startIcon={<PlayArrowIcon />} onClick={start} disabled={busy || positiveCount < 3}>
                {t("train.start")}
              </Button>
            ) : (
              <Button variant="outlined" color="error" size="large" startIcon={<StopIcon />} onClick={cancel} disabled={busy}>
                {t("train.cancel")}
              </Button>
            )}
            <Box sx={{ flex: 1 }}>
              <Typography variant="subtitle2">{stageText ?? t("train.waiting")}</Typography>
              <Typography variant="body2" color="text.secondary">
                {messageText ?? t("train.wakeWord", { word: state.wake_word ?? wakeWord })}
              </Typography>
            </Box>
          </Stack>

          {(running || state.status === "done") && (
            <Box>
              <LinearProgress variant={total > 0 ? "determinate" : "indeterminate"} value={pct} sx={{ height: 10, borderRadius: 5 }} />
              <Stack direction="row" justifyContent="space-between" sx={{ mt: 0.5 }}>
                <Typography variant="caption" color="text.secondary">
                  {isTraining
                    ? t("train.step", { step: state.step, total: state.total_steps })
                    : state.status === "downloading" && total > 0
                      ? `${formatBytes(current)} / ${formatBytes(total)}`
                      : total > 0
                        ? `${current} / ${total}`
                        : ""}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {pct.toFixed(0)} %
                </Typography>
              </Stack>
            </Box>
          )}

          {state.train_metrics && (
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              <Metric label={t("train.m.lossBatch")} value={state.train_metrics.loss.toFixed(4)} />
              <Metric label={t("train.m.accuracy")} value={`${(state.train_metrics.accuracy * 100).toFixed(1)} %`} />
              <Metric label={t("train.m.recall")} value={`${(state.train_metrics.recall * 100).toFixed(1)} %`} />
              <Metric label={t("train.m.precision")} value={`${(state.train_metrics.precision * 100).toFixed(1)} %`} />
              {lastValidation && <Metric label={t("train.m.valLoss")} value={lastValidation.loss.toFixed(4)} />}
              {lastValidation && <Metric label={t("train.m.valRecall")} value={`${(lastValidation.recall * 100).toFixed(1)} %`} />}
              {lastValidation && <Metric label={t("train.m.faph")} value={lastValidation.false_positives_per_hour.toFixed(2)} />}
            </Stack>
          )}

          {state.validation.length > 0 && (
            <Box sx={{ overflowX: "auto" }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>{t("train.t.step")}</TableCell>
                    <TableCell align="right">{t("train.m.valLoss")}</TableCell>
                    <TableCell align="right">{t("train.m.accuracy")}</TableCell>
                    <TableCell align="right">{t("train.m.recall")}</TableCell>
                    <TableCell align="right">{t("train.m.precision")}</TableCell>
                    <TableCell align="right">{t("train.t.auc")}</TableCell>
                    <TableCell align="right">{t("train.t.recallNoFa")}</TableCell>
                    <TableCell align="right">{t("train.t.faph")}</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {[...state.validation].slice(-8).map((v) => (
                    <TableRow key={v.step}>
                      <TableCell>{v.step}</TableCell>
                      <TableCell align="right">{v.loss.toFixed(4)}</TableCell>
                      <TableCell align="right">{(v.accuracy * 100).toFixed(1)} %</TableCell>
                      <TableCell align="right">{(v.recall * 100).toFixed(1)} %</TableCell>
                      <TableCell align="right">{(v.precision * 100).toFixed(1)} %</TableCell>
                      <TableCell align="right">{v.auc.toFixed(3)}</TableCell>
                      <TableCell align="right">{(v.recall_at_no_faph * 100).toFixed(1)} %</TableCell>
                      <TableCell align="right">{v.false_positives_per_hour.toFixed(2)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Box>
          )}

          {state.status === "interrupted" && (
            <Alert
              severity="warning"
              action={
                state.resumable ? (
                  <Button color="inherit" size="small" startIcon={<RestartAltIcon />} onClick={resume} disabled={busy}>
                    {t("train.resume")}
                  </Button>
                ) : undefined
              }
            >
              {t("train.resumeHint")}
            </Alert>
          )}

          {state.validation.length > 1 && <TrainingCharts validation={state.validation} roc={state.final_metrics?.points} />}

          {state.status === "failed" && (
            <Alert severity="error">
              {state.error ?? t("train.failed")} {t("train.failedHint")}
            </Alert>
          )}

          {state.status === "done" && state.model_url && (
            <Alert severity="success" variant="outlined">
              <Stack spacing={1}>
                <Typography variant="body2">
                  {t("train.done")}{" "}
                  {state.final_metrics &&
                    t("train.finalMetrics", {
                      auc: state.final_metrics.auc?.toFixed(3) ?? "–",
                      cutoff: state.final_metrics.cutoff.toFixed(2),
                      frr: Math.round(state.final_metrics.frr * 100),
                      faph: state.final_metrics.faph.toFixed(2),
                    })}
                </Typography>
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                  <Button variant="contained" startIcon={<DownloadIcon />} href={state.model_url} download>
                    {t("train.downloadModel")}
                  </Button>
                  {state.manifest_url && (
                    <Button variant="outlined" startIcon={<DownloadIcon />} href={state.manifest_url} download>
                      {t("train.downloadManifest")}
                    </Button>
                  )}
                  {state.export_url && (
                    <Button variant="outlined" startIcon={<ArchiveIcon />} href={state.export_url} download>
                      {t("train.export")}
                    </Button>
                  )}
                </Stack>
              </Stack>
            </Alert>
          )}

          <Box>
            <Button size="small" onClick={() => setShowLog((v) => !v)} startIcon={showLog ? <ExpandLessIcon /> : <ExpandMoreIcon />}>
              {t("train.log", { n: log.length })}
            </Button>
            <Collapse in={showLog}>
              <Box
                ref={logRef}
                sx={{
                  mt: 1,
                  maxHeight: 320,
                  overflow: "auto",
                  p: 1.5,
                  borderRadius: 2,
                  bgcolor: (th) => (th.palette.mode === "dark" ? "#05080f" : "#0f172a"),
                  color: "#cbd5e1",
                  fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
                  fontSize: 11,
                  lineHeight: 1.5,
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-all",
                }}
              >
                {log.length === 0 ? "—" : log.join("\n")}
              </Box>
            </Collapse>
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <Box sx={{ px: 1.5, py: 1, border: 1, borderColor: "divider", borderRadius: 2, minWidth: 120 }}>
      <Typography variant="caption" color="text.secondary" display="block">
        {label}
      </Typography>
      <Typography variant="subtitle1" fontWeight={600}>
        {value}
      </Typography>
    </Box>
  );
}
