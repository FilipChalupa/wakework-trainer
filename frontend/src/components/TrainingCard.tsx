import { useEffect, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  Collapse,
  LinearProgress,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import ModelTrainingIcon from "@mui/icons-material/ModelTraining";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import DownloadIcon from "@mui/icons-material/Download";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import { api, type TrainingState } from "../api";

type Props = {
  state: TrainingState;
  log: string[];
  connected: boolean;
  positiveCount: number;
  wakeWord: string;
  onError: (message: string) => void;
  onFinished: () => void;
};

const STATUS_LABEL: Record<TrainingState["status"], { label: string; color: "default" | "info" | "success" | "error" | "warning" }> = {
  idle: { label: "Připraveno", color: "default" },
  downloading: { label: "Stahuji data", color: "info" },
  preparing: { label: "Příprava dat", color: "info" },
  training: { label: "Trénuji", color: "info" },
  converting: { label: "Konverze do TFLite", color: "info" },
  done: { label: "Hotovo", color: "success" },
  failed: { label: "Chyba", color: "error" },
  cancelled: { label: "Zrušeno", color: "warning" },
};

function formatBytes(n: number) {
  if (n > 1 << 30) return `${(n / (1 << 30)).toFixed(2)} GB`;
  if (n > 1 << 20) return `${(n / (1 << 20)).toFixed(1)} MB`;
  return `${(n / 1024).toFixed(0)} kB`;
}

export function TrainingCard({ state, log, connected, positiveCount, wakeWord, onError, onFinished }: Props) {
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
      onError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const cancel = async () => {
    setBusy(true);
    try {
      await api.cancelTraining();
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const isTraining = state.status === "training";
  const total = isTraining ? state.total_steps : state.progress.total;
  const current = isTraining ? state.step : state.progress.current;
  const pct = total > 0 ? Math.min(100, (current / total) * 100) : 0;
  const lastValidation = state.validation[state.validation.length - 1];
  const statusMeta = STATUS_LABEL[state.status] ?? STATUS_LABEL.idle;

  return (
    <Card>
      <CardHeader
        avatar={<ModelTrainingIcon color="primary" />}
        title="4. Trénování modelu"
        subheader="microWakeWord (MixedNet) → kvantizovaný streamovaný TensorFlow Lite model"
        action={
          <Stack direction="row" spacing={1} alignItems="center">
            {!connected && <Chip size="small" label="offline" color="warning" variant="outlined" />}
            <Chip label={statusMeta.label} color={statusMeta.color} />
          </Stack>
        }
      />
      <CardContent>
        <Stack spacing={2}>
          {positiveCount < 20 && !running && (
            <Alert severity={positiveCount < 3 ? "error" : "warning"} variant="outlined">
              {positiveCount < 3
                ? "Pro trénování nahrajte alespoň 3 vzorky wake wordu."
                : `Máte ${positiveCount} vzorků. Model půjde natrénovat, ale s 20–40 vzorky bude výrazně spolehlivější.`}
            </Alert>
          )}

          <Stack direction="row" spacing={2} alignItems="center">
            {!running ? (
              <Button variant="contained" size="large" startIcon={<PlayArrowIcon />} onClick={start} disabled={busy || positiveCount < 3}>
                Spustit trénování
              </Button>
            ) : (
              <Button variant="outlined" color="error" size="large" startIcon={<StopIcon />} onClick={cancel} disabled={busy}>
                Zrušit
              </Button>
            )}
            <Box sx={{ flex: 1 }}>
              <Typography variant="subtitle2">{state.stage ?? "Čeká na spuštění"}</Typography>
              <Typography variant="body2" color="text.secondary">
                {state.message ?? (state.wake_word ? `Wake word: ${state.wake_word}` : `Wake word: ${wakeWord}`)}
              </Typography>
            </Box>
          </Stack>

          {(running || state.status === "done") && (
            <Box>
              <LinearProgress variant={total > 0 ? "determinate" : "indeterminate"} value={pct} sx={{ height: 10, borderRadius: 5 }} />
              <Stack direction="row" justifyContent="space-between" sx={{ mt: 0.5 }}>
                <Typography variant="caption" color="text.secondary">
                  {isTraining
                    ? `Krok ${state.step} / ${state.total_steps}`
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
              <Metric label="Loss (batch)" value={state.train_metrics.loss.toFixed(4)} />
              <Metric label="Přesnost" value={`${(state.train_metrics.accuracy * 100).toFixed(1)} %`} />
              <Metric label="Recall" value={`${(state.train_metrics.recall * 100).toFixed(1)} %`} />
              <Metric label="Precision" value={`${(state.train_metrics.precision * 100).toFixed(1)} %`} />
              {lastValidation && <Metric label="Val. loss" value={lastValidation.loss.toFixed(4)} />}
              {lastValidation && <Metric label="Val. recall" value={`${(lastValidation.recall * 100).toFixed(1)} %`} />}
              {lastValidation && <Metric label="Falešné aktivace / h" value={lastValidation.false_positives_per_hour.toFixed(2)} />}
            </Stack>
          )}

          {state.validation.length > 0 && (
            <Box sx={{ overflowX: "auto" }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Krok</TableCell>
                    <TableCell align="right">Val. loss</TableCell>
                    <TableCell align="right">Přesnost</TableCell>
                    <TableCell align="right">Recall</TableCell>
                    <TableCell align="right">Precision</TableCell>
                    <TableCell align="right">AUC</TableCell>
                    <TableCell align="right">Recall bez FA</TableCell>
                    <TableCell align="right">FA / hod</TableCell>
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

          {state.status === "failed" && (
            <Alert severity="error">
              {state.error ?? "Trénování selhalo."} Podrobnosti najdete v logu níže.
            </Alert>
          )}

          {state.status === "done" && state.model_url && (
            <Alert severity="success" variant="outlined" action={null}>
              <Stack spacing={1}>
                <Typography variant="body2">
                  Model je hotový. {state.final_metrics && <span>Výsledek na testovací sadě: {state.final_metrics}</span>}
                </Typography>
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                  <Button variant="contained" startIcon={<DownloadIcon />} href={state.model_url} download>
                    Stáhnout .tflite
                  </Button>
                  {state.manifest_url && (
                    <Button variant="outlined" startIcon={<DownloadIcon />} href={state.manifest_url} download>
                      Manifest pro ESPHome (.json)
                    </Button>
                  )}
                </Stack>
              </Stack>
            </Alert>
          )}

          <Box>
            <Button size="small" onClick={() => setShowLog((v) => !v)} startIcon={showLog ? <ExpandLessIcon /> : <ExpandMoreIcon />}>
              Log trénování ({log.length})
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
                  bgcolor: (t) => (t.palette.mode === "dark" ? "#05080f" : "#0f172a"),
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
