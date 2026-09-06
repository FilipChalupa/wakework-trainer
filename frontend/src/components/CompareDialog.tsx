import { useState } from "react";
import { Box, Button, Chip, Dialog, DialogActions, DialogContent, DialogTitle, LinearProgress, Stack, Table, TableBody, TableCell, TableHead, TableRow, Typography } from "@mui/material";
import { api, type Evaluation, type Job } from "../api";
import { errorText, useI18n } from "../i18n";

type Props = { open: boolean; jobs: [Job, Job] | null; onClose: () => void; onError: (m: string) => void };

/** Side-by-side comparison of two finished runs: parameters, metrics and per-recording evaluation. */
export function CompareDialog({ open, jobs, onClose, onError }: Props) {
  const { t } = useI18n();
  const [evals, setEvals] = useState<[Evaluation, Evaluation] | null>(null);
  const [busy, setBusy] = useState(false);

  const evaluate = async () => {
    if (!jobs) return;
    setBusy(true);
    try {
      const results = [] as Evaluation[];
      for (const job of jobs) {
        const cutoff = job.final_metrics?.manifest_cutoff ?? 0.97;
        results.push(await api.evaluateJob(job.job_id, cutoff, 5));
      }
      setEvals([results[0], results[1]]);
    } catch (e) {
      onError(errorText(t, e));
    } finally {
      setBusy(false);
    }
  };

  if (!jobs) return null;
  const [a, b] = jobs;
  const lastVal = (job: Job) => job.validation_last ?? undefined;
  const rows: { label: string; values: [string, string] }[] = [
    { label: t("compare.steps"), values: [String(a.training?.training_steps ?? "–"), String(b.training?.training_steps ?? "–")] },
    { label: t("compare.lr"), values: [String(a.training?.learning_rate ?? "–"), String(b.training?.learning_rate ?? "–")] },
    { label: t("compare.aug"), values: [String(a.training?.augmentations_per_sample ?? "–"), String(b.training?.augmentations_per_sample ?? "–")] },
    { label: t("compare.negWeight"), values: [String(a.training?.negative_class_weight ?? "–"), String(b.training?.negative_class_weight ?? "–")] },
    { label: t("compare.auc"), values: [a.final_metrics?.auc?.toFixed(3) ?? "–", b.final_metrics?.auc?.toFixed(3) ?? "–"] },
    { label: t("compare.autoCutoff"), values: [a.final_metrics?.manifest_cutoff?.toFixed(2) ?? "–", b.final_metrics?.manifest_cutoff?.toFixed(2) ?? "–"] },
    { label: t("compare.valLoss"), values: [lastVal(a)?.loss.toFixed(4) ?? "–", lastVal(b)?.loss.toFixed(4) ?? "–"] },
    { label: t("compare.recallNoFa"), values: [lastVal(a) ? `${Math.round(lastVal(a)!.recall_at_no_faph * 100)} %` : "–", lastVal(b) ? `${Math.round(lastVal(b)!.recall_at_no_faph * 100)} %` : "–"] },
  ];
  if (evals) {
    rows.push({ label: t("compare.detected"), values: [`${evals[0].summary.positive_detected} / ${evals[0].summary.positive_total}`, `${evals[1].summary.positive_detected} / ${evals[1].summary.positive_total}`] });
    rows.push({ label: t("compare.falseAccepts"), values: [`${evals[0].summary.negative_triggered} / ${evals[0].summary.negative_total}`, `${evals[1].summary.negative_triggered} / ${evals[1].summary.negative_total}`] });
  }
  const byId = evals ? new Map(evals[1].items.map((i) => [i.id, i])) : null;

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="md">
      <DialogTitle>{t("compare.title")}</DialogTitle>
      <DialogContent>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>{t("compare.param")}</TableCell>
              <TableCell>A · {new Date(a.created_at).toLocaleString()}</TableCell>
              <TableCell>B · {new Date(b.created_at).toLocaleString()}</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((r) => (
              <TableRow key={r.label}>
                <TableCell>{r.label}</TableCell>
                <TableCell>{r.values[0]}</TableCell>
                <TableCell>{r.values[1]}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 2 }}>
          <Button variant="outlined" onClick={evaluate} disabled={busy}>
            {t("compare.evaluate")}
          </Button>
          {busy && <LinearProgress sx={{ flex: 1 }} />}
        </Stack>
        {evals && byId && (
          <Box sx={{ maxHeight: 320, overflow: "auto", mt: 2 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell>{t("compare.recording")}</TableCell>
                  <TableCell>A</TableCell>
                  <TableCell>B</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {evals[0].items.map((item) => {
                  const other = byId.get(item.id);
                  const cell = (p: number | null | undefined, det: number | undefined, kind: string) => {
                    if (p === null || p === undefined) return "–";
                    const ok = kind === "positive" ? (det ?? 0) > 0 : (det ?? 0) === 0;
                    return <Chip size="small" label={`${Math.round(p * 100)} %`} color={ok ? "success" : "error"} variant={ok ? "outlined" : "filled"} />;
                  };
                  return (
                    <TableRow key={item.id}>
                      <TableCell>
                        <Typography variant="body2">{item.id}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {item.kind}
                          {item.tag ? ` · ${item.tag}` : ""}
                        </Typography>
                      </TableCell>
                      <TableCell>{cell(item.max_probability, item.detections, item.kind)}</TableCell>
                      <TableCell>{cell(other?.max_probability, other?.detections, item.kind)}</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>{t("compare.close")}</Button>
      </DialogActions>
    </Dialog>
  );
}
