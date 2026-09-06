import { Box, Button, Card, CardContent, CardHeader, Chip, IconButton, Stack, Table, TableBody, TableCell, TableHead, TableRow, Tooltip, Typography } from "@mui/material";
import HistoryIcon from "@mui/icons-material/History";
import DownloadIcon from "@mui/icons-material/Download";
import DeleteIcon from "@mui/icons-material/Delete";
import { useState } from "react";
import CompareArrowsIcon from "@mui/icons-material/CompareArrows";
import { Checkbox } from "@mui/material";
import { api, type Job } from "../api";
import { errorText, useI18n, type TKey } from "../i18n";
import { CompareDialog } from "./CompareDialog";

type Props = { jobs: Job[]; disabled: boolean; onChanged: () => void; onError: (message: string) => void };

const COLORS: Record<string, "success" | "error" | "warning" | "info" | "default"> = { done: "success", failed: "error", cancelled: "warning", running: "info", interrupted: "warning" };
const STATUSES = new Set(["done", "failed", "cancelled", "running", "interrupted"]);

export function JobsCard({ jobs, disabled, onChanged, onError }: Props) {
  const { t } = useI18n();
  const [selected, setSelected] = useState<string[]>([]);
  const [compareOpen, setCompareOpen] = useState(false);
  const toggle = (id: string) => setSelected((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur.slice(-1), id]));
  const pair = selected.length === 2 ? (selected.map((id) => jobs.find((j) => j.job_id === id)).filter(Boolean) as [Job, Job]) : null;
  const remove = async (id: string) => {
    try {
      await api.deleteJob(id);
      onChanged();
    } catch (e) {
      onError(errorText(t, e));
    }
  };

  return (
    <Card>
      <CardHeader
        avatar={<HistoryIcon color="primary" />}
        title={t("jobs.title")}
        subheader={t("jobs.subtitle")}
        action={
          <Tooltip title={t("compare.pick")}>
            <span>
              <Button size="small" startIcon={<CompareArrowsIcon />} onClick={() => setCompareOpen(true)} disabled={!pair || pair.length < 2}>
                {t("compare.title")} ({selected.length}/2)
              </Button>
            </span>
          </Tooltip>
        }
      />
      <CardContent>
        {jobs.length === 0 ? (
          <Typography color="text.secondary">{t("jobs.empty")}</Typography>
        ) : (
          <Box sx={{ overflowX: "auto" }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell padding="checkbox" />
                  <TableCell>{t("jobs.started")}</TableCell>
                  <TableCell>{t("jobs.wakeWord")}</TableCell>
                  <TableCell>{t("jobs.status")}</TableCell>
                  <TableCell>{t("jobs.samples")}</TableCell>
                  <TableCell>{t("jobs.steps")}</TableCell>
                  <TableCell>{t("jobs.model")}</TableCell>
                  <TableCell />
                </TableRow>
              </TableHead>
              <TableBody>
                {jobs.map((job) => (
                  <TableRow key={job.job_id} hover selected={selected.includes(job.job_id)}>
                    <TableCell padding="checkbox">
                      <Checkbox size="small" checked={selected.includes(job.job_id)} onChange={() => toggle(job.job_id)} disabled={job.status !== "done"} />
                    </TableCell>
                    <TableCell>{new Date(job.created_at).toLocaleString()}</TableCell>
                    <TableCell>
                      {job.wake_word}
                      {job.final_metrics && (
                        <Typography variant="caption" color="text.secondary" display="block">
                          {t("jobs.metrics", {
                            auc: job.final_metrics.auc?.toFixed(3) ?? "–",
                            cutoff: job.final_metrics.cutoff.toFixed(2),
                            frr: Math.round(job.final_metrics.frr * 100),
                            faph: job.final_metrics.faph.toFixed(2),
                          })}
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell>
                      <Chip size="small" label={t((STATUSES.has(job.status) ? `jobs.s.${job.status}` : "jobs.s.failed") as TKey)} color={COLORS[job.status] ?? "default"} />
                    </TableCell>
                    <TableCell>{job.positive_count}</TableCell>
                    <TableCell>{job.training?.training_steps}</TableCell>
                    <TableCell>
                      {job.model_url ? (
                        <Stack direction="row" spacing={1}>
                          <Button size="small" variant="outlined" startIcon={<DownloadIcon />} href={job.model_url} download>
                            {job.slug}.tflite{job.model_size ? ` (${(job.model_size / 1024).toFixed(0)} kB)` : ""}
                          </Button>
                          {job.manifest_url && (
                            <Button size="small" href={job.manifest_url} download>
                              {t("jobs.manifest")}
                            </Button>
                          )}
                          {job.export_url && (
                            <Button size="small" href={job.export_url} download>
                              {t("jobs.export")}
                            </Button>
                          )}
                        </Stack>
                      ) : (
                        <Typography variant="caption" color="text.secondary">
                          —
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell align="right">
                      <Tooltip title={t("jobs.deleteTooltip")}>
                        <span>
                          <IconButton size="small" onClick={() => remove(job.job_id)} disabled={disabled || job.status === "running"}>
                            <DeleteIcon />
                          </IconButton>
                        </span>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        )}
      </CardContent>
      <CompareDialog open={compareOpen && !!pair} jobs={pair} onClose={() => setCompareOpen(false)} onError={onError} />
    </Card>
  );
}
