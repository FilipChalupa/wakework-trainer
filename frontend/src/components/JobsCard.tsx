import { Box, Button, Card, CardContent, CardHeader, Chip, IconButton, Stack, Table, TableBody, TableCell, TableHead, TableRow, Tooltip, Typography } from "@mui/material";
import HistoryIcon from "@mui/icons-material/History";
import DownloadIcon from "@mui/icons-material/Download";
import DeleteIcon from "@mui/icons-material/Delete";
import { api, type Job } from "../api";

type Props = { jobs: Job[]; disabled: boolean; onChanged: () => void; onError: (message: string) => void };

const COLORS: Record<string, "success" | "error" | "warning" | "info" | "default"> = {
  done: "success",
  failed: "error",
  cancelled: "warning",
  running: "info",
};

export function JobsCard({ jobs, disabled, onChanged, onError }: Props) {
  const remove = async (id: string) => {
    try {
      await api.deleteJob(id);
      onChanged();
    } catch (e) {
      onError((e as Error).message);
    }
  };

  return (
    <Card>
      <CardHeader avatar={<HistoryIcon color="primary" />} title="5. Natrénované modely" subheader="Historie trénování a stažení výstupů" />
      <CardContent>
        {jobs.length === 0 ? (
          <Typography color="text.secondary">Zatím žádné trénování.</Typography>
        ) : (
          <Box sx={{ overflowX: "auto" }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Spuštěno</TableCell>
                  <TableCell>Wake word</TableCell>
                  <TableCell>Stav</TableCell>
                  <TableCell>Vzorků</TableCell>
                  <TableCell>Kroků</TableCell>
                  <TableCell>Model</TableCell>
                  <TableCell />
                </TableRow>
              </TableHead>
              <TableBody>
                {jobs.map((job) => (
                  <TableRow key={job.job_id} hover>
                    <TableCell>{new Date(job.created_at).toLocaleString()}</TableCell>
                    <TableCell>{job.wake_word}</TableCell>
                    <TableCell>
                      <Chip size="small" label={job.status} color={COLORS[job.status] ?? "default"} />
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
                              manifest
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
                      <Tooltip title="Smazat běh včetně modelu">
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
    </Card>
  );
}
