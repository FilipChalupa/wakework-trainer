import { useEffect, useState } from "react";
import { Box, Button, Card, CardContent, CardHeader, Chip, IconButton, LinearProgress, Stack, Tooltip, Typography } from "@mui/material";
import StorageIcon from "@mui/icons-material/Storage";
import DownloadIcon from "@mui/icons-material/Download";
import DeleteIcon from "@mui/icons-material/Delete";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import { api, type Dataset } from "../api";

type Props = { disabled: boolean; onError: (message: string) => void };

export function DatasetsCard({ disabled, onError }: Props) {
  const [items, setItems] = useState<Dataset[]>([]);

  const refresh = async () => {
    try {
      const res = await api.listDatasets();
      setItems(res.items);
    } catch (e) {
      onError((e as Error).message);
    }
  };

  useEffect(() => {
    refresh();
    const timer = setInterval(() => {
      refresh();
    }, 2000);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const download = async (id: string) => {
    try {
      await api.downloadDataset(id);
      await refresh();
    } catch (e) {
      onError((e as Error).message);
    }
  };

  const remove = async (id: string) => {
    try {
      await api.deleteDataset(id);
      await refresh();
    } catch (e) {
      onError((e as Error).message);
    }
  };

  return (
    <Card>
      <CardHeader avatar={<StorageIcon color="primary" />} title="3. Negativní datasety" subheader="Řeč a hluk, na které model nemá reagovat" />
      <CardContent>
        <Stack spacing={2}>
          {items.map((ds) => {
            const dl = ds.download;
            const active = dl && (dl.state === "downloading" || dl.state === "extracting");
            const pct = dl && dl.total && dl.received ? (dl.received / dl.total) * 100 : undefined;
            return (
              <Box key={ds.id} sx={{ p: 1.5, border: 1, borderColor: "divider", borderRadius: 2 }}>
                <Stack direction="row" spacing={1} alignItems="center">
                  <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                      <Typography variant="subtitle2">{ds.title}</Typography>
                      <Chip size="small" label={`${ds.size_mb} MB`} variant="outlined" />
                      {ds.required && <Chip size="small" label="základní" color="primary" variant="outlined" />}
                      {ds.installed && <Chip size="small" icon={<CheckCircleIcon />} label="staženo" color="success" />}
                    </Stack>
                    <Typography variant="body2" color="text.secondary">
                      {ds.description}
                    </Typography>
                    {dl?.state === "error" && (
                      <Typography variant="body2" color="error">
                        Chyba stahování: {dl.error}
                      </Typography>
                    )}
                  </Box>
                  {!ds.installed && (
                    <Button size="small" variant="outlined" startIcon={<DownloadIcon />} onClick={() => download(ds.id)} disabled={disabled || !!active}>
                      {active ? (dl?.state === "extracting" ? "Rozbaluji…" : "Stahuji…") : "Stáhnout"}
                    </Button>
                  )}
                  {ds.installed && (
                    <Tooltip title="Smazat stažená data">
                      <span>
                        <IconButton size="small" onClick={() => remove(ds.id)} disabled={disabled}>
                          <DeleteIcon />
                        </IconButton>
                      </span>
                    </Tooltip>
                  )}
                </Stack>
                {active && <LinearProgress sx={{ mt: 1 }} variant={pct !== undefined && dl?.state === "downloading" ? "determinate" : "indeterminate"} value={pct} />}
              </Box>
            );
          })}
          <Typography variant="caption" color="text.secondary">
            Kromě datasetů se při trénování automaticky generuje syntetický šum (bílý, růžový, hnědý, brum) a dlouhý „ambientní“ záznam pro odhad
            falešných aktivací za hodinu. Vlastní negativní nahrávky z kroku 2 se přidávají také.
          </Typography>
        </Stack>
      </CardContent>
    </Card>
  );
}
