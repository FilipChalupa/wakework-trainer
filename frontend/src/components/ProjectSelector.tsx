import { useRef, useState } from "react";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import FileUploadIcon from "@mui/icons-material/FileUpload";
import { Button, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, MenuItem, Select, Stack, TextField, Tooltip, Typography } from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import DeleteIcon from "@mui/icons-material/Delete";
import { api, type ProjectSummary } from "../api";
import { errorText, useI18n } from "../i18n";

type Props = { projects: ProjectSummary[]; current: string; disabled: boolean; onChanged: (items: ProjectSummary[], current: string) => void; onError: (m: string) => void };

export function ProjectSelector({ projects, current, disabled, onChanged, onError }: Props) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [wakeWord, setWakeWord] = useState("");
  const [busy, setBusy] = useState(false);
  const [importing, setImporting] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const active = projects.find((p) => p.id === current);

  const importFile = async (file: File) => {
    setImporting(true);
    try {
      const res = await api.importProject(file);
      onChanged(res.items, res.current);
    } catch (e) {
      onError(errorText(t, e));
    } finally {
      setImporting(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const run = async (fn: () => Promise<{ items: ProjectSummary[]; current: string }>) => {
    setBusy(true);
    try {
      const res = await fn();
      onChanged(res.items, res.current);
    } catch (e) {
      onError(errorText(t, e));
    } finally {
      setBusy(false);
    }
  };

  const create = async () => {
    if (!name.trim()) return;
    await run(() => api.createProject(name.trim(), wakeWord.trim() || name.trim()));
    setOpen(false);
    setName("");
    setWakeWord("");
  };

  const remove = async () => {
    if (!active) return;
    if (!window.confirm(t("proj.deleteConfirm", { name: active.name }))) return;
    await run(() => api.deleteProject(active.id));
  };

  return (
    <Stack direction="row" spacing={0.5} alignItems="center">
      <Select size="small" value={current} onChange={(e) => run(() => api.selectProject(e.target.value))} disabled={disabled || busy} sx={{ minWidth: 150 }} aria-label={t("proj.label")} renderValue={(v) => projects.find((p) => p.id === v)?.name ?? v}>
        {projects.map((p) => (
          <MenuItem key={p.id} value={p.id}>
            <Stack>
              <Typography variant="body2">
                {p.name} · „{p.wake_word}“
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {t("proj.summary", { pos: p.positive_count, jobs: p.jobs })}
              </Typography>
            </Stack>
          </MenuItem>
        ))}
      </Select>
      <Tooltip title={t("proj.new")}>
        <span>
          <IconButton size="small" onClick={() => setOpen(true)} disabled={disabled || busy}>
            <AddIcon />
          </IconButton>
        </span>
      </Tooltip>
      <Tooltip title={t("proj.export")}>
        <span>
          <IconButton size="small" component="a" href={active ? api.exportProjectUrl(active.id) : undefined} download disabled={!active || busy}>
            <FileDownloadIcon />
          </IconButton>
        </span>
      </Tooltip>
      <Tooltip title={importing ? t("proj.importing") : t("proj.import")}>
        <span>
          <IconButton size="small" onClick={() => fileRef.current?.click()} disabled={disabled || busy || importing}>
            <FileUploadIcon />
          </IconButton>
        </span>
      </Tooltip>
      <input ref={fileRef} type="file" accept=".zip,application/zip" hidden onChange={(e) => e.target.files?.[0] && importFile(e.target.files[0])} />
      <Tooltip title={t("proj.delete")}>
        <span>
          <IconButton size="small" onClick={remove} disabled={disabled || busy || projects.length < 2}>
            <DeleteIcon />
          </IconButton>
        </span>
      </Tooltip>
      <Dialog open={open} onClose={() => setOpen(false)} fullWidth maxWidth="xs">
        <DialogTitle>{t("proj.new")}</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField autoFocus label={t("proj.newName")} value={name} onChange={(e) => setName(e.target.value)} fullWidth />
            <TextField label={t("proj.newWakeWord")} value={wakeWord} onChange={(e) => setWakeWord(e.target.value)} placeholder={name} fullWidth />
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>{t("proj.cancel")}</Button>
          <Button variant="contained" onClick={create} disabled={!name.trim() || busy}>
            {t("proj.create")}
          </Button>
        </DialogActions>
      </Dialog>
    </Stack>
  );
}
