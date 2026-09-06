import { useEffect, useState } from "react";
import { Accordion, AccordionDetails, AccordionSummary, Box, Button, Card, CardContent, CardHeader, MenuItem, Stack, TextField, Typography } from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import TuneIcon from "@mui/icons-material/Tune";
import SaveIcon from "@mui/icons-material/Save";
import { api, type Project, type TrainingParams } from "../api";
import { errorText, useI18n, type TKey } from "../i18n";

type Props = {
  project: Project;
  defaults: TrainingParams;
  disabled: boolean;
  onSaved: (project: Project) => void;
  onError: (message: string) => void;
};

const FIELDS: { key: keyof TrainingParams; step?: number; min?: number }[] = [
  { key: "training_steps", step: 100, min: 100 },
  { key: "learning_rate", step: 0.0001, min: 0.00001 },
  { key: "batch_size", step: 16, min: 16 },
  { key: "eval_step_interval", step: 50, min: 25 },
  { key: "augmentations_per_sample", step: 5, min: 1 },
  { key: "clip_duration_ms", step: 100, min: 800 },
  { key: "negative_class_weight", step: 1, min: 1 },
];

export function ConfigCard({ project, defaults, disabled, onSaved, onError }: Props) {
  const { t } = useI18n();
  const [wakeWord, setWakeWord] = useState(project.wake_word);
  const [duration, setDuration] = useState(project.sample_duration_s);
  const [training, setTraining] = useState<TrainingParams>(project.training);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setWakeWord(project.wake_word);
    setDuration(project.sample_duration_s);
    setTraining(project.training);
  }, [project]);

  const dirty = wakeWord !== project.wake_word || duration !== project.sample_duration_s || JSON.stringify(training) !== JSON.stringify(project.training);

  const save = async () => {
    setSaving(true);
    try {
      const res = await api.saveConfig({ wake_word: wakeWord, sample_duration_s: duration, training });
      onSaved(res.project);
    } catch (e) {
      onError(errorText(t, e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card>
      <CardHeader avatar={<TuneIcon color="primary" />} title={t("config.title")} subheader={t("config.subtitle")} />
      <CardContent>
        <Stack spacing={2}>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
            <TextField label={t("config.wakeWord")} value={wakeWord} onChange={(e) => setWakeWord(e.target.value)} placeholder="chaloupko" fullWidth disabled={disabled} helperText={t("config.wakeWordHelp")} />
            <TextField select label={t("config.sampleDuration")} value={duration} onChange={(e) => setDuration(Number(e.target.value))} disabled={disabled} sx={{ minWidth: 200 }} helperText={t("config.sampleDurationHelp")}>
              {[1.5, 2, 2.5, 3].map((v) => (
                <MenuItem key={v} value={v}>
                  {v.toFixed(1)} s
                </MenuItem>
              ))}
            </TextField>
          </Stack>

          <Accordion disableGutters variant="outlined">
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle2">{t("config.trainingParams")}</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ display: "grid", gap: 2, gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr", md: "1fr 1fr 1fr" } }}>
                {FIELDS.map((f) => (
                  <TextField
                    key={f.key}
                    type="number"
                    label={t(`config.f.${f.key}` as TKey)}
                    value={training[f.key]}
                    onChange={(e) => setTraining({ ...training, [f.key]: Number(e.target.value) })}
                    inputProps={{ step: f.step, min: f.min }}
                    helperText={t(`config.h.${f.key}` as TKey)}
                    disabled={disabled}
                    size="small"
                  />
                ))}
              </Box>
              <Button size="small" sx={{ mt: 1 }} onClick={() => setTraining({ ...defaults })} disabled={disabled}>
                {t("config.resetDefaults")}
              </Button>
            </AccordionDetails>
          </Accordion>

          <Stack direction="row" spacing={2} alignItems="center">
            <Button variant="contained" startIcon={<SaveIcon />} onClick={save} disabled={disabled || saving || !dirty || !wakeWord.trim()}>
              {t("config.save")}
            </Button>
            <Typography variant="body2" color="text.secondary">
              {dirty ? t("config.unsaved") : t("config.saved")}
            </Typography>
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
}
