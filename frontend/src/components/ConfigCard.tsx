import { useEffect, useState } from "react";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  MenuItem,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import TuneIcon from "@mui/icons-material/Tune";
import SaveIcon from "@mui/icons-material/Save";
import { api, type Project, type TrainingParams } from "../api";

type Props = {
  project: Project;
  defaults: TrainingParams;
  disabled: boolean;
  onSaved: (project: Project) => void;
  onError: (message: string) => void;
};

const FIELDS: { key: keyof TrainingParams; label: string; help: string; step?: number; min?: number }[] = [
  { key: "training_steps", label: "Trénovací kroky", help: "Počet mini-batch kroků (u microWakeWord odpovídá „epochám“). 3 000–10 000.", step: 100, min: 100 },
  { key: "learning_rate", label: "Learning rate", help: "Rychlost učení optimalizátoru Adam.", step: 0.0001, min: 0.00001 },
  { key: "batch_size", label: "Batch size", help: "Počet spektrogramů v jednom kroku.", step: 16, min: 16 },
  { key: "eval_step_interval", label: "Interval validace", help: "Po kolika krocích se model vyhodnotí na validační sadě.", step: 50, min: 25 },
  { key: "augmentations_per_sample", label: "Augmentací na vzorek", help: "Kolik variant (šum, gain, pitch…) se vygeneruje z každé nahrávky.", step: 5, min: 1 },
  { key: "clip_duration_ms", label: "Délka okna modelu (ms)", help: "Maximální délka wake wordu, kterou streamovaný model sleduje. Automaticky se zvětší, pokud jsou nahrávky delší.", step: 100, min: 800 },
  { key: "negative_class_weight", label: "Váha negativní třídy", help: "Vyšší hodnota = méně falešných aktivací, ale hůře se spouští.", step: 1, min: 1 },
];

export function ConfigCard({ project, defaults, disabled, onSaved, onError }: Props) {
  const [wakeWord, setWakeWord] = useState(project.wake_word);
  const [duration, setDuration] = useState(project.sample_duration_s);
  const [training, setTraining] = useState<TrainingParams>(project.training);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setWakeWord(project.wake_word);
    setDuration(project.sample_duration_s);
    setTraining(project.training);
  }, [project]);

  const dirty =
    wakeWord !== project.wake_word ||
    duration !== project.sample_duration_s ||
    JSON.stringify(training) !== JSON.stringify(project.training);

  const save = async () => {
    setSaving(true);
    try {
      const res = await api.saveConfig({ wake_word: wakeWord, sample_duration_s: duration, training });
      onSaved(res.project);
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card>
      <CardHeader avatar={<TuneIcon color="primary" />} title="1. Konfigurace wake wordu" subheader="Cílové slovo a parametry trénování" />
      <CardContent>
        <Stack spacing={2}>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
            <TextField
              label="Wake word"
              value={wakeWord}
              onChange={(e) => setWakeWord(e.target.value)}
              placeholder="chaloupko"
              fullWidth
              disabled={disabled}
              helperText="Slovo, které budete nahrávat. Název modelu se z něj odvodí (např. chaloupko.tflite)."
            />
            <TextField
              select
              label="Délka jednoho vzorku"
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              disabled={disabled}
              sx={{ minWidth: 200 }}
              helperText="Pevná délka nahrávky."
            >
              {[1.5, 2, 2.5, 3].map((v) => (
                <MenuItem key={v} value={v}>
                  {v.toFixed(1)} s
                </MenuItem>
              ))}
            </TextField>
          </Stack>

          <Accordion disableGutters variant="outlined">
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle2">Parametry trénování</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ display: "grid", gap: 2, gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr", md: "1fr 1fr 1fr" } }}>
                {FIELDS.map((f) => (
                  <TextField
                    key={f.key}
                    type="number"
                    label={f.label}
                    value={training[f.key]}
                    onChange={(e) => setTraining({ ...training, [f.key]: Number(e.target.value) })}
                    inputProps={{ step: f.step, min: f.min }}
                    helperText={f.help}
                    disabled={disabled}
                    size="small"
                  />
                ))}
              </Box>
              <Button size="small" sx={{ mt: 1 }} onClick={() => setTraining({ ...defaults })} disabled={disabled}>
                Obnovit výchozí hodnoty
              </Button>
            </AccordionDetails>
          </Accordion>

          <Stack direction="row" spacing={2} alignItems="center">
            <Button variant="contained" startIcon={<SaveIcon />} onClick={save} disabled={disabled || saving || !dirty || !wakeWord.trim()}>
              Uložit konfiguraci
            </Button>
            <Typography variant="body2" color="text.secondary">
              {dirty ? "Neuložené změny" : "Uloženo"}
            </Typography>
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
}
