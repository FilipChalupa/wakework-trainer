import { useCallback, useEffect, useState } from "react";
import { Alert, AppBar, Box, Container, Snackbar, Stack, Toolbar, Typography } from "@mui/material";
import RecordVoiceOverIcon from "@mui/icons-material/RecordVoiceOver";
import { AppThemeProvider } from "./theme";
import { api, type Job, type Project, type TrainingParams } from "./api";
import { ConfigCard } from "./components/ConfigCard";
import { RecorderCard } from "./components/RecorderCard";
import { DatasetsCard } from "./components/DatasetsCard";
import { TrainingCard } from "./components/TrainingCard";
import { JobsCard } from "./components/JobsCard";
import { useTrainingStream } from "./lib/useTrainingStream";

export default function App() {
  return (
    <AppThemeProvider>
      <Main />
    </AppThemeProvider>
  );
}

function Main() {
  const [project, setProject] = useState<Project | null>(null);
  const [defaults, setDefaults] = useState<TrainingParams | null>(null);
  const [counts, setCounts] = useState({ positive: 0, negative: 0 });
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState<string | null>(null);
  const { state, log, connected } = useTrainingStream();

  const showError = useCallback((message: string) => setError(message), []);

  const loadJobs = useCallback(() => {
    api.listJobs().then((r) => setJobs(r.items)).catch((e) => showError((e as Error).message));
  }, [showError]);

  useEffect(() => {
    api
      .getConfig()
      .then((r) => {
        setProject(r.project);
        setDefaults(r.defaults);
      })
      .catch((e) => showError((e as Error).message));
    loadJobs();
  }, [loadJobs, showError]);

  const running = ["downloading", "preparing", "training", "converting"].includes(state.status);

  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "background.default" }}>
      <AppBar position="sticky" color="default" elevation={0} sx={{ borderBottom: 1, borderColor: "divider", bgcolor: "background.paper" }}>
        <Toolbar>
          <RecordVoiceOverIcon color="primary" sx={{ mr: 1.5 }} />
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" component="h1" lineHeight={1.2}>
              Wake Word Trainer
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Nahrajte vzorky, natrénujte microWakeWord model a stáhněte .tflite pro ESPHome
            </Typography>
          </Box>
          {project && (
            <Typography variant="subtitle1" fontWeight={600} color="primary">
              „{project.wake_word}“
            </Typography>
          )}
        </Toolbar>
      </AppBar>

      <Container maxWidth="md" sx={{ py: 3 }}>
        <Stack spacing={3}>
          {project && defaults && (
            <ConfigCard project={project} defaults={defaults} disabled={running} onSaved={setProject} onError={showError} />
          )}
          {project && (
            <RecorderCard wakeWord={project.wake_word} durationS={project.sample_duration_s} disabled={running} onCountsChange={setCounts} onError={showError} />
          )}
          <DatasetsCard disabled={running} onError={showError} />
          <TrainingCard
            state={state}
            log={log}
            connected={connected}
            positiveCount={counts.positive}
            wakeWord={project?.wake_word ?? ""}
            onError={showError}
            onFinished={loadJobs}
          />
          <JobsCard jobs={jobs} disabled={running} onChanged={loadJobs} onError={showError} />
          <Typography variant="caption" color="text.secondary" textAlign="center">
            Data se ukládají do svazku <code>/data</code>. Trénování běží na serveru (GPU, pokud je dostupné), model je kompatibilní s ESPHome
            komponentou <code>micro_wake_word</code>.
          </Typography>
        </Stack>
      </Container>

      <Snackbar open={!!error} autoHideDuration={8000} onClose={() => setError(null)} anchorOrigin={{ vertical: "bottom", horizontal: "center" }}>
        <Alert severity="error" onClose={() => setError(null)} variant="filled">
          {error}
        </Alert>
      </Snackbar>
    </Box>
  );
}
