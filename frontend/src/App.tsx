import { useCallback, useEffect, useState } from "react";
import { Alert, AppBar, Box, Container, MenuItem, Select, Snackbar, Stack, Toolbar, Typography } from "@mui/material";
import RecordVoiceOverIcon from "@mui/icons-material/RecordVoiceOver";
import { AppThemeProvider } from "./theme";
import { api, type Job, type Project, type TrainingParams } from "./api";
import { ConfigCard } from "./components/ConfigCard";
import { RecorderCard } from "./components/RecorderCard";
import { DatasetsCard } from "./components/DatasetsCard";
import { TrainingCard } from "./components/TrainingCard";
import { JobsCard } from "./components/JobsCard";
import { TestCard } from "./components/TestCard";
import { useTrainingStream } from "./lib/useTrainingStream";
import { errorText, I18nProvider, useI18n, type Lang } from "./i18n";

export default function App() {
  return (
    <I18nProvider>
      <AppThemeProvider>
        <Main />
      </AppThemeProvider>
    </I18nProvider>
  );
}

function Main() {
  const { t, lang, setLang } = useI18n();
  const [project, setProject] = useState<Project | null>(null);
  const [defaults, setDefaults] = useState<TrainingParams | null>(null);
  const [counts, setCounts] = useState({ positive: 0, negative: 0 });
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState<string | null>(null);
  const { state, log, connected } = useTrainingStream();

  const showError = useCallback((message: string) => setError(message), []);

  const loadJobs = useCallback(() => {
    api
      .listJobs()
      .then((r) => setJobs(r.items))
      .catch((e) => showError(errorText(t, e)));
  }, [showError, t]);

  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);

  useEffect(() => {
    api
      .getConfig()
      .then((r) => {
        setProject(r.project);
        setDefaults(r.defaults);
      })
      .catch((e) => showError(errorText(t, e)));
    loadJobs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const running = ["downloading", "preparing", "training", "converting"].includes(state.status);

  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "background.default" }}>
      <AppBar position="sticky" color="default" elevation={0} sx={{ borderBottom: 1, borderColor: "divider", bgcolor: "background.paper" }}>
        <Toolbar>
          <RecordVoiceOverIcon color="primary" sx={{ mr: 1.5 }} />
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography variant="h6" component="h1" lineHeight={1.2}>
              {t("app.title")}
            </Typography>
            <Typography variant="caption" color="text.secondary" noWrap display="block">
              {t("app.subtitle")}
            </Typography>
          </Box>
          {project && (
            <Typography variant="subtitle1" fontWeight={600} color="primary" sx={{ mr: 2, display: { xs: "none", sm: "block" } }}>
              „{project.wake_word}“
            </Typography>
          )}
          <Select size="small" value={lang} onChange={(e) => setLang(e.target.value as Lang)} aria-label={t("app.language")} sx={{ minWidth: 90 }}>
            <MenuItem value="cs">Čeština</MenuItem>
            <MenuItem value="en">English</MenuItem>
          </Select>
        </Toolbar>
      </AppBar>

      <Container maxWidth="md" sx={{ py: 3 }}>
        <Stack spacing={3}>
          {project && defaults && <ConfigCard project={project} defaults={defaults} disabled={running} onSaved={setProject} onError={showError} />}
          {project && <RecorderCard wakeWord={project.wake_word} durationS={project.sample_duration_s} disabled={running} onCountsChange={setCounts} onError={showError} />}
          <DatasetsCard disabled={running} onError={showError} />
          <TrainingCard state={state} log={log} connected={connected} positiveCount={counts.positive} wakeWord={project?.wake_word ?? ""} onError={showError} onFinished={loadJobs} />
          <JobsCard jobs={jobs} disabled={running} onChanged={loadJobs} onError={showError} />
          <TestCard jobs={jobs} wakeWord={project?.wake_word ?? ""} disabled={running} onError={showError} />
          <Typography variant="caption" color="text.secondary" textAlign="center">
            {t("app.footer")}
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
