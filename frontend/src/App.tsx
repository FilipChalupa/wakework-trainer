import { useCallback, useEffect, useState } from "react";
import { Alert, AppBar, Badge, Box, Container, MenuItem, Select, Snackbar, Stack, Tab, Tabs, Toolbar, Typography } from "@mui/material";
import RecordVoiceOverIcon from "@mui/icons-material/RecordVoiceOver";
import { AppThemeProvider } from "./theme";
import { api, type Job, type Project, type ProjectSummary, type TrainingParams } from "./api";
import { ProjectSelector } from "./components/ProjectSelector";
import { SystemChip } from "./components/SystemChip";
import { DeployCard } from "./components/DeployCard";
import { ContributePage } from "./ContributePage";
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
        {location.pathname.replace(/\/$/, "") === "/contribute" ? <ContributePage /> : <Main />}
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
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [currentProject, setCurrentProject] = useState<string>("");
  const [reloadKey, setReloadKey] = useState(0);
  const [jobsVersion, setJobsVersion] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const { state, log, connected } = useTrainingStream();
  const TABS = ["data", "train", "test", "deploy"] as const;
  type TabId = (typeof TABS)[number];
  const [tab, setTab] = useState<TabId>(() => {
    const hash = location.hash.replace("#", "") as TabId;
    return TABS.includes(hash) ? hash : "data";
  });
  const selectTab = (next: TabId) => {
    setTab(next);
    history.replaceState(null, "", `#${next}`);
  };

  const showError = useCallback((message: string) => setError(message), []);

  const loadJobs = useCallback(() => {
    api
      .listJobs()
      .then((r) => {
        setJobs(r.items);
        setJobsVersion((v) => v + 1);
      })
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
    api
      .listProjects()
      .then((r) => {
        setProjects(r.items);
        setCurrentProject(r.current);
      })
      .catch((e) => showError(errorText(t, e)));
    loadJobs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reloadKey]);

  const onProjectsChanged = (items: ProjectSummary[], current: string) => {
    setProjects(items);
    setCurrentProject(current);
    setProject(null);
    setReloadKey((k) => k + 1);
  };

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
            <Typography variant="subtitle1" fontWeight={600} color="primary" sx={{ mr: 2, display: { xs: "none", lg: "block" } }}>
              „{project.wake_word}“
            </Typography>
          )}
          <SystemChip />
          {projects.length > 0 && (
            <Box sx={{ mr: 1 }}>
              <ProjectSelector projects={projects} current={currentProject} disabled={running} onChanged={onProjectsChanged} onError={showError} />
            </Box>
          )}
          <Select size="small" value={lang} onChange={(e) => setLang(e.target.value as Lang)} aria-label={t("app.language")} sx={{ minWidth: 90 }}>
            <MenuItem value="cs">Čeština</MenuItem>
            <MenuItem value="en">English</MenuItem>
          </Select>
        </Toolbar>
        <Tabs value={tab} onChange={(_, v) => selectTab(v as TabId)} variant="scrollable" allowScrollButtonsMobile sx={{ px: 1 }}>
          <Tab value="data" label={t("tabs.data")} />
          <Tab
            value="train"
            label={
              <Badge color="info" variant="dot" invisible={!running}>
                {t("tabs.train")}
              </Badge>
            }
          />
          <Tab value="test" label={t("tabs.test")} />
          <Tab value="deploy" label={t("tabs.deploy")} />
        </Tabs>
      </AppBar>

      <Container maxWidth="md" sx={{ py: 3 }}>
        <Stack spacing={3}>
          {tab === "data" && project && defaults && <ConfigCard project={project} defaults={defaults} disabled={running} onSaved={setProject} onError={showError} />}
          {tab === "data" && project && <RecorderCard key={project.id} wakeWord={project.wake_word} durationS={project.sample_duration_s} disabled={running} onCountsChange={setCounts} onError={showError} />}
          {tab === "data" && <DatasetsCard disabled={running} onError={showError} />}
          {tab === "train" && <TrainingCard state={state} log={log} connected={connected} positiveCount={counts.positive} wakeWord={project?.wake_word ?? ""} onError={showError} onFinished={loadJobs} />}
          {tab === "train" && <JobsCard jobs={jobs} disabled={running} onChanged={loadJobs} onError={showError} />}
          {tab === "test" && <TestCard jobs={jobs} wakeWord={project?.wake_word ?? ""} disabled={running} onError={showError} onInfo={setInfo} />}
          {tab === "deploy" && project && <DeployCard projectId={project.id} jobsVersion={jobsVersion} onError={showError} />}
          <Typography variant="caption" color="text.secondary" textAlign="center">
            {t("app.footer")}
          </Typography>
        </Stack>
      </Container>

      <Snackbar open={!!info} autoHideDuration={6000} onClose={() => setInfo(null)} anchorOrigin={{ vertical: "bottom", horizontal: "center" }}>
        <Alert severity="success" onClose={() => setInfo(null)} variant="filled">
          {info}
        </Alert>
      </Snackbar>

      <Snackbar open={!!error} autoHideDuration={8000} onClose={() => setError(null)} anchorOrigin={{ vertical: "bottom", horizontal: "center" }}>
        <Alert severity="error" onClose={() => setError(null)} variant="filled">
          {error}
        </Alert>
      </Snackbar>
    </Box>
  );
}
