import { useEffect, useMemo, useState } from "react";
import { Alert, AppBar, Box, Button, Card, CardContent, Container, LinearProgress, MenuItem, Select, Snackbar, Stack, TextField, Toolbar, Typography } from "@mui/material";
import RecordVoiceOverIcon from "@mui/icons-material/RecordVoiceOver";
import { contributeClient, type ContributeInfo } from "./api";
import { RecorderCard } from "./components/RecorderCard";
import { errorText, useI18n, type Lang } from "./i18n";

const NAME_KEY = "wakeword-trainer.contributor";

/** Record-only page reachable via a shared link: /contribute?token=… */
export function ContributePage() {
  const { t, lang, setLang } = useI18n();
  const token = new URLSearchParams(location.search).get("token") ?? "";
  const [name, setName] = useState<string>(() => {
    try {
      return localStorage.getItem(NAME_KEY) ?? "";
    } catch {
      return "";
    }
  });
  const [draft, setDraft] = useState(name);
  const [info, setInfo] = useState<ContributeInfo | null>(null);
  const [invalid, setInvalid] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [count, setCount] = useState<number | null>(null);
  const target = info?.contributor_target ?? 10;
  const client = useMemo(() => (name ? contributeClient(token, name) : null), [token, name]);

  useEffect(() => {
    contributeClient(token, "x")
      .info()
      .then(setInfo)
      .catch(() => setInvalid(true));
  }, [token]);

  const start = () => {
    const n = draft.trim();
    if (!n) return;
    try {
      localStorage.setItem(NAME_KEY, n);
    } catch {
      /* ignore */
    }
    setName(n);
  };

  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "background.default" }}>
      <AppBar position="sticky" color="default" elevation={0} sx={{ borderBottom: 1, borderColor: "divider", bgcolor: "background.paper" }}>
        <Toolbar>
          <RecordVoiceOverIcon color="primary" sx={{ mr: 1.5 }} />
          <Typography variant="h6" component="h1" sx={{ flex: 1 }}>
            {t("contrib.title")}
          </Typography>
          <Select size="small" value={lang} onChange={(e) => setLang(e.target.value as Lang)} sx={{ minWidth: 90 }}>
            <MenuItem value="cs">Čeština</MenuItem>
            <MenuItem value="en">English</MenuItem>
          </Select>
        </Toolbar>
      </AppBar>
      <Container maxWidth="sm" sx={{ py: { xs: 1.5, sm: 3 }, px: { xs: 1, sm: 3 } }}>
        <Stack spacing={3}>
          {invalid && <Alert severity="error">{t("contrib.invalid")}</Alert>}
          {info && !name && (
            <Card>
              <CardContent>
                <Stack spacing={2}>
                  <Typography>{t("contrib.intro", { word: info.wake_word, project: info.project })}</Typography>
                  <TextField label={t("contrib.name")} value={draft} onChange={(e) => setDraft(e.target.value)} onKeyDown={(e) => e.key === "Enter" && start()} autoFocus />
                  <Button variant="contained" onClick={start} disabled={!draft.trim()}>
                    {t("contrib.start")}
                  </Button>
                </Stack>
              </CardContent>
            </Card>
          )}
          {info && name && client && (
            <>
              <Alert severity={count !== null && count >= target ? "success" : "info"} action={<Button color="inherit" size="small" onClick={() => setName("")}>{t("contrib.change")}</Button>}>
                {t("contrib.intro", { word: info.wake_word, project: info.project })}
                {count !== null && (
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="body2" fontWeight={600}>
                      {t("contrib.progress", { n: count, target })} {count >= target && `· ${t("contrib.done")}`}
                    </Typography>
                    <LinearProgress variant="determinate" value={Math.min(100, (count / target) * 100)} sx={{ height: 8, borderRadius: 4, mt: 0.5 }} color={count >= target ? "success" : "primary"} />
                  </Box>
                )}
              </Alert>
              <RecorderCard
                wakeWord={info.wake_word}
                durationS={info.sample_duration_s}
                disabled={false}
                client={client}
                compact
                title={`${t("contrib.title")} · ${name}`}
                onCountsChange={(c) => setCount(c.positive)}
                onError={(m) => setError(errorText(t, new Error(m)))}
              />
            </>
          )}
        </Stack>
      </Container>
      <Snackbar open={!!error} autoHideDuration={8000} onClose={() => setError(null)}>
        <Alert severity="error" onClose={() => setError(null)} variant="filled">
          {error}
        </Alert>
      </Snackbar>
    </Box>
  );
}
