import { useEffect, useRef, useState } from "react";
import { Alert, Box, Button, Card, CardContent, CardHeader, Chip, Collapse, Stack, TextField, Typography } from "@mui/material";
import RocketLaunchIcon from "@mui/icons-material/RocketLaunch";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import DeveloperBoardIcon from "@mui/icons-material/DeveloperBoard";
import { api, type DeviceEvent, type PublicUrls } from "../api";
import { errorText, useI18n } from "../i18n";

type Props = { projectId: string; jobsVersion: number; onError: (m: string) => void };

/** ESPHome deployment: manifest URL, YAML snippet, device event timeline and ESPHome version check. */
export function DeployCard({ projectId, jobsVersion, onError }: Props) {
  const { t } = useI18n();
  const [urls, setUrls] = useState<PublicUrls | null>(null);
  const [showSnippet, setShowSnippet] = useState(false);
  const [copied, setCopied] = useState(false);
  const [events, setEvents] = useState<DeviceEvent[]>([]);
  const [outdated, setOutdated] = useState<string[]>([]);
  const [minVersion, setMinVersion] = useState("2024.7.0");
  const since = useRef<string | undefined>(undefined);

  useEffect(() => {
    api.publicUrls(projectId).then(setUrls).catch((e) => onError(errorText(t, e)));
  }, [projectId, jobsVersion, onError, t]);

  useEffect(() => {
    since.current = undefined;
    setEvents([]);
    const poll = () =>
      api
        .deviceEvents(since.current)
        .then((r) => {
          if (r.items.length) setEvents((prev) => [...r.items.slice().reverse(), ...prev].slice(0, 50));
          since.current = r.now;
          setOutdated(r.outdated_versions);
          setMinVersion(r.minimum_esphome_version);
        })
        .catch(() => undefined);
    poll();
    const timer = setInterval(poll, 3000);
    return () => clearInterval(timer);
  }, [projectId]);

  const wyoming = urls?.target === "wyoming";

  const copy = async () => {
    if (!urls) return;
    try {
      await navigator.clipboard.writeText(urls.snippet);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setShowSnippet(true);
    }
  };

  return (
    <Card>
      <CardHeader avatar={<RocketLaunchIcon color="primary" />} title={wyoming ? t("deploy.titleWyoming") : t("deploy.title")} subheader={wyoming ? t("deploy.subtitleWyoming") : t("deploy.subtitle")} />
      <CardContent>
        <Stack spacing={2}>
          {!wyoming && (
            <Typography variant="body2" color="text.secondary">
              {t("deploy.help")}
            </Typography>
          )}
          {urls && !urls.has_model && <Alert severity="info">{t("deploy.noModel")}</Alert>}
          {urls && urls.target === "wyoming" && <Alert severity="info" variant="outlined">{t("deploy.wyomingHelp")}</Alert>}
          {urls && (
            <>
              <TextField size="small" label={urls.target === "wyoming" ? t("deploy.modelUrl") : t("deploy.manifest")} value={urls.target === "wyoming" ? urls.model_url : urls.manifest_url} fullWidth InputProps={{ readOnly: true }} onFocus={(e) => e.target.select()} />
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap alignItems="center">
                <Button variant="outlined" startIcon={<ContentCopyIcon />} onClick={copy}>
                  {copied ? t("share.copied") : wyoming ? t("deploy.copyWyoming") : t("deploy.copy")}
                </Button>
                <Button startIcon={showSnippet ? <ExpandLessIcon /> : <ExpandMoreIcon />} onClick={() => setShowSnippet((v) => !v)}>
                  {urls.target === "wyoming" ? t("deploy.snippetWyoming") : t("deploy.snippet")}
                </Button>
                {urls.target !== "wyoming" && <Chip size="small" variant="outlined" label={t("deploy.minVersion", { min: urls.minimum_esphome_version })} />}
              </Stack>
              <Collapse in={showSnippet}>
                <Box component="pre" sx={{ m: 0, p: 1.5, borderRadius: 2, bgcolor: (th) => (th.palette.mode === "dark" ? "#05080f" : "#0f172a"), color: "#cbd5e1", fontSize: 11, overflowX: "auto" }}>
                  {urls.snippet}
                </Box>
              </Collapse>
              {!wyoming && (
                <Typography variant="caption" color="text.secondary">
                  {t("deploy.https")}
                </Typography>
              )}
            </>
          )}

          {!wyoming && (
          <Box>
            <Stack direction="row" spacing={1} alignItems="center">
              <DeveloperBoardIcon fontSize="small" color="primary" />
              <Typography variant="subtitle2">{t("deploy.devices")}</Typography>
            </Stack>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              {t("deploy.devicesHelp")}
            </Typography>
            {outdated.length > 0 && (
              <Alert severity="warning" sx={{ mb: 1 }}>
                {t("deploy.outdated", { versions: outdated.join(", "), min: minVersion })}
              </Alert>
            )}
            {events.length === 0 ? (
              <Typography variant="caption" color="text.secondary">
                {t("deploy.noEvents")}
              </Typography>
            ) : (
              <Stack spacing={0.5} sx={{ maxHeight: 200, overflow: "auto" }}>
                {events.map((e, i) => (
                  <Stack key={`${e.at}-${i}`} direction="row" spacing={1} alignItems="center">
                    <Typography variant="caption" color="text.secondary" sx={{ minWidth: 80 }}>
                      {new Date(e.at).toLocaleTimeString()}
                    </Typography>
                    <Chip size="small" color="secondary" variant="outlined" label={t("monitor.source.device", { device: e.device })} />
                    <Typography variant="body2">{e.wake_word}</Typography>
                    {e.esphome_version && (
                      <Typography variant="caption" color="text.secondary">
                        ESPHome {e.esphome_version}
                      </Typography>
                    )}
                  </Stack>
                ))}
              </Stack>
            )}
          </Box>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
}
