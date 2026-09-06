import { useEffect, useState } from "react";
import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Card, CardContent, CardHeader, Chip, Divider, FormControlLabel, MenuItem, Stack, Switch, TextField, Typography } from "@mui/material";
import ShareIcon from "@mui/icons-material/Share";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import LinkOffIcon from "@mui/icons-material/LinkOff";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import TuneIcon from "@mui/icons-material/Tune";
import SaveIcon from "@mui/icons-material/Save";
import QRCode from "qrcode";
import QrCode2Icon from "@mui/icons-material/QrCode2";
import { api, type Contributor, type Project, type TrainingParams } from "../api";
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
  const [training, setTraining] = useState<TrainingParams>(project.training);
  const [webhook, setWebhook] = useState(project.webhook_url ?? "");
  const [target, setTarget] = useState(project.contributor_target ?? 10);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setWakeWord(project.wake_word);
    setTraining(project.training);
    setWebhook(project.webhook_url ?? "");
    setTarget(project.contributor_target ?? 10);
  }, [project]);

  const dirty =
    wakeWord !== project.wake_word ||
    webhook !== (project.webhook_url ?? "") ||
    target !== (project.contributor_target ?? 10) ||
    JSON.stringify(training) !== JSON.stringify(project.training);

  const save = async () => {
    setSaving(true);
    try {
      const res = await api.saveConfig({ wake_word: wakeWord, training, webhook_url: webhook, contributor_target: target });
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
          <TextField label={t("config.wakeWord")} value={wakeWord} onChange={(e) => setWakeWord(e.target.value)} placeholder="chaloupko" fullWidth disabled={disabled} helperText={t("config.wakeWordHelp")} />

          <TextField
            select
            label={t("config.platform")}
            value={training.target ?? "esphome"}
            onChange={(e) => setTraining({ ...training, target: e.target.value as TrainingParams["target"] })}
            disabled={disabled}
            helperText={t("config.platformHelp")}
          >
            <MenuItem value="esphome">{t("target.esphome")}</MenuItem>
            <MenuItem value="wyoming">{t("target.wyoming")}</MenuItem>
          </TextField>

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
              <FormControlLabel
                sx={{ mt: 1 }}
                control={<Switch checked={!!training.hard_negatives} onChange={(e) => setTraining({ ...training, hard_negatives: e.target.checked })} disabled={disabled} />}
                label={
                  <Box>
                    <Typography variant="body2">{t("config.f.hard_negatives")}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {t("config.h.hard_negatives")}
                    </Typography>
                  </Box>
                }
              />
              <Box sx={{ display: "grid", gap: 2, gridTemplateColumns: { xs: "1fr", sm: "2fr 1fr" }, mt: 2 }}>
                <TextField size="small" label={t("config.webhook")} value={webhook} onChange={(e) => setWebhook(e.target.value)} helperText={t("config.webhookHelp")} disabled={disabled} placeholder="https://homeassistant.local:8123/api/webhook/…" />
                <TextField size="small" type="number" label={t("config.target")} value={target} onChange={(e) => setTarget(Math.max(1, Number(e.target.value)))} inputProps={{ min: 1 }} disabled={disabled} />
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

          <Divider />
          <ShareSection project={project} onSaved={onSaved} onError={onError} />
        </Stack>
      </CardContent>
    </Card>
  );
}

function ShareSection({ project, onSaved, onError }: { project: Project; onSaved: (p: Project) => void; onError: (m: string) => void }) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState(false);
  const [qr, setQr] = useState<string | null>(null);
  const [showQr, setShowQr] = useState(false);
  const [contributors, setContributors] = useState<Contributor[]>([]);
  const link = project.share_token ? `${location.origin}/contribute?token=${project.share_token}` : null;

  useEffect(() => {
    if (!link) {
      setQr(null);
      return;
    }
    QRCode.toDataURL(link, { width: 220, margin: 1 }).then(setQr).catch(() => setQr(null));
  }, [link]);

  useEffect(() => {
    const load = () => api.contributors().then((r) => setContributors(r.items)).catch(() => undefined);
    load();
    const timer = setInterval(load, 15000);
    return () => clearInterval(timer);
  }, [project.id]);

  const toggle = async (enabled: boolean) => {
    setBusy(true);
    try {
      const res = await api.setShare(project.id, enabled);
      onSaved({ ...project, share_token: res.share_token });
    } catch (e) {
      onError(errorText(t, e));
    } finally {
      setBusy(false);
    }
  };

  const copy = async () => {
    if (!link) return;
    try {
      await navigator.clipboard.writeText(link);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      onError(link);
    }
  };

  const totalPositive = contributors.reduce((a, c) => a + c.positive, 0);
  const top = contributors[0];
  const imbalance = top && totalPositive >= 10 && top.positive / totalPositive >= 0.8 ? { share: Math.round((top.positive / totalPositive) * 100), name: top.name === "owner" ? t("share.owner") : top.name } : null;

  return (
    <Box>
      {imbalance && (
        <Alert severity="warning" variant="outlined" sx={{ mb: 1.5 }}>
          {t("share.imbalance", imbalance)}
        </Alert>
      )}
      <Stack direction="row" spacing={1} alignItems="center">
        <ShareIcon fontSize="small" color="primary" />
        <Typography variant="subtitle2">{t("share.title")}</Typography>
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5, mb: 1 }}>
        {t("share.help")} {t("share.https")}
      </Typography>
      {link ? (
        <Stack spacing={1}>
          <TextField size="small" value={link} fullWidth InputProps={{ readOnly: true }} onFocus={(e) => e.target.select()} />
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            <Button variant="outlined" startIcon={<ContentCopyIcon />} onClick={copy}>
              {copied ? t("share.copied") : t("share.copy")}
            </Button>
            <Button variant="outlined" startIcon={<QrCode2Icon />} onClick={() => setShowQr((v) => !v)} disabled={!qr}>
              {t("share.qr")}
            </Button>
            <Button color="error" startIcon={<LinkOffIcon />} onClick={() => toggle(false)} disabled={busy}>
              {t("share.disable")}
            </Button>
          </Stack>
        </Stack>
      ) : (
        <Button variant="outlined" startIcon={<ShareIcon />} onClick={() => toggle(true)} disabled={busy}>
          {t("share.enable")}
        </Button>
      )}
      {showQr && qr && link && (
        <Box sx={{ mt: 1 }}>
          <img src={qr} alt="QR" width={220} height={220} style={{ borderRadius: 8, background: "#fff", padding: 4 }} />
        </Box>
      )}
      {contributors.length > 0 && (
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mt: 1.5 }} alignItems="center">
          <Typography variant="caption" color="text.secondary">
            {t("share.contributors")}:
          </Typography>
          {contributors.map((c) => (
            <Chip
              key={c.name}
              size="small"
              variant="outlined"
              color={c.name !== "owner" && c.positive >= (project.contributor_target ?? 10) ? "success" : "default"}
              label={`${c.name === "owner" ? t("share.owner") : c.name}: ${c.positive}${c.negative ? ` (+${c.negative})` : ""}`}
            />
          ))}
        </Stack>
      )}
    </Box>
  );
}
