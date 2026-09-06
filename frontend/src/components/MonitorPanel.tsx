import { Box, Button, Chip, IconButton, Stack, Tooltip, Typography } from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import DeleteIcon from "@mui/icons-material/Delete";
import AddCircleOutlineIcon from "@mui/icons-material/AddCircleOutline";
import type { MonitorItem } from "../api";
import { useI18n } from "../i18n";

type Props = {
  saved: MonitorItem[];
  disabled: boolean;
  onPlay: (item: MonitorItem) => void;
  onToNegative: (item: MonitorItem) => void;
  onDelete: (item: MonitorItem) => void;
  onClear: () => void;
  onRetrain: () => void;
};

/** Saved activations from the long-run false-accept test with play / adopt-as-negative / delete actions. */
export function MonitorPanel({ saved, disabled, onPlay, onToNegative, onDelete, onClear, onRetrain }: Props) {
  const { t } = useI18n();
  return (
    <Box sx={{ p: 1.5, border: 1, borderColor: "divider", borderRadius: 2 }}>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
        <Typography variant="subtitle2" sx={{ flex: 1 }}>
          {t("monitor.title")}
        </Typography>
        {saved.length > 0 && (
          <>
            <Button size="small" variant="contained" onClick={onRetrain} disabled={disabled}>
              {t("monitor.retrain")}
            </Button>
            <Button size="small" color="error" onClick={onClear}>
              {t("monitor.clear")}
            </Button>
          </>
        )}
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
        {t("monitor.help")}
      </Typography>
      {saved.length === 0 ? (
        <Typography variant="caption" color="text.secondary">
          {t("monitor.empty")}
        </Typography>
      ) : (
        <Stack spacing={0.5} sx={{ maxHeight: 240, overflow: "auto" }}>
          {saved.map((item) => (
            <Stack key={item.id} direction="row" spacing={1} alignItems="center">
              <IconButton size="small" onClick={() => onPlay(item)}>
                <PlayArrowIcon />
              </IconButton>
              <Typography variant="body2" sx={{ flex: 1 }} noWrap>
                {new Date(item.created).toLocaleString()} · {item.duration.toFixed(1)} s
              </Typography>
              <Chip size="small" variant="outlined" label={t("monitor.source.browser")} />
              <Tooltip title={t("monitor.toNegative")}>
                <IconButton size="small" color="primary" onClick={() => onToNegative(item)}>
                  <AddCircleOutlineIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title={t("monitor.delete")}>
                <IconButton size="small" onClick={() => onDelete(item)}>
                  <DeleteIcon />
                </IconButton>
              </Tooltip>
            </Stack>
          ))}
        </Stack>
      )}
    </Box>
  );
}
