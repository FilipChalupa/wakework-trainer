import { Box, Checkbox, Chip, IconButton, Stack, Tooltip, Typography, useTheme } from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import DeleteIcon from "@mui/icons-material/Delete";
import type { Recording } from "../api";
import { useI18n, type TKey } from "../i18n";

type Props = {
  items: Recording[];
  compact: boolean;
  disabled: boolean;
  selected: Set<string>;
  playingId: string | null;
  playingProgress?: number;
  onToggleSelect: (id: string) => void;
  onTogglePlay: (rec: Recording) => void;
  onDelete: (rec: Recording) => void;
  onClearReview: (rec: Recording) => void;
};

/** Newest-first list of recordings with mini waveforms, quality/tag/contributor chips and actions. */
export function RecordingList({ items, compact, disabled, selected, playingId, playingProgress, onToggleSelect, onTogglePlay, onDelete, onClearReview }: Props) {
  const { t } = useI18n();
  const theme = useTheme();
  const list = items;
  return (
          <Box sx={{ maxHeight: 360, overflow: "auto", border: 1, borderColor: "divider", borderRadius: 2 }}>
            {list.length === 0 && (
              <Typography sx={{ p: 2 }} color="text.secondary">
                {t("rec.empty")}
              </Typography>
            )}
            {[...list].reverse().map((rec, idx) => {
              const isPlaying = playingId === rec.id;
              return (
                <Stack
                  key={rec.id}
                  direction="row"
                  spacing={1}
                  alignItems="center"
                  sx={{
                    px: 1,
                    py: 0.5,
                    borderBottom: idx < list.length - 1 ? 1 : 0,
                    borderColor: "divider",
                    bgcolor: isPlaying ? "action.selected" : selected.has(rec.id) ? "action.hover" : "transparent",
                  }}
                >
                  {!compact && <Checkbox size="small" checked={selected.has(rec.id)} onChange={() => onToggleSelect(rec.id)} disabled={disabled} />}
                  <IconButton size="small" onClick={() => onTogglePlay(rec)} color={isPlaying ? "primary" : "default"}>
                    {isPlaying ? <StopIcon /> : <PlayArrowIcon />}
                  </IconButton>
                  <Box sx={{ width: 140, cursor: "pointer" }} onClick={() => onTogglePlay(rec)}>
                    <Waveform peaks={rec.peaks} color={isPlaying ? theme.palette.primary.main : theme.palette.text.secondary} height={28} progress={isPlaying ? playingProgress : undefined} />
                  </Box>
                  <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Typography variant="body2" noWrap>
                      #{list.length - idx} · {rec.duration.toFixed(2)} s · {new Date(rec.created).toLocaleTimeString()}
                      {rec.contributor && !compact && (
                        <Chip size="small" variant="outlined" label={t("rec.by", { name: rec.contributor })} sx={{ ml: 1, height: 18, fontSize: 11 }} />
                      )}
                      {rec.tag && rec.tag !== "normal" && (
                        <Chip size="small" color="secondary" variant="outlined" label={t(`tag.${rec.tag}` as TKey)} sx={{ ml: 1, height: 18, fontSize: 11 }} />
                      )}
                      {rec.review && !compact && (
                        <Tooltip title={t("rec.reviewHint")}>
                          <Chip size="small" color="warning" label={t("rec.review")} onClick={() => onClearReview(rec)} sx={{ ml: 1, height: 18, fontSize: 11 }} />
                        </Tooltip>
                      )}
                    </Typography>
                    <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                      {rec.quality.issues.map((issue) => (
                        <Tooltip key={issue} title={t(`rec.issueHint.${issue}` as TKey)}>
                          <Chip size="small" color="warning" variant="outlined" label={t(`rec.issue.${issue}` as TKey)} sx={{ height: 20, fontSize: 11 }} />
                        </Tooltip>
                      ))}
                      {rec.quality.issues.length === 0 && (
                        <Typography variant="caption" color="text.secondary">
                          {t("rec.peakInfo", { peak: Math.round((rec.quality.peak ?? 0) * 100), db: rec.quality.rms_db ?? 0 })}
                        </Typography>
                      )}
                    </Stack>
                  </Box>
                  <Tooltip title={t("rec.deleteTooltip")}>
                    <span>
                      <IconButton size="small" onClick={() => onDelete(rec)} disabled={disabled}>
                        <DeleteIcon />
                      </IconButton>
                    </span>
                  </Tooltip>
                </Stack>
              );
            })}
          </Box>
  );
}

export function Waveform({ peaks, color, height, progress }: { peaks: number[]; color: string; height: number; progress?: number }) {
  const max = Math.max(0.05, ...peaks);
  return (
    <Box sx={{ position: "relative", display: "flex", alignItems: "center", gap: "1px", height }}>
      {peaks.map((p, i) => {
        const played = progress !== undefined && i / peaks.length <= progress;
        return <Box key={i} sx={{ flex: 1, height: `${Math.max(6, (p / max) * 100)}%`, bgcolor: color, borderRadius: 1, opacity: played ? 1 : progress !== undefined ? 0.35 : 0.75 }} />;
      })}
    </Box>
  );
}
