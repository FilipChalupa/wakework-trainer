import { Alert, Box, Button, Chip, LinearProgress, Stack, Table, TableBody, TableCell, TableHead, TableRow, Typography } from "@mui/material";
import FlagIcon from "@mui/icons-material/Flag";
import type { Evaluation } from "../api";
import { useI18n, type TKey } from "../i18n";

type Props = { evaluation: Evaluation; flagged: number | null; onFlagOutliers: () => void };

/** Per-recording results of a model evaluation with the per-mode summary and outlier flagging. */
export function EvaluationTable({ evaluation, flagged, onFlagOutliers }: Props) {
  const { t } = useI18n();
  return (
    <Box>
      <Typography variant="body2" gutterBottom>
        {t("test.evalSummary", {
          pd: evaluation.summary.positive_detected,
          pt: evaluation.summary.positive_total,
          nt: evaluation.summary.negative_triggered,
          ntot: evaluation.summary.negative_total,
          cutoff: evaluation.cutoff.toFixed(2),
        })}
      </Typography>
      {evaluation.summary.outliers > 0 && (
        <Alert
          severity="warning"
          variant="outlined"
          sx={{ mb: 1 }}
          action={
            <Button color="inherit" size="small" startIcon={<FlagIcon />} onClick={onFlagOutliers} disabled={flagged !== null}>
              {t("eval.flag")}
            </Button>
          }
        >
          {t("eval.outliers", { n: evaluation.summary.outliers })} – {t("eval.outliersHelp", { median: Math.round(evaluation.summary.median_positive * 100) })}
          {flagged !== null && ` ${t("eval.flagged", { n: flagged })}`}
        </Alert>
      )}
      {Object.keys(evaluation.summary.by_tag ?? {}).length > 1 && (
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mb: 1 }} alignItems="center">
          <Typography variant="caption" color="text.secondary">
            {t("test.byTag")}:
          </Typography>
          {Object.entries(evaluation.summary.by_tag).map(([tg, v]) => (
            <Chip key={tg} size="small" variant="outlined" color={v.detected === v.total ? "success" : v.detected === 0 ? "error" : "warning"} label={`${t(`tag.${tg}` as TKey)}: ${v.detected} / ${v.total}`} />
          ))}
        </Stack>
      )}
      <Box sx={{ maxHeight: 320, overflow: "auto" }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell>{t("test.t.recording")}</TableCell>
              <TableCell>{t("test.t.kind")}</TableCell>
              <TableCell>{t("test.t.max")}</TableCell>
              <TableCell>{t("test.t.result")}</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {evaluation.items.map((item) => {
              const ok = item.kind === "positive" ? item.detections > 0 : item.detections === 0;
              const label = item.kind === "positive" ? (ok ? t("test.r.ok") : t("test.r.missed")) : ok ? t("test.r.ok") : t("test.r.false");
              const p = item.max_probability ?? 0;
              return (
                <TableRow key={item.id} hover selected={item.outlier}>
                  <TableCell>
                    <Typography variant="body2" component="a" href={item.url} target="_blank" rel="noreferrer" sx={{ color: "inherit" }}>
                      {item.id}
                    </Typography>
                    {item.outlier && <Chip size="small" color="warning" label={t("rec.review")} sx={{ ml: 1, height: 18, fontSize: 11 }} />}
                  </TableCell>
                  <TableCell>
                    {t(item.kind === "positive" ? "test.kind.positive" : "test.kind.negative")}
                    {item.tag && item.tag !== "normal" ? ` · ${t(`tag.${item.tag}` as TKey)}` : ""}
                  </TableCell>
                  <TableCell sx={{ minWidth: 160 }}>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <LinearProgress variant="determinate" value={p * 100} sx={{ flex: 1, height: 8, borderRadius: 4 }} color={p >= evaluation.cutoff ? "success" : "inherit"} />
                      <Typography variant="caption">{(p * 100).toFixed(0)} %</Typography>
                    </Stack>
                  </TableCell>
                  <TableCell>
                    <Chip size="small" label={item.error ?? label} color={item.error ? "default" : ok ? "success" : "error"} variant={ok ? "outlined" : "filled"} />
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Box>
    </Box>
  );
}
