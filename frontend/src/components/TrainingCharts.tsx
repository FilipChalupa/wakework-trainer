import { Box, Stack, Typography, useTheme } from "@mui/material";
import { LineChart } from "@mui/x-charts/LineChart";
import type { RocPoint, ValidationEntry } from "../api";
import { useI18n } from "../i18n";

type Props = { validation: ValidationEntry[]; roc?: RocPoint[] };

/** Validation loss, validation rates (same 0–1 scale) and the test-set ROC curve – one axis per chart. */
export function TrainingCharts({ validation, roc }: Props) {
  const { t } = useI18n();
  const theme = useTheme();
  const steps = validation.map((v) => v.step);
  const colors = [theme.palette.primary.main, theme.palette.secondary.main, theme.palette.mode === "dark" ? "#fbbf24" : "#b45309"];
  const rocSorted = roc && roc.length > 1 ? [...roc].sort((a, b) => a.faph - b.faph) : null;
  const common = { height: 220, margin: { left: 8, right: 16, top: 24, bottom: 8 }, grid: { horizontal: true } as const };

  return (
    <Stack spacing={1}>
      <Box sx={{ display: "grid", gap: 2, gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" } }}>
        <Box>
          <Typography variant="subtitle2">{t("charts.loss")}</Typography>
          <LineChart
            {...common}
            colors={[colors[0]]}
            xAxis={[{ data: steps, label: t("charts.stepAxis"), valueFormatter: (v: number) => String(v) }]}
            yAxis={[{ min: 0 }]}
            series={[{ data: validation.map((v) => v.loss), label: t("charts.loss"), showMark: true, curve: "monotoneX" }]}
            hideLegend
          />
        </Box>
        <Box>
          <Typography variant="subtitle2">{t("charts.rates")}</Typography>
          <LineChart
            {...common}
            colors={colors}
            xAxis={[{ data: steps, label: t("charts.stepAxis"), valueFormatter: (v: number) => String(v) }]}
            yAxis={[{ min: 0, max: 1, valueFormatter: (v: number) => `${Math.round(v * 100)} %` }]}
            series={[
              { data: validation.map((v) => v.recall_at_no_faph), label: t("charts.recallNoFa"), showMark: true, curve: "monotoneX" },
              { data: validation.map((v) => v.recall), label: t("charts.recall"), showMark: true, curve: "monotoneX" },
              { data: validation.map((v) => v.accuracy), label: t("charts.accuracy"), showMark: true, curve: "monotoneX" },
            ]}
          />
        </Box>
      </Box>
      {rocSorted && (
        <Box>
          <Typography variant="subtitle2">{t("charts.roc")}</Typography>
          <LineChart
            {...common}
            colors={[colors[1]]}
            xAxis={[{ data: rocSorted.map((p) => p.faph), label: t("charts.faph"), min: 0, valueFormatter: (v: number) => v.toFixed(2) }]}
            yAxis={[{ min: 0, max: 1, valueFormatter: (v: number) => `${Math.round(v * 100)} %` }]}
            series={[{ data: rocSorted.map((p) => p.frr), label: t("charts.frr"), showMark: true, curve: "linear", valueFormatter: (v) => (v === null ? "" : `${Math.round(v * 100)} %`) }]}
            hideLegend
          />
        </Box>
      )}
    </Stack>
  );
}
