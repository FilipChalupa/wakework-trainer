import { useEffect, useState } from "react";
import { Chip, Tooltip } from "@mui/material";
import MemoryIcon from "@mui/icons-material/Memory";
import DeveloperBoardIcon from "@mui/icons-material/DeveloperBoard";
import { api, type SystemInfo } from "../api";
import { useI18n } from "../i18n";

/** Header chip showing whether GPU acceleration is available to the trainer. */
export function SystemChip() {
  const { t } = useI18n();
  const [info, setInfo] = useState<SystemInfo | null>(null);

  useEffect(() => {
    const load = () => api.system().then(setInfo).catch(() => undefined);
    load();
    const timer = setInterval(load, 30000);
    return () => clearInterval(timer);
  }, []);

  if (!info) return null;
  const gpuOk = info.gpu_available && info.tensorflow_cuda;
  const last = info.last_training ? t("sys.lastTraining", { device: info.last_training.gpu ? info.last_training.devices.join(", ") : "CPU" }) : "";
  let tip: string;
  if (info.gpu && info.tensorflow_cuda) {
    tip = t("sys.gpuTip", { name: info.gpu.name, used: info.gpu.memory_used_mb, total: info.gpu.memory_total_mb, driver: info.gpu.driver, cuda: t("sys.yes") });
  } else if (info.gpu) {
    tip = t("sys.cudaMissing");
  } else {
    tip = t("sys.cpuTip", { cpus: info.cpu_count ?? "?" });
  }
  return (
    <>
      <Tooltip title={info.update_available ? t("sys.update", { latest: info.latest_version ?? "" }) : t("sys.upToDate")}>
        <Chip
          size="small"
          variant="outlined"
          color={info.update_available ? "info" : "default"}
          label={t("sys.version", { version: info.version })}
          component="a"
          href={info.releases_url}
          target="_blank"
          clickable
          sx={{ mr: 1, display: { xs: "none", md: "inline-flex" } }}
        />
      </Tooltip>
    <Tooltip title={`${tip} ${last}`.trim()}>
      <Chip
        size="small"
        icon={gpuOk ? <DeveloperBoardIcon /> : <MemoryIcon />}
        color={gpuOk ? "success" : info.gpu ? "warning" : "default"}
        variant="outlined"
        label={info.gpu ? t("sys.gpu", { name: info.gpu.name.replace(/^NVIDIA\s+/, "") }) : t("sys.cpu")}
        sx={{ mr: 1, display: { xs: "none", sm: "inline-flex" } }}
      />
    </Tooltip>
    </>
  );
}
