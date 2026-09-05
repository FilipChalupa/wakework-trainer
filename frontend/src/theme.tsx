import { useMemo, type ReactNode } from "react";
import { createTheme, CssBaseline, ThemeProvider, useMediaQuery } from "@mui/material";

/** Follows the user agent's colour scheme (prefers-color-scheme) and updates live. */
export function AppThemeProvider({ children }: { children: ReactNode }) {
  const prefersDark = useMediaQuery("(prefers-color-scheme: dark)");
  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: prefersDark ? "dark" : "light",
          primary: { main: prefersDark ? "#4ade80" : "#15803d" },
          secondary: { main: prefersDark ? "#7dd3fc" : "#0369a1" },
          background: prefersDark
            ? { default: "#0b1220", paper: "#111a2e" }
            : { default: "#f3f5f9", paper: "#ffffff" },
        },
        shape: { borderRadius: 12 },
        components: {
          MuiCard: { defaultProps: { elevation: 0 }, styleOverrides: { root: { border: "1px solid", borderColor: prefersDark ? "#1f2a44" : "#e2e8f0" } } },
          MuiButton: { defaultProps: { disableElevation: true } },
        },
      }),
    [prefersDark],
  );
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {children}
    </ThemeProvider>
  );
}
