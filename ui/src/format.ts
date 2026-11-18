export const pct = (x: number | null | undefined, digits = 1) =>
  x == null || !Number.isFinite(x) ? "n/a" : `${(100 * x).toFixed(digits)}%`;

export const int = (x: number | null | undefined) => (x == null ? "n/a" : x.toLocaleString("en-GB"));

export function duration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h) return `${h} h ${String(m).padStart(2, "0")} min`;
  if (m) return `${m} min`;
  return `${s} s`;
}

export function ms(x: number): string {
  return x >= 1000 ? `${(x / 1000).toFixed(2)} s` : `${Math.round(x)} ms`;
}
