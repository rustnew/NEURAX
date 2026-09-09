/**
 * Formatting shared by every runtime surface.
 *
 * These lived in `mockRuntime.ts`, and six views that contain nothing
 * simulated imported them from a file named "mock". That is not a cosmetic
 * complaint: when the training runtime lands, `mockRuntime` should be
 * deletable, and it could not be while half the Training workspace depended
 * on it for `formatBytes`.
 *
 * They are also the reason two panels can show the same number the same way.
 * A run reporting 1.49 GB and a prediction of 1.45 GB have to be comparable
 * at a glance, which they are not if one rounds to two decimals and the other
 * to none.
 */

const GB = 1024 ** 3;

/** Bytes, at the precision the magnitude deserves — two decimals under 10 GB,
 *  none above, because "23.00 GB" reads as false precision on a card size. */
export function formatBytes(n: number): string {
  if (!Number.isFinite(n)) return '—';
  if (n >= GB) return `${(n / GB).toFixed(n / GB >= 10 ? 0 : 2)} GB`;
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(0)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} kB`;
  return `${n} B`;
}

/** Thousands separated. Fixed to en-US rather than the viewer's locale: these
 *  sit beside monospace figures in tables that assume a known width. */
export function formatCount(n: number): string {
  return Number.isFinite(n) ? n.toLocaleString('en-US') : '—';
}

/** Hours as a human duration. Under an hour it is minutes, because "0.4 h"
 *  is a number nobody converts in their head. */
export function formatDuration(hours: number): string {
  if (!Number.isFinite(hours) || hours < 0) return '—';
  const h = Math.floor(hours);
  const m = Math.round((hours - h) * 60);
  return h > 0 ? `${h} h ${String(m).padStart(2, '0')}` : `${m} min`;
}
