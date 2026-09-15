export const panelClass =
  'rounded-3xl border border-white/10 bg-slate-900/70 p-5 shadow-2xl shadow-black/20 backdrop-blur-xl sm:p-7'

export const inputClass =
  'w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3.5 py-3 text-sm text-slate-100 outline-none transition placeholder:text-slate-600 hover:border-slate-600 focus:border-emerald-400 focus:ring-4 focus:ring-emerald-400/10 disabled:cursor-not-allowed disabled:opacity-60'

export const primaryButtonClass =
  'inline-flex min-h-11 items-center justify-center gap-2 rounded-xl bg-emerald-400 px-4 py-2.5 text-sm font-bold text-slate-950 shadow-lg shadow-emerald-950/30 transition hover:-translate-y-0.5 hover:bg-emerald-300 focus:outline-none focus:ring-4 focus:ring-emerald-400/20 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:translate-y-0'

export const secondaryButtonClass =
  'inline-flex min-h-10 items-center justify-center gap-2 rounded-xl border border-slate-700 bg-slate-800/70 px-4 py-2 text-sm font-semibold text-slate-200 transition hover:border-slate-600 hover:bg-slate-700 focus:outline-none focus:ring-4 focus:ring-slate-500/20 disabled:cursor-not-allowed disabled:opacity-50'

export const labelClass =
  'text-xs font-semibold uppercase tracking-wider text-slate-400'

export const alertErrorClass =
  'mt-4 rounded-xl border border-rose-400/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200'

export const alertSuccessClass =
  'mt-4 rounded-xl border border-emerald-400/20 bg-emerald-400/10 px-4 py-3 text-sm text-emerald-100'

export const statusBaseClass =
  'inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-bold capitalize'

export function statusTone(value: string): string {
  if (['healthy', 'stable', 'promoted', 'ready', 'idle'].includes(value)) {
    return 'border-emerald-400/20 bg-emerald-400/10 text-emerald-300'
  }
  if (['warning', 'insufficient_data', 'in progress'].includes(value)) {
    return 'border-amber-400/20 bg-amber-400/10 text-amber-300'
  }
  if (['critical', 'unhealthy', 'failed', 'degraded'].includes(value)) {
    return 'border-rose-400/20 bg-rose-400/10 text-rose-300'
  }
  return 'border-slate-600 bg-slate-800 text-slate-300'
}
