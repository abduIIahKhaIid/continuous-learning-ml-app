import type { PredictionFeedbackSummary } from '../../types/prediction'
import type { TrainingStatus } from '../../types/training'
import { panelClass, statusBaseClass, statusTone } from '../../lib/ui'

interface SystemOverviewProps {
  training: TrainingStatus | null
  feedback: PredictionFeedbackSummary | null
}

function percent(value: number, target: number): number {
  if (target <= 0) return 0
  return Math.min(100, Math.max(0, (value / target) * 100))
}

function Progress({
  label,
  value,
  target,
  helper,
  accent = 'bg-emerald-300',
}: {
  label: string
  value: number
  target: number
  helper: string
  accent?: string
}) {
  const progress = percent(value, target)

  return (
    <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs font-semibold text-slate-300">{label}</span>
        <span className="text-xs font-bold text-white">{value} / {target}</span>
      </div>
      <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-800" aria-label={`${label}: ${progress.toFixed(0)}%`} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress)}>
        <div className={`h-full rounded-full ${accent} transition-[width] duration-500 motion-reduce:transition-none`} style={{ width: `${progress}%` }} />
      </div>
      <p className="mt-2 text-xs leading-5 text-slate-500">{helper}</p>
    </div>
  )
}

export function SystemOverview({ training, feedback }: SystemOverviewProps) {
  if (!training) {
    return (
      <section className={`${panelClass} mb-6`} aria-label="Loading system overview">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {[0, 1, 2, 3].map((item) => <div className="h-24 animate-pulse rounded-2xl bg-slate-800/70 motion-reduce:animate-none" key={item} />)}
        </div>
      </section>
    )
  }

  const additionalVerifiedNeeded = Math.max(
    training.new_verified_samples_needed,
    training.verified_samples_needed_for_minimum,
  )
  const feedbackCoverage = feedback && feedback.total_predictions > 0
    ? (feedback.feedback_received / feedback.total_predictions) * 100
    : 0
  const infrastructureReady = training.redis_available && training.celery_worker_available

  return (
    <section className={`${panelClass} mb-6`} aria-labelledby="overview-title">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-300">Live overview</p>
          <h2 id="overview-title" className="mt-1 text-2xl font-bold tracking-tight text-white">Your data and training readiness</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">A plain-language summary of what is stored, what the model is using, and what must happen before the next automatic training run.</p>
        </div>
        <span className={`${statusBaseClass} ${statusTone(infrastructureReady ? 'ready' : 'degraded')}`}>{infrastructureReady ? 'Services ready' : 'Services need attention'}</span>
      </div>

      <dl className="mt-7 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4"><dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Total stored samples</dt><dd className="mt-2 text-3xl font-black tracking-tight text-white">{training.total_samples}</dd><p className="mt-1 text-xs text-slate-500">Manual entries + feedback samples</p></div>
        <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4"><dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Labelled samples</dt><dd className="mt-2 text-3xl font-black tracking-tight text-white">{training.labelled_samples}</dd><p className="mt-1 text-xs text-slate-500">{training.unlabelled_samples} currently unlabelled</p></div>
        <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4"><dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Verified feedback</dt><dd className="mt-2 text-3xl font-black tracking-tight text-white">{training.verified_feedback_samples}</dd><p className="mt-1 text-xs text-slate-500">Eligible for automatic training</p></div>
        <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4"><dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Predictions</dt><dd className="mt-2 text-3xl font-black tracking-tight text-white">{feedback?.total_predictions ?? '—'}</dd><p className="mt-1 text-xs text-slate-500">{feedback?.feedback_pending ?? '—'} waiting for ground truth</p></div>
      </dl>

      <div className="mt-3 grid gap-3 lg:grid-cols-3">
        <Progress label="Verified dataset minimum" value={training.verified_feedback_samples} target={training.minimum_training_samples} helper={training.verified_samples_needed_for_minimum === 0 ? 'Minimum verified dataset requirement is complete.' : `${training.verified_samples_needed_for_minimum} more verified outcomes needed for the minimum dataset.`} />
        <Progress label="New feedback since last decision" value={training.new_verified_samples} target={training.retrain_threshold} accent="bg-cyan-300" helper={training.new_verified_samples_needed === 0 ? 'The new-feedback trigger threshold is complete.' : `${training.new_verified_samples_needed} more new verified outcomes needed for the trigger.`} />
        <Progress label="Prediction feedback coverage" value={feedback?.feedback_received ?? 0} target={feedback?.total_predictions ?? 0} accent="bg-violet-300" helper={feedback?.total_predictions ? `${feedbackCoverage.toFixed(1)}% of predictions have verified ground truth.` : 'Make predictions, then submit their real outcomes.'} />
      </div>

      <div className={`mt-4 rounded-2xl border px-4 py-3 ${training.retraining_data_ready ? 'border-emerald-400/20 bg-emerald-400/10' : 'border-amber-400/20 bg-amber-400/10'}`}>
        <p className={`text-sm font-bold ${training.retraining_data_ready ? 'text-emerald-200' : 'text-amber-200'}`}>
          {training.training_in_progress
            ? 'A training job is currently in progress.'
            : training.retraining_data_ready
              ? 'The data requirements for automatic retraining are complete.'
              : `${additionalVerifiedNeeded} additional verified ${additionalVerifiedNeeded === 1 ? 'outcome is' : 'outcomes are'} needed before automatic retraining.`}
        </p>
        <p className="mt-1 text-xs leading-5 text-slate-400">Automatic retraining counts verified ground truth from prediction feedback. A manually submitted label is stored and usable for manual training, but it does not count as a new automatic trigger.</p>
      </div>
    </section>
  )
}
