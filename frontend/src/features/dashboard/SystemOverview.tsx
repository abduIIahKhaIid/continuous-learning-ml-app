import { panelClass, statusBaseClass, statusTone } from '../../lib/ui'
import type { PredictionFeedbackSummary } from '../../types/prediction'
import type { TrainingStatus } from '../../types/training'

interface SystemOverviewProps {
  training: TrainingStatus | null
  feedback: PredictionFeedbackSummary | null
}

function percentage(value: number, target: number): number {
  if (target <= 0) return 0
  return Math.min(100, Math.max(0, (value / target) * 100))
}

function Requirement({
  title,
  current,
  target,
  remaining,
  color,
}: {
  title: string
  current: number
  target: number
  remaining: number
  color: string
}) {
  const progress = percentage(current, target)

  return (
    <div>
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="font-semibold text-slate-300">{title}</span>
        <span className="font-mono font-bold text-white">{current} / {target}</span>
      </div>
      <div
        aria-label={`${title}: ${Math.round(progress)}% complete`}
        aria-valuemax={100}
        aria-valuemin={0}
        aria-valuenow={Math.round(progress)}
        className="mt-2 h-2 overflow-hidden rounded-full bg-slate-800"
        role="progressbar"
      >
        <div
          className={`h-full rounded-full ${color} transition-[width] duration-500 motion-reduce:transition-none`}
          style={{ width: `${progress}%` }}
        />
      </div>
      <p className="mt-1.5 text-[11px] text-slate-500">
        {remaining === 0 ? 'Requirement complete' : `${remaining} remaining`}
      </p>
    </div>
  )
}

export function SystemOverview({ training, feedback }: SystemOverviewProps) {
  if (!training) {
    return (
      <section className={`${panelClass} mb-6`} aria-label="Loading dashboard overview">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {[0, 1, 2, 3].map((item) => (
            <div
              className="h-24 animate-pulse rounded-2xl bg-slate-800/70 motion-reduce:animate-none"
              key={item}
            />
          ))}
        </div>
      </section>
    )
  }

  const additionalVerifiedNeeded = Math.max(
    training.new_verified_samples_needed,
    training.verified_samples_needed_for_minimum,
  )
  const servicesReady =
    training.redis_available && training.celery_worker_available
  const trainingHeadline = training.training_in_progress
    ? 'Training is running now'
    : training.retraining_data_ready
      ? 'Data requirements are complete'
      : `${additionalVerifiedNeeded} verified outcomes needed`

  const cards = [
    {
      label: 'Stored samples',
      value: training.total_samples,
      detail: 'All records in the dataset',
    },
    {
      label: 'Labelled samples',
      value: training.labelled_samples,
      detail: `${training.unlabelled_samples} without a label`,
    },
    {
      label: 'Verified outcomes',
      value: training.verified_feedback_samples,
      detail: 'Eligible for auto-training',
    },
    {
      label: 'Predictions',
      value: feedback?.total_predictions ?? '—',
      detail: `${feedback?.feedback_pending ?? '—'} awaiting feedback`,
    },
  ]

  return (
    <section className={`${panelClass} mb-6`} aria-labelledby="overview-title">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-300">
            Dashboard overview
          </p>
          <h2
            className="mt-1 text-2xl font-bold tracking-tight text-white"
            id="overview-title"
          >
            Data at a glance
          </h2>
        </div>
        <span
          className={`${statusBaseClass} ${statusTone(servicesReady ? 'ready' : 'degraded')}`}
        >
          {servicesReady ? 'Redis & worker ready' : 'Infrastructure needs attention'}
        </span>
      </div>

      <dl className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {cards.map((card) => (
          <div
            className="rounded-2xl border border-white/10 bg-slate-950/45 p-4"
            key={card.label}
          >
            <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              {card.label}
            </dt>
            <dd className="mt-2 text-3xl font-black tracking-tight text-white">
              {card.value}
            </dd>
            <p className="mt-1 text-xs text-slate-500">{card.detail}</p>
          </div>
        ))}
      </dl>

      <div className="mt-3 grid overflow-hidden rounded-2xl border border-white/10 bg-slate-950/45 lg:grid-cols-[minmax(16rem,.8fr)_minmax(0,1.2fr)]">
        <div className="border-b border-white/10 p-5 lg:border-b-0 lg:border-r">
          <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-slate-500">
            Next automatic training
          </p>
          <p className={`mt-2 text-2xl font-black tracking-tight ${training.retraining_data_ready || training.training_in_progress ? 'text-emerald-300' : 'text-amber-300'}`}>
            {trainingHeadline}
          </p>
          <p className="mt-2 text-xs leading-5 text-slate-400">
            Both requirements on the right must be complete. Each new verified
            outcome advances both counters.
          </p>
        </div>

        <div className="grid gap-5 p-5 sm:grid-cols-2">
          <Requirement
            color="bg-emerald-300"
            current={training.verified_feedback_samples}
            remaining={training.verified_samples_needed_for_minimum}
            target={training.minimum_training_samples}
            title="Verified dataset minimum"
          />
          <Requirement
            color="bg-cyan-300"
            current={training.new_verified_samples}
            remaining={training.new_verified_samples_needed}
            target={training.retrain_threshold}
            title="New outcomes since last decision"
          />
        </div>
      </div>

      <div className="mt-3 flex items-start gap-3 rounded-xl border border-cyan-300/15 bg-cyan-300/5 px-4 py-3">
        <svg
          aria-hidden="true"
          className="mt-0.5 size-4 shrink-0 text-cyan-300"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          viewBox="0 0 24 24"
        >
          <circle cx="12" cy="12" r="9" />
          <path d="M12 11v5M12 8h.01" />
        </svg>
        <p className="text-xs leading-5 text-slate-400">
          <strong className="text-slate-200">What counts?</strong> Automatic
          training uses verified outcomes submitted from Prediction History.
          Manually entered labels are stored for manual training, but do not
          advance the automatic trigger.
        </p>
      </div>
    </section>
  )
}
