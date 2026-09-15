import { useState } from 'react'

import { SystemOverview } from './features/dashboard/SystemOverview'
import { DataSubmissionForm } from './features/data/DataSubmissionForm'
import { ModelMonitoring } from './features/monitoring/ModelMonitoring'
import { ModelStatus } from './features/prediction/ModelStatus'
import { PredictionForm } from './features/prediction/PredictionForm'
import { PredictionHistory } from './features/prediction/PredictionHistory'
import { ContinuousTrainingStatus } from './features/training/ContinuousTrainingStatus'
import { panelClass } from './lib/ui'
import type { PredictionFeedbackSummary } from './types/prediction'
import type { TrainingStatus } from './types/training'

const navigation = [
  ['Workspace', '#workspace'],
  ['Training', '#training'],
  ['Monitoring', '#monitoring'],
  ['Feedback', '#history'],
] as const

function App() {
  const [historyRefreshToken, setHistoryRefreshToken] = useState(0)
  const [modelRefreshToken, setModelRefreshToken] = useState(0)
  const [trainingRefreshToken, setTrainingRefreshToken] = useState(0)
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null)
  const [feedbackSummary, setFeedbackSummary] = useState<PredictionFeedbackSummary | null>(null)

  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-950 font-sans text-slate-100 antialiased selection:bg-emerald-300 selection:text-slate-950">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[42rem] bg-[radial-gradient(circle_at_15%_10%,rgba(16,185,129,0.16),transparent_32%),radial-gradient(circle_at_85%_0%,rgba(34,211,238,0.12),transparent_28%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(148,163,184,0.025)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.025)_1px,transparent_1px)] bg-[size:48px_48px]" />

      <a className="fixed left-4 top-4 z-50 -translate-y-24 rounded-lg bg-emerald-300 px-4 py-2 font-semibold text-slate-950 transition focus:translate-y-0 focus:outline-none focus:ring-4 focus:ring-emerald-300/30" href="#workspace">
        Skip to workspace
      </a>

      <div className="relative mx-auto max-w-[90rem] px-4 pb-16 sm:px-6 lg:px-8">
        <header className="flex min-h-20 items-center justify-between border-b border-white/10">
          <a className="flex items-center gap-3" href="#top" aria-label="ML Control home">
            <span className="grid size-10 place-items-center rounded-xl bg-gradient-to-br from-emerald-300 to-cyan-400 shadow-lg shadow-emerald-950/60">
              <svg aria-hidden="true" className="size-5 text-slate-950" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 17l5-5 4 4 7-8" /><path d="M15 8h5v5" /></svg>
            </span>
            <span>
              <span className="block text-sm font-extrabold tracking-tight text-white">ML Control</span>
              <span className="block text-[10px] font-medium uppercase tracking-[0.2em] text-slate-500">Learning operations</span>
            </span>
          </a>

          <nav className="hidden items-center gap-1 rounded-full border border-white/10 bg-slate-900/60 p-1 backdrop-blur md:flex" aria-label="Dashboard sections">
            {navigation.map(([label, href]) => (
              <a key={href} className="rounded-full px-4 py-2 text-xs font-semibold text-slate-400 transition hover:bg-white/5 hover:text-white focus:outline-none focus:ring-2 focus:ring-emerald-400/30" href={href}>{label}</a>
            ))}
          </nav>

          <div className="flex items-center gap-2 text-xs font-semibold text-slate-400">
            <span className="size-2 rounded-full bg-cyan-300" />
            Operations console
          </div>
        </header>

        <main id="top">
          <section className="grid gap-8 py-12 lg:grid-cols-[minmax(0,1fr)_22rem] lg:items-end lg:py-20">
            <div className="max-w-4xl">
              <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-emerald-300/20 bg-emerald-300/5 px-3 py-1.5 text-xs font-semibold text-emerald-300"><span className="size-1.5 rounded-full bg-emerald-300" />Continuous learning platform</div>
              <h1 className="max-w-4xl text-4xl font-black tracking-[-0.045em] text-white sm:text-6xl lg:text-7xl">
                Model intelligence,
                <span className="block bg-gradient-to-r from-emerald-300 via-teal-300 to-cyan-300 bg-clip-text text-transparent">under your control.</span>
              </h1>
              <p className="mt-6 max-w-2xl text-base leading-7 text-slate-400 sm:text-lg">Submit labelled samples, run predictions, monitor drift, and follow every training job from one operational workspace.</p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-slate-900/70 p-5 shadow-2xl shadow-black/30 backdrop-blur-xl">
              <div className="mb-3 flex items-center justify-between"><span className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">Serving model</span><svg aria-hidden="true" className="size-4 text-slate-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 3v18M3 12h18" /></svg></div>
              <ModelStatus refreshToken={modelRefreshToken} />
            </div>
          </section>

          <SystemOverview training={trainingStatus} feedback={feedbackSummary} />

          <div className="mb-6 flex gap-2 overflow-x-auto pb-1 md:hidden" aria-label="Mobile dashboard sections">
            {navigation.map(([label, href]) => <a key={href} className="shrink-0 rounded-full border border-slate-700 bg-slate-900 px-4 py-2 text-xs font-semibold text-slate-300" href={href}>{label}</a>)}
          </div>

          <section id="workspace" className="scroll-mt-6">
            <div className="mb-6 flex items-end justify-between gap-4">
              <div><p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-400">Workspace</p><h2 className="mt-2 text-2xl font-bold tracking-tight text-white sm:text-3xl">Run the model</h2></div>
              <span className="hidden text-sm text-slate-500 sm:block">Three-feature binary classifier</span>
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <article className={`${panelClass} relative overflow-hidden`}>
                <div className="absolute right-0 top-0 size-32 rounded-full bg-emerald-400/5 blur-3xl" />
                <div className="relative mb-7 flex items-start gap-4">
                  <span className="grid size-11 shrink-0 place-items-center rounded-xl border border-emerald-300/20 bg-emerald-300/10 text-emerald-300"><svg aria-hidden="true" className="size-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12h14M13 6l6 6-6 6" /></svg></span>
                  <div><h3 className="text-lg font-bold text-white">Make a prediction</h3><p className="mt-1 text-sm leading-6 text-slate-400">Run three numerical features through the active model.</p></div>
                </div>
                <PredictionForm onPredictionCreated={() => setHistoryRefreshToken((current) => current + 1)} />
              </article>

              <article className={`${panelClass} relative overflow-hidden`}>
                <div className="absolute right-0 top-0 size-32 rounded-full bg-cyan-400/5 blur-3xl" />
                <div className="relative mb-7 flex items-start gap-4">
                  <span className="grid size-11 shrink-0 place-items-center rounded-xl border border-cyan-300/20 bg-cyan-300/10 text-cyan-300"><svg aria-hidden="true" className="size-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14" /></svg></span>
                  <div><h3 className="text-lg font-bold text-white">Add labelled sample</h3><p className="mt-1 text-sm leading-6 text-slate-400">Store verified training data with an optional binary label.</p></div>
                </div>
                <DataSubmissionForm onDataCreated={() => setTrainingRefreshToken((current) => current + 1)} />
              </article>
            </div>
          </section>

          <div id="training" className="scroll-mt-6 pt-6"><ContinuousTrainingStatus refreshToken={trainingRefreshToken} onStatusChange={setTrainingStatus} /></div>
          <div id="monitoring" className="scroll-mt-6 pt-6"><ModelMonitoring onActiveModelChanged={() => setModelRefreshToken((current) => current + 1)} /></div>

          <section id="history" className={`${panelClass} mt-6 scroll-mt-6`} aria-labelledby="history-title">
            <div className="mb-7 flex items-start gap-4">
              <span className="grid size-11 shrink-0 place-items-center rounded-xl border border-violet-300/20 bg-violet-300/10 text-violet-300"><svg aria-hidden="true" className="size-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 12a9 9 0 109-9 9.75 9.75 0 00-6.74 2.74L3 8" /><path d="M3 3v5h5M12 7v5l3 2" /></svg></span>
              <div><p className="text-xs font-bold uppercase tracking-[0.18em] text-violet-300">Feedback loop</p><h2 id="history-title" className="mt-1 text-xl font-bold text-white">Prediction history &amp; ground truth</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">Add a verified outcome once it is known. Ground truth remains immutable and is never copied from a model prediction.</p></div>
            </div>
            <PredictionHistory
              refreshToken={historyRefreshToken}
              onFeedbackSubmitted={() => setTrainingRefreshToken((current) => current + 1)}
              onSummaryChange={setFeedbackSummary}
            />
          </section>
        </main>

        <footer className="mt-12 flex flex-col gap-2 border-t border-white/10 py-6 text-xs text-slate-600 sm:flex-row sm:items-center sm:justify-between"><span>ML Control · Continuous learning operations</span><span>FastAPI · React · Celery · Redis</span></footer>
      </div>
    </div>
  )
}

export default App
