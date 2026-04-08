'use client'

import { useState, useRef } from 'react'
import { BattlecardTab, type TabStatus } from '@/components/battlecard-tab'
import { Swords, Plus, X, Zap, ChevronDown, ArrowLeft, Loader2, CheckCircle } from 'lucide-react'

const MAX_COMPETITORS = 5

const EXAMPLES = [
  { yourCompany: 'Salesforce', competitors: ['HubSpot', 'Pipedrive'] },
  { yourCompany: 'Snowflake', competitors: ['Databricks', 'BigQuery'] },
  { yourCompany: 'Figma', competitors: ['Sketch', 'Adobe XD'] },
]

const COMPANY_SIZES = [
  { value: 'startup', label: 'Startup', description: '< 50 employees' },
  { value: 'smb', label: 'SMB', description: '50–500' },
  { value: 'mid-market', label: 'Mid-Market', description: '500–5k' },
  { value: 'enterprise', label: 'Enterprise', description: '5k+' },
]

interface Submitted {
  yourCompany: string
  competitors: string[]
  analysisContext?: string
  companySize?: string
}

export default function Home() {
  // Form inputs — held in state until the user submits
  const [yourCompany, setYourCompany] = useState('')
  const [competitors, setCompetitors] = useState<string[]>(['', '']) // Start with 2 empty inputs
  const [analysisContext, setAnalysisContext] = useState('')         // Optional deal context
  const [companySize, setCompanySize] = useState('')                 // Optional deal size tier

  // submitted: null = form view, non-null = results view.
  // Holds a snapshot of the form values at submit time. All BattlecardTab
  // children read from this snapshot — not from the live form state — so
  // editing the form after submit doesn't affect in-progress generation.
  const [submitted, setSubmitted] = useState<Submitted | null>(null)

  // activeTab: index into submitted.competitors — controls which battlecard is visible.
  // All BattlecardTab components stay mounted (display:none when inactive) so
  // generation continues in the background even when a tab isn't visible.
  const [activeTab, setActiveTab] = useState(0)

  // tabStatuses: tracks loading/complete/error per tab index for the tab bar indicators.
  // Each BattlecardTab reports its own status via onStatusChange callback.
  const [tabStatuses, setTabStatuses] = useState<Record<number, TabStatus>>({})

  const lastInputRef = useRef<HTMLInputElement>(null) // Used to auto-focus the newest competitor input

  const addCompetitor = () => {
    if (competitors.length >= MAX_COMPETITORS) return
    setCompetitors((prev) => [...prev, ''])
    setTimeout(() => lastInputRef.current?.focus(), 50)
  }

  const removeCompetitor = (i: number) => {
    setCompetitors((prev) => prev.filter((_, idx) => idx !== i))
  }

  const updateCompetitor = (i: number, value: string) => {
    setCompetitors((prev) => {
      const next = [...prev]
      next[i] = value
      return next
    })
  }

  const validCompetitors = competitors.filter((c) => c.trim())
  const canGenerate = yourCompany.trim().length > 0 && validCompetitors.length > 0

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!canGenerate) return
    setSubmitted({
      yourCompany: yourCompany.trim(),
      competitors: validCompetitors,
      analysisContext: analysisContext.trim() || undefined,
      companySize: companySize || undefined,
    })
    setActiveTab(0)
    setTabStatuses({})
  }

  const handleReset = () => {
    setSubmitted(null)
    setActiveTab(0)
    setTabStatuses({})
  }

  const loadExample = (ex: (typeof EXAMPLES)[0]) => {
    setYourCompany(ex.yourCompany)
    setCompetitors(ex.competitors)
    setSubmitted(null)
  }

  return (
    <main className="min-h-screen bg-slate-950">
      {/* ── Nav ── */}
      <nav className="border-b border-slate-900 px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center gap-2">
          <Swords size={18} className="text-indigo-400" />
          <span className="font-semibold text-slate-200 text-sm tracking-tight">
            Live Battlecards
          </span>
          <span className="ml-auto text-xs text-slate-600">
            Powered by{' '}
            <a
              href="https://exa.ai"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-500 hover:text-slate-400 transition-colors"
            >
              Exa
            </a>{' '}
            +{' '}
            <a
              href="https://fireworks.ai"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-500 hover:text-slate-400 transition-colors"
            >
              Fireworks AI
            </a>{' '}
            +{' '}
            <a
              href="https://vercel.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-500 hover:text-slate-400 transition-colors"
            >
              Vercel
            </a>
          </span>
        </div>
      </nav>

      <div className="max-w-5xl mx-auto px-4 py-10">
        {/* ── Hero ── */}
        <div className="text-center mb-8">
          <h1 className="text-4xl sm:text-5xl font-bold text-slate-100 mb-3 tracking-tight">
            Research any competitor{' '}
            <span className="text-indigo-400">in seconds</span>
          </h1>
          <p className="text-slate-400 text-lg max-w-xl mx-auto">
            Live competitive intelligence for your active deals.
          </p>
        </div>

        {/* ── Input form ── */}
        {!submitted && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 mb-6">
            <form onSubmit={handleSubmit} className="space-y-5">
              {/* Two-column: Your Company + Competitors */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                {/* Your company */}
                <div>
                  <label
                    htmlFor="yourCompany"
                    className="block text-xs font-semibold text-slate-500 uppercase tracking-widest mb-2"
                  >
                    Your Company
                  </label>
                  <input
                    id="yourCompany"
                    type="text"
                    value={yourCompany}
                    onChange={(e) => setYourCompany(e.target.value)}
                    placeholder="e.g. Salesforce"
                    autoComplete="off"
                    className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-slate-100 placeholder-slate-600 text-sm focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 transition-colors"
                  />
                </div>

                {/* Competitors */}
                <div>
                  <label className="block text-xs font-semibold text-slate-500 uppercase tracking-widest mb-2">
                    Competitors{' '}
                    <span className="text-slate-700 normal-case font-normal">
                      ({competitors.filter((c) => c.trim()).length}/{MAX_COMPETITORS})
                    </span>
                  </label>
                  <div className="space-y-2">
                    {competitors.map((c, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <input
                          ref={i === competitors.length - 1 ? lastInputRef : undefined}
                          type="text"
                          value={c}
                          onChange={(e) => updateCompetitor(i, e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              e.preventDefault()
                              addCompetitor()
                            }
                          }}
                          placeholder={`Competitor ${i + 1}`}
                          autoComplete="off"
                          className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 placeholder-slate-600 text-sm focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 transition-colors"
                        />
                        {competitors.length > 1 && (
                          <button
                            type="button"
                            onClick={() => removeCompetitor(i)}
                            className="text-slate-600 hover:text-slate-400 transition-colors p-1 shrink-0"
                            aria-label="Remove"
                          >
                            <X size={14} />
                          </button>
                        )}
                      </div>
                    ))}
                    {competitors.length < MAX_COMPETITORS && (
                      <button
                        type="button"
                        onClick={addCompetitor}
                        className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg border border-dashed border-slate-700 text-slate-600 hover:text-slate-400 hover:border-slate-600 text-sm transition-colors"
                      >
                        <Plus size={13} />
                        Add competitor
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Deal context row: Company Size + Analysis Focus */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {/* Company Size — segmented control */}
                <div>
                  <label className="block text-xs font-semibold text-slate-500 uppercase tracking-widest mb-2">
                    Deal Size{' '}
                    <span className="text-slate-700 normal-case font-normal">(optional)</span>
                  </label>
                  <div className="relative">
                    <select
                      value={companySize}
                      onChange={(e) => setCompanySize(e.target.value)}
                      className="w-full appearance-none bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 pr-9 text-sm focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 transition-colors text-slate-100"
                    >
                      <option value="" className="text-slate-500">Any size</option>
                      {COMPANY_SIZES.map((s) => (
                        <option key={s.value} value={s.value}>
                          {s.label} ({s.description})
                        </option>
                      ))}
                    </select>
                    <ChevronDown
                      size={14}
                      className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-500"
                    />
                  </div>
                </div>

                {/* Analysis Focus */}
                <div className="sm:col-span-2">
                  <label className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 uppercase tracking-widest mb-2">
                    <Zap size={11} className="text-amber-500/70" />
                    Analysis Focus{' '}
                    <span className="text-slate-700 normal-case font-normal">(optional)</span>
                  </label>
                  <textarea
                    value={analysisContext}
                    onChange={(e) => setAnalysisContext(e.target.value)}
                    placeholder="e.g. Losing on price · CTO asking about API limits · Champion needs to justify the switch"
                    rows={2}
                    className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-slate-100 placeholder-slate-600 text-sm focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/20 transition-colors resize-none"
                  />
                </div>
              </div>

              {/* Submit */}
              <button
                type="submit"
                disabled={!canGenerate}
                className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 disabled:bg-slate-800 disabled:text-slate-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-xl text-sm transition-colors"
              >
                <Zap size={15} />
                Generate{validCompetitors.length > 1 ? ` ${validCompetitors.length} Battlecards` : ' Battlecard'}
              </button>
            </form>
          </div>
        )}

        {/* ── Quick examples (pre-submit only) ── */}
        {!submitted && (
          <div className="flex flex-wrap items-center gap-2 mb-8 justify-center">
            <span className="text-xs text-slate-600">Try:</span>
            {EXAMPLES.map((ex) => (
              <button
                key={ex.yourCompany}
                onClick={() => loadExample(ex)}
                className="text-xs px-3 py-1.5 rounded-lg border border-slate-800 text-slate-500 hover:text-slate-300 hover:border-slate-700 transition-colors bg-slate-900/50"
              >
                {ex.yourCompany} vs {ex.competitors.join(', ')}
              </button>
            ))}
          </div>
        )}

        {/* ── Results area ── */}
        {submitted && (
          <div>
            {/* Header row with reset */}
            <div className="flex items-center justify-between mb-4">
              <div className="text-sm text-slate-400">
                <span className="text-slate-200 font-medium">{submitted.yourCompany}</span>
                {' vs '}
                <span className="text-slate-400">{submitted.competitors.join(', ')}</span>
              </div>
              <button
                onClick={handleReset}
                className="flex items-center gap-1.5 text-xs text-slate-600 hover:text-slate-400 transition-colors border border-slate-800 hover:border-slate-700 px-3 py-1.5 rounded-lg"
              >
                <ArrowLeft size={12} />
                New search
              </button>
            </div>

            {/* Tab bar */}
            <TabBar
              competitors={submitted.competitors}
              activeTab={activeTab}
              onTabChange={setActiveTab}
              statuses={tabStatuses}
            />

            {/* Tab content — all mounted in parallel, only active is visible */}
            {submitted.competitors.map((competitor, i) => (
              <div
                key={`${submitted.yourCompany}-${competitor}`}
                style={{ display: activeTab === i ? 'block' : 'none' }}
              >
                <BattlecardTab
                  yourCompany={submitted.yourCompany}
                  competitor={competitor}
                  analysisContext={submitted.analysisContext}
                  companySize={submitted.companySize}
                  onStatusChange={(status) => setTabStatuses((prev) => ({ ...prev, [i]: status }))}
                />
              </div>
            ))}
          </div>
        )}
      </div>
    </main>
  )
}

// ── Tab bar with live status indicators ──
function TabBar({
  competitors,
  activeTab,
  onTabChange,
  statuses,
}: {
  competitors: string[]
  activeTab: number
  onTabChange: (i: number) => void
  statuses: Record<number, TabStatus>
}) {
  return (
    <div className="flex gap-1 mb-4 overflow-x-auto pb-1">
      {competitors.map((competitor, i) => {
        const status = statuses[i] ?? 'loading'
        return (
          <button
            key={competitor}
            onClick={() => onTabChange(i)}
            className={[
              'flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium whitespace-nowrap transition-all',
              activeTab === i
                ? 'bg-slate-800 text-slate-100 border border-slate-700'
                : 'text-slate-500 hover:text-slate-300 hover:bg-slate-900 border border-transparent',
            ].join(' ')}
          >
            {status === 'loading' ? (
              <Loader2 size={11} className="animate-spin text-indigo-400/70 shrink-0" />
            ) : status === 'error' ? (
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500 shrink-0" />
            ) : activeTab === i ? (
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 shrink-0" />
            ) : (
              <CheckCircle size={11} className="text-emerald-500/70 shrink-0" />
            )}
            {competitor}
          </button>
        )
      })}
    </div>
  )
}
