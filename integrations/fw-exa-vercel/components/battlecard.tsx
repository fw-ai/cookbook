'use client'

import { useState } from 'react'
import type { DeepPartial } from 'ai'
import type { Battlecard } from '@/lib/schema'
import {
  TrendingUp,
  AlertTriangle,
  Zap,
  MessageSquare,
  DollarSign,
  Users,
  Newspaper,
  CheckCircle,
  ArrowDown,
  ChevronDown,
  GitCompare,
  Trophy,
} from 'lucide-react'

interface BattlecardProps {
  data: DeepPartial<Battlecard>
  isStreaming: boolean
  yourCompany: string
  analysisContext?: string
  companySize?: string
}

const SIZE_LABELS: Record<string, string> = {
  startup: 'Startup',
  smb: 'SMB',
  'mid-market': 'Mid-Market',
  enterprise: 'Enterprise',
}

/**
 * Reusable collapsible section card used throughout BattlecardView.
 * Cards default to open so all content is visible on first load.
 * The chevron rotates 180° when open and animates back when collapsed.
 * Content expands/collapses with a CSS grid height animation (no JS height measurement needed).
 *
 * @param icon     - Lucide icon component for the section header
 * @param title    - Section heading text (rendered uppercase)
 * @param color    - Tailwind text color class applied to icon + title
 * @param count    - Optional badge showing the number of items in the section
 * @param children - Section body content
 */
function CollapsibleCard({
  icon: Icon,
  title,
  color,
  count,
  children,
  className = '',
}: {
  icon: React.ComponentType<{ size?: number; className?: string }>
  title: string
  color: string
  count?: number
  children: React.ReactNode
  className?: string
}) {
  const [open, setOpen] = useState(true) // Open by default — user can collapse

  return (
    <div
      className={`section-reveal border rounded-2xl overflow-hidden bg-slate-900/70 border-slate-800/70 ${className}`}
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={`w-full flex items-center justify-between px-5 pt-5 transition-colors hover:bg-slate-800/30 ${open ? 'pb-4' : 'pb-5'}`}
      >
        <div className={`flex items-center gap-2 ${color}`}>
          <Icon size={15} />
          <span className="text-[11px] font-bold uppercase tracking-[0.12em]">{title}</span>
          {count !== undefined && count > 0 && (
            <span className="ml-1 text-[10px] font-semibold bg-slate-800 text-slate-500 px-2 py-0.5 rounded-full tabular-nums">
              {count}
            </span>
          )}
        </div>
        <ChevronDown
          size={14}
          className={`text-slate-600 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
        />
      </button>
      {/* CSS grid trick for smooth open/close animation without measuring height */}
      <div
        style={{ gridTemplateRows: open ? '1fr' : '0fr' }}
        className="grid transition-[grid-template-rows] duration-200 ease-in-out"
      >
        <div className="overflow-hidden">
          <div className="px-5 pb-5">{children}</div>
        </div>
      </div>
    </div>
  )
}

const VERDICT_CONFIG = {
  win: { label: 'Win', bg: 'bg-emerald-950/40', border: 'border-emerald-900/40', text: 'text-emerald-400' },
  lose: { label: 'Lose', bg: 'bg-rose-950/40', border: 'border-rose-900/40', text: 'text-rose-400' },
  neutral: { label: 'Neutral', bg: 'bg-slate-800/60', border: 'border-slate-700/50', text: 'text-slate-500' },
}

export function BattlecardView({ data, isStreaming, yourCompany, analysisContext, companySize }: BattlecardProps) {
  const hasData = !!data.competitor
  const initial = data.competitor?.[0]?.toUpperCase() ?? '?'

  return (
    <div
      className={[
        'rounded-2xl border transition-all duration-500 overflow-hidden',
        isStreaming ? 'streaming-card' : 'border-slate-700/60',
      ].join(' ')}
    >
      {/* ── Header ── */}
      <div className="relative bg-gradient-to-b from-slate-900 to-slate-950 px-6 pt-6 pb-5 border-b border-slate-800/60">
        {/* Top accent line */}
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-indigo-500/50 to-transparent" />

        <div className="flex items-start gap-4">
          {/* Competitor initial avatar */}
          <div className="shrink-0 w-11 h-11 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
            {data.competitor ? (
              <span className="text-lg font-bold text-indigo-400">{initial}</span>
            ) : (
              <div className="w-5 h-5 skeleton rounded-md" />
            )}
          </div>

          <div className="flex-1 min-w-0">
            {/* Status row */}
            <div className="flex items-center gap-2 mb-1.5 flex-wrap">
              {isStreaming && (
                <span className="flex items-center gap-1.5 text-[10px] text-indigo-400 animate-pulse">
                  <span className="inline-block w-1 h-1 rounded-full bg-indigo-400" />
                  Generating
                </span>
              )}
              {!isStreaming && hasData && (
                <span className="flex items-center gap-1 text-[10px] text-emerald-400">
                  <CheckCircle size={10} />
                  Ready
                </span>
              )}
              {companySize && SIZE_LABELS[companySize] && (
                <span className="text-[10px] text-slate-500 bg-slate-800/60 border border-slate-700/50 px-2 py-0.5 rounded-full">
                  {SIZE_LABELS[companySize]}
                </span>
              )}
              {/* analysisContext badge — amber color signals this is a focused/tailored analysis */}
              {analysisContext && (
                <span className="flex items-center gap-1 text-[10px] text-amber-400/90 bg-amber-950/30 border border-amber-900/30 px-2 py-0.5 rounded-full max-w-[220px]">
                  <Zap size={9} className="shrink-0" />
                  <span className="truncate">{analysisContext}</span>
                </span>
              )}
            </div>

            {/* Card title */}
            {data.competitor ? (
              <h2 className="text-lg font-bold leading-snug">
                <span className="text-slate-300">{yourCompany}</span>
                <span className="text-slate-600 font-normal"> rep playbook for </span>
                <span className="text-slate-100">{data.competitor}</span>
              </h2>
            ) : (
              <div className="skeleton h-6 w-64" />
            )}

            {/* Tagline */}
            {data.tagline ? (
              <p className="mt-1 text-slate-400 text-sm">{data.tagline}</p>
            ) : isStreaming ? (
              <div className="skeleton h-4 w-64 mt-2" />
            ) : null}

            {/* Win Themes pills — shown in header once generation is complete */}
            {!isStreaming && data.winThemes && data.winThemes.filter(Boolean).length > 0 && (
              <div className="mt-3">
                <div className="flex items-center gap-1 mb-1.5">
                  <Trophy size={10} className="text-violet-400/60" />
                  <span className="text-[10px] text-violet-400/60 font-bold uppercase tracking-widest">Win Themes</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {data.winThemes.filter(Boolean).map((theme, i) => (
                    <span
                      key={i}
                      className="text-[10px] px-2.5 py-1 rounded-full bg-violet-950/40 border border-violet-900/40 text-violet-300 font-medium"
                    >
                      {theme}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Overview */}
        {data.overview && (
          <p className="mt-5 text-slate-300 text-sm leading-relaxed border-t border-slate-800/60 pt-4">
            {data.overview}
          </p>
        )}
      </div>

      {/* ── Body — ordered for a rep in a live deal ── */}
      <div className="p-5 space-y-3 bg-slate-950/70">

        {/* 1. Head-to-Head — first so the rep immediately sees where they win/lose */}
        {data.contextualAnalysis && (
          <CollapsibleCard
            icon={GitCompare}
            title="Head-to-Head"
            color="text-cyan-400"
            className="!bg-gradient-to-br from-cyan-950/20 via-slate-900/70 to-cyan-950/20 !border-cyan-900/30"
          >
            {data.contextualAnalysis.headline && (
              <p className="text-slate-300 text-sm mb-4 italic">{data.contextualAnalysis.headline}</p>
            )}

            {data.contextualAnalysis.dimensions && data.contextualAnalysis.dimensions.length > 0 && (
              <div className="space-y-3 mb-4">
                {data.contextualAnalysis.dimensions.map((dim, i) =>
                  dim?.aspect ? (
                    <div key={i} className="rounded-xl border border-slate-800/80 overflow-hidden">
                      {/* Aspect header with verdict */}
                      <div className="flex items-center justify-between px-4 py-2.5 bg-slate-800/50">
                        <span className="text-xs font-semibold text-slate-300">{dim.aspect}</span>
                        {dim.verdict && (
                          <span
                            className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border ${VERDICT_CONFIG[dim.verdict].bg} ${VERDICT_CONFIG[dim.verdict].border} ${VERDICT_CONFIG[dim.verdict].text}`}
                          >
                            {dim.verdict === 'win' ? `${yourCompany} wins` : dim.verdict === 'lose' ? 'They win' : 'Neutral'}
                          </span>
                        )}
                      </div>
                      {/* Two-column comparison */}
                      <div className="grid grid-cols-2 divide-x divide-slate-800/60">
                        <div className="px-4 py-3">
                          <div className="text-[10px] text-indigo-400 font-bold uppercase tracking-widest mb-1">
                            {yourCompany}
                          </div>
                          <p className="text-xs text-slate-300 leading-relaxed">{dim.yourPosition}</p>
                        </div>
                        <div className="px-4 py-3">
                          <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">
                            {data.competitor}
                          </div>
                          <p className="text-xs text-slate-300 leading-relaxed">{dim.theirPosition}</p>
                        </div>
                      </div>
                    </div>
                  ) : null,
                )}
              </div>
            )}

            {data.contextualAnalysis.playbook && (
              <div className="bg-cyan-950/20 border border-cyan-900/25 rounded-xl p-3.5">
                <div className="text-[10px] text-cyan-500/60 font-bold uppercase tracking-widest mb-1.5">
                  How to Win
                </div>
                <p className="text-slate-300 text-sm leading-relaxed">{data.contextualAnalysis.playbook}</p>
              </div>
            )}
          </CollapsibleCard>
        )}

        {/* 2. Differentiators — talk tracks for the call */}
        {data.differentiators && data.differentiators.length > 0 && (
          <CollapsibleCard
            icon={Zap}
            title={`Why ${yourCompany} Wins`}
            color="text-indigo-400"
            count={data.differentiators.filter((d) => d?.point).length}
          >
            <div className="space-y-4">
              {data.differentiators.map((d, i) =>
                d?.point ? (
                  <div key={i} className="border-l-2 border-indigo-500/40 pl-4">
                    <div className="text-slate-100 font-semibold text-sm mb-2">{d.point}</div>
                    {d.talkTrack && (
                      <div className="bg-indigo-950/25 border border-indigo-900/25 rounded-xl p-3.5">
                        <div className="text-[10px] text-indigo-500/60 font-bold uppercase tracking-widest mb-1.5">
                          Talk Track
                        </div>
                        <p className="text-slate-300 text-sm leading-relaxed italic">
                          &ldquo;{d.talkTrack}&rdquo;
                        </p>
                      </div>
                    )}
                  </div>
                ) : null,
              )}
            </div>
          </CollapsibleCard>
        )}

        {/* 3. Objection Handling — rebuttals ready for the call */}
        {data.objections && data.objections.length > 0 && (
          <CollapsibleCard
            icon={MessageSquare}
            title="Objection Handling"
            color="text-amber-400"
            count={data.objections.filter((o) => o?.objection).length}
          >
            <div className="space-y-5">
              {data.objections.map((o, i) =>
                o?.objection ? (
                  <div key={i}>
                    <div className="bg-amber-950/25 border border-amber-900/25 rounded-xl p-4">
                      <div className="text-[10px] text-amber-500/70 font-bold uppercase tracking-widest mb-1.5">
                        Prospect says
                      </div>
                      <p className="text-slate-300 text-sm italic">&ldquo;{o.objection}&rdquo;</p>
                    </div>
                    {o.rebuttal && (
                      <>
                        <div className="flex justify-start pl-5 my-1.5">
                          <ArrowDown size={13} className="text-slate-700" />
                        </div>
                        <div className="bg-emerald-950/20 border border-emerald-900/25 rounded-xl p-4">
                          <div className="text-[10px] text-emerald-500/70 font-bold uppercase tracking-widest mb-1.5">
                            You respond
                          </div>
                          <p className="text-slate-300 text-sm">{o.rebuttal}</p>
                        </div>
                      </>
                    )}
                  </div>
                ) : null,
              )}
            </div>
          </CollapsibleCard>
        )}

        {/* 4. Strengths & Weaknesses — know the landscape */}
        {((data.strengths && data.strengths.length > 0) ||
          (data.weaknesses && data.weaknesses.length > 0)) && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {data.strengths && data.strengths.length > 0 && (
              <CollapsibleCard
                icon={TrendingUp}
                title="Their Strengths"
                color="text-emerald-400"
                count={data.strengths.filter(Boolean).length}
              >
                <ul className="space-y-2.5">
                  {data.strengths.map((s, i) =>
                    s ? (
                      <li key={i} className="flex items-start gap-2.5 text-sm text-slate-300">
                        <span className="shrink-0 mt-0.5 text-emerald-700 font-bold text-xs w-4 text-right">
                          {i + 1}.
                        </span>
                        <span className="leading-snug">{s}</span>
                      </li>
                    ) : null,
                  )}
                </ul>
              </CollapsibleCard>
            )}
            {data.weaknesses && data.weaknesses.length > 0 && (
              <CollapsibleCard
                icon={AlertTriangle}
                title="Their Weaknesses"
                color="text-rose-400"
                count={data.weaknesses.filter(Boolean).length}
              >
                <ul className="space-y-2.5">
                  {data.weaknesses.map((w, i) =>
                    w ? (
                      <li key={i} className="flex items-start gap-2.5 text-sm text-slate-300">
                        <span className="shrink-0 mt-0.5 text-rose-700 font-bold text-xs w-4 text-right">
                          {i + 1}.
                        </span>
                        <span className="leading-snug">{w}</span>
                      </li>
                    ) : null,
                  )}
                </ul>
              </CollapsibleCard>
            )}
          </div>
        )}

        {/* 5. Recent Developments — news & momentum */}
        {data.recentDevelopments && data.recentDevelopments.length > 0 && (
          <CollapsibleCard
            icon={Newspaper}
            title="Recent Developments"
            color="text-slate-400"
            count={data.recentDevelopments.filter(Boolean).length}
          >
            <ul className="space-y-3">
              {data.recentDevelopments.map((item, i) =>
                item ? (
                  <li key={i} className="flex gap-3 text-sm text-slate-300">
                    <span className="shrink-0 mt-2 w-1.5 h-1.5 rounded-full bg-slate-600 ring-2 ring-slate-800/80" />
                    <span className="leading-relaxed">{item}</span>
                  </li>
                ) : null,
              )}
            </ul>
          </CollapsibleCard>
        )}

        {/* 6. Pricing & ICP — for budget and fit questions */}
        {(data.pricing || data.idealCustomerProfile) && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {data.pricing && (
              <CollapsibleCard icon={DollarSign} title="Pricing Intel" color="text-slate-400">
                <p className="text-sm text-slate-300 leading-relaxed">{data.pricing}</p>
              </CollapsibleCard>
            )}
            {data.idealCustomerProfile && (
              <CollapsibleCard icon={Users} title="Their Sweet Spot" color="text-slate-400">
                <p className="text-sm text-slate-300 leading-relaxed">
                  {data.idealCustomerProfile}
                </p>
              </CollapsibleCard>
            )}
          </div>
        )}

      </div>

      {/* ── Footer ── */}
      <div className="bg-slate-900/80 border-t border-slate-800/60 px-6 py-3 flex items-center justify-between">
        <div className="text-xs text-slate-600">
          Researched with{' '}
          <span className="text-slate-500 font-medium">Exa</span>
          {' · '}
          Analyzed by{' '}
          <span className="text-slate-500 font-medium">Fireworks AI</span>
          {' · '}
          Streamed by{' '}
          <span className="text-slate-500 font-medium">Vercel AI SDK</span>
        </div>
        <div className="text-xs text-slate-600">
          {new Date().toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric',
          })}
        </div>
      </div>
    </div>
  )
}
