'use client'

import { useState, useRef } from 'react'
import type { DeepPartial } from 'ai'
import type { Battlecard } from '@/lib/schema'
import { Loader2, Map, Cpu, Search, Users } from 'lucide-react'

interface FollowUpPanelProps {
  yourCompany: string
  competitor: string
  battlecard: DeepPartial<Battlecard>  // The completed battlecard from BattlecardTab
  analysisContext?: string
  companySize?: string
}

// The four preset options shown as buttons in the "Dive Deeper" section.
// Each maps to a distinct prompt in /api/followup/route.ts via presetType.
const PRESETS = [
  { id: 'positioning-playbook', label: 'Positioning Playbook', icon: Map },
  { id: 'technical-deep-dive', label: 'Technical Deep-Dive', icon: Cpu },
  { id: 'discovery-questions', label: 'Discovery Questions', icon: Search },
  { id: 'champion-brief', label: 'Champion Brief', icon: Users },
] as const

/**
 * Converts the structured battlecard object into a compact plain-text summary
 * to send as LLM context to /api/followup. This avoids sending the full JSON
 * schema and keeps the token count manageable while preserving the key facts
 * the model needs: overview, strengths, weaknesses, differentiators, pricing,
 * ICP, and objections.
 *
 * Fields are filtered for truthiness since the battlecard may be DeepPartial
 * (partially streamed) when accessed — though in practice FollowUpPanel only
 * renders after isLoading is false.
 */
function buildBattlecardSummary(battlecard: DeepPartial<Battlecard>, competitor: string): string {
  const lines: string[] = []

  if (battlecard.overview) {
    lines.push(`OVERVIEW: ${battlecard.overview}`)
  }

  if (battlecard.strengths?.filter(Boolean).length) {
    lines.push(`${competitor.toUpperCase()} STRENGTHS:\n${battlecard.strengths.filter(Boolean).map(s => `- ${s}`).join('\n')}`)
  }

  if (battlecard.weaknesses?.filter(Boolean).length) {
    lines.push(`${competitor.toUpperCase()} WEAKNESSES:\n${battlecard.weaknesses.filter(Boolean).map(w => `- ${w}`).join('\n')}`)
  }

  if (battlecard.differentiators?.filter(d => d?.point).length) {
    const diffs = battlecard.differentiators!
      .filter(d => d?.point)
      .map(d => `- ${d!.point}: "${d!.talkTrack}"`)
      .join('\n')
    lines.push(`OUR DIFFERENTIATORS:\n${diffs}`)
  }

  if (battlecard.pricing) {
    lines.push(`THEIR PRICING: ${battlecard.pricing}`)
  }

  if (battlecard.idealCustomerProfile) {
    lines.push(`THEIR ICP: ${battlecard.idealCustomerProfile}`)
  }

  if (battlecard.objections?.filter(o => o?.objection).length) {
    const objs = battlecard.objections!
      .filter(o => o?.objection)
      .map(o => `- "${o!.objection}" → ${o!.rebuttal}`)
      .join('\n')
    lines.push(`KEY OBJECTIONS:\n${objs}`)
  }

  return lines.join('\n\n')
}

// Minimal markdown-to-JSX renderer for bold + paragraphs.
// Handles **bold** inline spans and double-newline paragraph breaks.
// Full markdown libraries (remark, react-markdown) are intentionally avoided
// to keep the bundle small for this single use case.
function RenderMarkdown({ text }: { text: string }) {
  return (
    <div className="space-y-3">
      {text.split('\n\n').map((block, i) => {
        const lines = block.split('\n')
        return (
          <div key={i} className="space-y-1">
            {lines.map((line, j) => {
              // Split on **bold** tokens to render inline <strong> elements
              const parts = line.split(/(\*\*[^*]+\*\*)/)
              return (
                <p key={j} className={line.startsWith('**') && line.endsWith('**') ? 'text-slate-200 font-semibold text-sm mt-3 first:mt-0' : 'text-slate-300 text-sm leading-relaxed'}>
                  {parts.map((part, k) =>
                    part.startsWith('**') && part.endsWith('**')
                      ? <strong key={k} className="text-slate-100 font-semibold">{part.slice(2, -2)}</strong>
                      : part
                  )}
                </p>
              )
            })}
          </div>
        )
      })}
    </div>
  )
}

/**
 * Panel rendered below each battlecard offering four preset deep-dive options.
 * Each preset fires a separate POST to /api/followup with the battlecard summary
 * as context and streams the plain-text response back via ReadableStream.
 *
 * State per tab (isolated because each BattlecardTab mounts its own instance):
 * - activePreset: which button is currently selected
 * - text: accumulated streamed response text
 * - isLoading: true while a request is in flight
 * - abortRef: AbortController ref to cancel the previous request when the user
 *   clicks a different preset before the current one finishes
 */
export function FollowUpPanel({ yourCompany, competitor, battlecard, analysisContext, companySize }: FollowUpPanelProps) {
  const [activePreset, setActivePreset] = useState<string | null>(null)
  const [text, setText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  // Ref (not state) so updating it doesn't trigger a re-render
  const abortRef = useRef<AbortController | null>(null)

  /**
   * Fires a streaming request to /api/followup for the given preset.
   * Cancels any in-flight request first so switching presets mid-stream
   * doesn't leave orphaned responses appending to the text state.
   *
   * Uses native fetch + ReadableStream rather than the AI SDK's useObject/
   * useCompletion hooks because this is unstructured text (not JSON schema),
   * and we need per-tab isolated state without lifting it to a parent.
   */
  const runPreset = async (presetId: string) => {
    // Cancel any previous in-flight request before starting a new one
    if (abortRef.current) abortRef.current.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setActivePreset(presetId)
    setText('')
    setIsLoading(true)

    try {
      const res = await fetch('/api/followup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,   // Allows this request to be aborted
        body: JSON.stringify({
          yourCompany,
          competitor,
          presetType: presetId,                                        // Which of the 4 prompts to use
          battlecardSummary: buildBattlecardSummary(battlecard, competitor), // Compressed context for the LLM
          analysisContext,
          companySize,
        }),
      })

      if (!res.ok || !res.body) {
        setText('Failed to generate. Please try again.')
        setIsLoading(false)
        return
      }

      // Read the response as a stream, appending chunks to text as they arrive
      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        // { stream: true } tells the decoder more chunks are coming — prevents
        // it from flushing incomplete multi-byte characters prematurely
        setText(prev => prev + decoder.decode(value, { stream: true }))
      }
    } catch (err) {
      // AbortError is expected when the user switches presets — don't show error
      if ((err as Error).name !== 'AbortError') {
        setText('Failed to generate. Please try again.')
      }
    } finally {
      setIsLoading(false)
    }
  }

  const activeLabel = PRESETS.find(p => p.id === activePreset)?.label

  return (
    <div className="mt-3 rounded-2xl border border-indigo-900/25 bg-indigo-950/10 overflow-hidden">
      {/* Top accent line signals this is a separate AI-powered action zone */}
      <div className="h-px bg-gradient-to-r from-transparent via-indigo-500/25 to-transparent" />
      {/* Header + preset buttons */}
      <div className="px-5 py-4 border-b border-indigo-900/20">
        <p className="text-[10px] text-indigo-400/50 uppercase tracking-widest font-medium mb-3">
          Dive Deeper
        </p>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              onClick={() => runPreset(id)}
              disabled={isLoading}
              className={[
                'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border',
                activePreset === id
                  ? 'bg-indigo-600 border-indigo-500 text-white'
                  : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200 hover:border-slate-600 disabled:opacity-50 disabled:cursor-not-allowed',
              ].join(' ')}
            >
              <Icon size={12} />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Response area — three display states:
          1. isLoading && !text  → spinner only
          2. text && isLoading   → content + "Generating" pulse badge
          3. text && !isLoading  → final content, no badge */}
      {(text || isLoading) && (
        <div className="px-5 py-4">
          {isLoading && !text && (
            <div className="flex items-center gap-2 text-sm text-slate-500 py-4">
              <Loader2 size={14} className="animate-spin text-indigo-400" />
              {activePreset === 'technical-deep-dive' ? 'Researching technical docs...' : `Generating ${activeLabel}...`}
            </div>
          )}
          {text && (
            <>
              <div className="flex items-center gap-2 mb-4">
                <span className="text-[10px] text-indigo-500/70 font-bold uppercase tracking-widest">
                  {activeLabel}
                </span>
                {isLoading && (
                  <span className="flex items-center gap-1 text-[10px] text-indigo-400 animate-pulse">
                    <span className="w-1 h-1 rounded-full bg-indigo-400 inline-block" />
                    Generating
                  </span>
                )}
              </div>
              <RenderMarkdown text={text} />
            </>
          )}
        </div>
      )}
    </div>
  )
}
