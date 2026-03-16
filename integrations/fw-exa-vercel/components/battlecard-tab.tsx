'use client'

// experimental_useObject is the correct export name in ai v4.1.0 — aliased to
// useObject for cleaner usage. A non-experimental useObject export will be
// available in a future AI SDK version; swap the import when upgrading past 4.1.0.
import { experimental_useObject as useObject } from 'ai/react'
import { useEffect } from 'react'
import { BattlecardSchema } from '@/lib/schema'
import { BattlecardView } from '@/components/battlecard'
import { FollowUpPanel } from '@/components/followup-panel'
import { Loader2, AlertCircle } from 'lucide-react'

interface BattlecardTabProps {
  yourCompany: string
  competitor: string
  analysisContext?: string
  companySize?: string
  onStatusChange?: (status: TabStatus) => void
}

/**
 * Self-contained tab component that owns its own streaming battlecard state.
 *
 * Each competitor gets its own BattlecardTab instance. By mounting all tabs
 * simultaneously (with display:none hiding inactive ones), all battlecards
 * generate in parallel — the user doesn't wait when switching tabs.
 *
 * Data flow: submit() → POST /api/battlecard → streamObject response
 * → BattlecardSchema partial hydration → BattlecardView renders as chunks arrive.
 */
export function BattlecardTab({ yourCompany, competitor, analysisContext, companySize, onStatusChange }: BattlecardTabProps) {
  // useObject streams structured JSON from the API route and progressively
  // hydrates the schema as chunks arrive. `object` is DeepPartial<Battlecard>
  // while streaming, and a complete Battlecard when isLoading is false.
  const { object, submit, isLoading, error } = useObject({
    api: '/api/battlecard',       // POSTs to app/api/battlecard/route.ts
    schema: BattlecardSchema,     // Zod schema used to validate + type the streamed JSON
  })

  useEffect(() => {
    submit({ yourCompany, competitor, analysisContext, companySize })
  // Deps intentionally empty — we only want to fire once on mount.
  // Props are captured at submit time; re-submitting on prop changes
  // would restart generation mid-stream.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Report loading/complete/error status to parent so the tab bar can show live indicators
  useEffect(() => {
    if (error) {
      onStatusChange?.('error')
    } else if (isLoading) {
      onStatusChange?.('loading')
    } else if (object) {
      onStatusChange?.('complete')
    }
  }, [isLoading, error, object]) // eslint-disable-line react-hooks/exhaustive-deps

  if (error) {
    return (
      <div className="flex flex-col items-center gap-3 py-20 text-center">
        <AlertCircle size={24} className="text-red-400" />
        <p className="text-sm text-red-400">Failed to generate battlecard for {competitor}.</p>
        <p className="text-xs text-slate-600">{error.message}</p>
      </div>
    )
  }

  if (!object && isLoading) {
    return (
      <div className="flex flex-col items-center gap-4 py-20">
        <div className="flex flex-col items-center gap-3">
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <Loader2 size={16} className="animate-spin text-indigo-400" />
            Researching...
          </div>
          <div className="text-xs text-slate-600">
            Researching {competitor} · This takes 10–20 seconds
          </div>
        </div>
      </div>
    )
  }

  if (!object) return null

  return (
    <>
      <BattlecardView
        data={object}
        isStreaming={isLoading}
        yourCompany={yourCompany}
        analysisContext={analysisContext}
        companySize={companySize}
      />
      {/* FollowUpPanel only renders after generation completes — it needs the
          full battlecard object to build the summary sent to /api/followup */}
      {!isLoading && (
        <FollowUpPanel
          yourCompany={yourCompany}
          competitor={competitor}
          battlecard={object}
          analysisContext={analysisContext}
          companySize={companySize}
        />
      )}
    </>
  )
}

// Exported type for tab status tracking in parent
export type TabStatus = 'loading' | 'complete' | 'error'
