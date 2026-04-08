import { streamText } from 'ai'
import { createOpenAI } from '@ai-sdk/openai'
import Exa from 'exa-js'
import { FOLLOWUP_SYSTEM_PROMPT, buildFollowupPrompt } from '@/lib/prompts/followup'

export const maxDuration = 60

if (!process.env.FIREWORKS_API_KEY) {
  console.error('[followup] FIREWORKS_API_KEY is not set — model calls will fail')
}
if (!process.env.EXA_API_KEY) {
  console.error('[followup] EXA_API_KEY is not set — technical deep-dive Exa search will fail')
}

const exa = new Exa(process.env.EXA_API_KEY ?? '')

const fireworks = createOpenAI({
  apiKey: process.env.FIREWORKS_API_KEY ?? '',
  baseURL: process.env.VERCEL_TEAM_SLUG
    ? `https://gateway.ai.vercel.app/v1/${process.env.VERCEL_TEAM_SLUG}/live-battlecards/fireworks`
    : 'https://api.fireworks.ai/inference/v1',
})


export async function POST(req: Request) {
  const { yourCompany, competitor, presetType, battlecardSummary, analysisContext, companySize } = await req.json()

  if (!yourCompany?.trim() || !competitor?.trim() || !presetType) {
    return new Response(
      JSON.stringify({ error: 'yourCompany, competitor, and presetType are required' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } },
    )
  }

  console.log(`[followup] Generating "${presetType}" for ${yourCompany} vs ${competitor}`)

  let techContext: string | undefined
  if (presetType === 'technical-deep-dive') {
    // Extra Exa search fires only for technical-deep-dive — other presets don't
    // need developer docs and we avoid the added latency for those requests.
    try {
      const techResults = await exa.searchAndContents(
        `${competitor} developer API REST endpoints webhooks rate limits documentation`,
        {
          type: 'neural',      // Semantic search — finds dev docs even without exact keyword match
          livecrawl: 'always', // Re-crawls at request time for up-to-date API docs
          numResults: 3,       // Number of pages to return
          text: { maxCharacters: 1500 }, // Max extracted text per page
          // No category filter — 'company' was too restrictive and returned homepages;
          // removing it lets Exa surface developer portals and API reference pages
        },
      )
      techContext = techResults.results
        .map((r) => `[${r.title ?? r.url}]\n${(r.text ?? '').slice(0, 1200)}`)
        .join('\n\n---\n\n')
      console.log(`[followup] Exa technical search: ${techResults.results.length} results for "${competitor}"`)
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      console.error(`[followup] Exa technical search failed for "${competitor}":`, message)
      // Continue without tech context — model will use battlecard summary only
    }
  }

  try {
    // streamText is used here (vs streamObject in the battlecard route) because
    // follow-up presets return free-text markdown, not structured JSON.
    // The client reads the raw text stream via ReadableStream in followup-panel.tsx.
    const result = streamText({
      model: fireworks('accounts/fireworks/models/kimi-k2p5'),
      system: FOLLOWUP_SYSTEM_PROMPT,
      prompt: buildFollowupPrompt(presetType, yourCompany, competitor, battlecardSummary, analysisContext, companySize, techContext),
    })

    return result.toTextStreamResponse()
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    console.error(`[followup] Fireworks streamText failed for "${presetType}" (${competitor}):`, message)
    return new Response(
      JSON.stringify({ error: 'Failed to generate follow-up', detail: message }),
      { status: 500, headers: { 'Content-Type': 'application/json' } },
    )
  }
}
