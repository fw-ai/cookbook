import { streamObject } from 'ai'
import { createOpenAI } from '@ai-sdk/openai'
import Exa from 'exa-js'
import { BattlecardSchema } from '@/lib/schema'
import { BATTLECARD_SYSTEM_PROMPT, SIZE_INSTRUCTIONS, buildBattlecardPrompt } from '@/lib/prompts/battlecard'

export const maxDuration = 60

if (!process.env.FIREWORKS_API_KEY) {
  console.error('[battlecard] FIREWORKS_API_KEY is not set — model calls will fail')
}
if (!process.env.EXA_API_KEY) {
  console.error('[battlecard] EXA_API_KEY is not set — Exa searches will fail')
}

const fireworks = createOpenAI({
  apiKey: process.env.FIREWORKS_API_KEY ?? '',
  baseURL: process.env.VERCEL_TEAM_SLUG
    ? `https://gateway.ai.vercel.app/v1/${process.env.VERCEL_TEAM_SLUG}/live-battlecards/fireworks`
    : 'https://api.fireworks.ai/inference/v1',
})

const exa = new Exa(process.env.EXA_API_KEY ?? '')

const currentYear = new Date().getFullYear()

export async function POST(req: Request) {
  const { yourCompany, competitor, analysisContext, companySize } = await req.json()

  if (!yourCompany?.trim() || !competitor?.trim()) {
    return new Response(
      JSON.stringify({ error: 'yourCompany and competitor are required' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } },
    )
  }

  console.log(`[battlecard] Generating: ${yourCompany} vs ${competitor}${companySize ? ` (${companySize})` : ''}${analysisContext ? ` | context: "${analysisContext}"` : ''}`)

  // Three Exa searches run in parallel to minimize latency.
  // Each has an independent .catch() so a single search failure doesn't crash
  // the entire request — the battlecard generates with whatever context remains.
  const [companyResults, newsResults, reviewResults] = await Promise.all([

    // Search 1: Official product pages — pricing tiers, feature lists, customer segments
    exa.searchAndContents(
      `${competitor} product features pricing plans enterprise SMB vs ${yourCompany}`,
      {
        type: 'neural',        // Semantic/embedding search — finds conceptually relevant pages
        category: 'company',   // Restricts results to official company/product pages
        livecrawl: 'always',   // Re-crawls pages at request time for fresh pricing/feature data
        numResults: 4,         // Number of pages to return
        text: { maxCharacters: 2500 }, // Max extracted text per page
      },
    ).catch((err) => {
      console.error(`[battlecard] Exa company search failed for "${competitor}":`, err.message)
      return { results: [] }
    }),

    // Search 2: News articles — funding rounds, product launches, partnerships, acquisitions
    exa.searchAndContents(
      `${competitor} news funding product launch partnership acquisition ${currentYear - 1} ${currentYear}`,
      {
        type: 'neural',
        category: 'news',      // Restricts results to news articles (not company blogs)
        livecrawl: 'always',
        numResults: 4,
        text: { maxCharacters: 1500 },
      },
    ).catch((err) => {
      console.error(`[battlecard] Exa news search failed for "${competitor}":`, err.message)
      return { results: [] }
    }),

    // Search 3: Analyst comparisons and reviews — surfaces weaknesses, complaints, alternatives
    exa.searchAndContents(
      `${competitor} vs ${yourCompany} comparison review complaints alternatives`,
      {
        type: 'neural',
        category: 'research paper', // Targets analyst reports and head-to-head comparisons (e.g. G2, Datamation)
        livecrawl: 'always',
        numResults: 3,
        text: { maxCharacters: 1200 },
      },
    ).catch((err) => {
      console.error(`[battlecard] Exa review search failed for "${competitor}":`, err.message)
      return { results: [] }
    }),
  ])

  console.log(`[battlecard] Exa results — company: ${companyResults.results.length}, news: ${newsResults.results.length}, reviews: ${reviewResults.results.length}`)

  const formatResults = (results: { results: { url: string; title?: string | null; text?: string | null }[] }) =>
    results.results
      .map((r) => `[${r.title ?? r.url}]\n${(r.text ?? '').slice(0, 1200)}`)
      .join('\n\n---\n\n')

  const context = `
## Company & Product Intelligence
${formatResults(companyResults)}

## Recent News & Developments
${formatResults(newsResults)}

## Customer Reviews & Complaints
${formatResults(reviewResults)}
`.trim()

  const sizeInstruction = companySize ? SIZE_INSTRUCTIONS[companySize] ?? '' : ''

  try {
    // streamObject streams a structured JSON response that conforms to BattlecardSchema.
    // As chunks arrive, the AI SDK progressively hydrates the partial object on the client
    // via experimental_useObject — enabling the UI to render sections as they complete.
    //
    // structuredOutputs: true — sends response_format: json_schema to Fireworks, which
    // instructs the model to follow the schema strictly. Required for reliable structured
    // output from Kimi K2.5; without it, json_object mode produces inconsistent results.
    // mode: 'json' — tells the AI SDK to use JSON output mode (vs 'tool' which Fireworks
    // does not support — it outputs tool calls as plain text in delta.content).
    const result = streamObject({
      model: fireworks('accounts/fireworks/models/kimi-k2p5', {
        structuredOutputs: true, // Sends json_schema format — required for Fireworks/Kimi reliability
      }),
      schema: BattlecardSchema,
      mode: 'json',             // Uses response_format: json_schema (not tool calls)
      system: BATTLECARD_SYSTEM_PROMPT,
      prompt: buildBattlecardPrompt({ yourCompany, competitor, context, sizeInstruction, analysisContext }),
    })

    return result.toTextStreamResponse()
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    console.error(`[battlecard] Fireworks streamObject failed for "${competitor}":`, message)
    return new Response(
      JSON.stringify({ error: 'Failed to generate battlecard', detail: message }),
      { status: 500, headers: { 'Content-Type': 'application/json' } },
    )
  }
}
