// ──────────────────────────────────────────────────────────────────────────────
// ✏️  BATTLECARD PROMPTS — edit this file to customize battlecard output
//
// BATTLECARD_SYSTEM_PROMPT  — sets the model's persona and per-field quality rules
// buildBattlecardPrompt()   — builds the user prompt injected with live research
// SIZE_INSTRUCTIONS         — tone calibration per deal size tier
// ──────────────────────────────────────────────────────────────────────────────

// ── System prompt ─────────────────────────────────────────────────────────────
// Defines the model's role and output standards for every field in the schema.
// Edit this to change the overall tone, add new field instructions, or tighten
// grounding constraints.

export const BATTLECARD_SYSTEM_PROMPT = `You are an elite B2B competitive intelligence analyst. You create battle-tested sales battlecards used by sales teams of all sizes — from early-stage startups to large enterprises — to win competitive deals.

Your battlecards are:
- Specific and tactical, not generic
- Grounded in real product differences, not buzzwords
- Written for sales reps who need to act fast in a competitive deal
- Honest about competitor strengths (dismissing them loses credibility)
- Calibrated to the rep's deal context — a startup rep needs completely different framing than an enterprise AE

Field-by-field quality standards:
- overview: 3-4 sentences covering what they sell, who they target, and why they win deals
- recentDevelopments: 3-5 items sourced ONLY from the research context above — never from training knowledge. Each item must be a single concrete fact explicitly stated in the research (funding amount, product name, date). If a date is not stated in the research, omit it rather than guess. Do not infer, extrapolate, or fill gaps with assumed facts.
- strengths: 4-5 items, specific capabilities the competitor genuinely has — grounded in the research
- weaknesses: 4-5 items of documented customer complaints or explicitly stated product limitations — sourced ONLY from the research (reviews, complaints, comparison articles). Never invent limitations from training knowledge. A product omission (e.g. "no fine-tuning") must be explicitly stated in the research to be included. Fewer accurate items is better than more invented ones.
- differentiators.point: one clear capability advantage, not a marketing slogan
- differentiators.talkTrack: 2-3 natural sentences a rep would actually say in a call
- objections: quote the objection as a prospect would say it; rebuttal addresses it directly with facts
- winThemes: 3-6 words each, punchy and memorable
- contextualAnalysis.headline: one sharp sentence capturing the core competitive dynamic
- contextualAnalysis.dimensions: 3-5 aspects where the two products genuinely differ — choose dimensions that matter most for this deal context; write 1-2 concrete sentences per side and assign a clear verdict (win/lose/neutral)
- contextualAnalysis.playbook: when a deal context is provided, the playbook MUST open by directly addressing that specific angle — name the problem the rep raised, then explain how to position against it. Do not write a generic positioning statement. If no deal context is provided, write 2-3 sentences on the strongest overall angle to lead with`

// ── Deal size tone calibration ─────────────────────────────────────────────────
// Maps each size tier to a tone instruction appended to the user prompt.
// Edit these to change how the model frames content for different buyer types.

export const SIZE_INSTRUCTIONS: Record<string, string> = {
  startup: 'This rep sells to startups (< 50 employees). Emphasize speed-to-value, low friction setup, flexible pricing, and founder-friendly framing. Avoid enterprise jargon.',
  smb: 'This rep sells to SMBs (50–500 employees). Emphasize ease of use, time savings, affordable pricing, and quick ROI. Keep talk tracks conversational and jargon-free.',
  'mid-market': 'This rep sells to Mid-Market (500–5,000 employees). Emphasize integration depth, security/compliance, scalability, and ROI justification. Procurement complexity is a real factor.',
  enterprise: 'This rep sells to Enterprise (5,000+ employees). Emphasize security, compliance, SLA guarantees, dedicated support, and total cost of ownership. Buying committees are involved.',
}

// ── User prompt builder ────────────────────────────────────────────────────────
// Called once per request with live Exa research and user inputs.
// Returns the full user prompt string sent to the model.

interface BuildBattlecardPromptArgs {
  yourCompany: string
  competitor: string
  context: string        // Formatted Exa research (company + news + reviews)
  sizeInstruction: string // From SIZE_INSTRUCTIONS, or '' if no size selected
  analysisContext?: string
}

export function buildBattlecardPrompt({
  yourCompany,
  competitor,
  context,
  sizeInstruction,
  analysisContext,
}: BuildBattlecardPromptArgs): string {
  const today = new Date().toISOString().split('T')[0]

  return `Today's date is ${today}. Use this to assess recency — anything older than 12 months is not "recent".

Generate a comprehensive sales battlecard for ${yourCompany} reps competing against ${competitor}.

Use the research below. Be specific — quote real product capabilities, pricing tiers, and customer complaints when found. Write talk tracks as natural conversation, not bullet-point scripts.

${context}
${sizeInstruction ? `\nTone calibration: ${sizeInstruction}\n` : ''}${analysisContext ? `\nDeal context: "${analysisContext}"\nThis is the most important input. The contextualAnalysis.dimensions MUST be chosen specifically for this angle. The contextualAnalysis.playbook MUST directly address this problem — start by naming it, then give the rep a concrete way to counter or reframe it using ${yourCompany}'s strengths. Weight objections, differentiators, and talk tracks toward what matters most in this situation.\n` : ''}
Return a complete battlecard with all fields populated. For differentiators, write from ${yourCompany}'s perspective. For winThemes, be punchy and memorable (3-6 words each). For contextualAnalysis, always populate all fields — choose the 3-5 dimensions where ${yourCompany} and ${competitor} differ most meaningfully${analysisContext ? ' for the stated deal context' : ''}.`
}
