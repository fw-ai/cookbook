// ──────────────────────────────────────────────────────────────────────────────
// ✏️  FOLLOW-UP PROMPTS — edit this file to customize the four deep-dive presets
//
// FOLLOWUP_SYSTEM_PROMPT  — sets the model's persona for all four presets
// buildFollowupPrompt()   — returns the prompt for a specific preset
//
// Preset types: 'positioning-playbook' | 'technical-deep-dive' |
//               'discovery-questions' | 'champion-brief'
// ──────────────────────────────────────────────────────────────────────────────

// ── System prompt ─────────────────────────────────────────────────────────────
// Shared across all four presets. Edit to change overall tone or output format.

export const FOLLOWUP_SYSTEM_PROMPT = `You are an elite B2B sales strategist. Generate highly specific, tactical sales content grounded in the competitive intelligence provided. Never be generic — use exact feature names, product capabilities, and facts from the research. Use markdown headers and bullet points for structure. Write for a sales rep who needs to act fast in a live deal.`

// ── Prompt builder ─────────────────────────────────────────────────────────────
// Returns the full user prompt for a given preset. Each case below is one preset
// — edit the template strings to change what that preset generates.
//
// @param presetType       - Which preset to build ('positioning-playbook', etc.)
// @param yourCompany      - The rep's company name (the "winning" side)
// @param competitor       - The competitor being analyzed
// @param battlecardSummary - Compressed plain-text battlecard context from the client
// @param analysisContext  - Optional free-text deal angle from the rep
// @param companySize      - Optional tier: 'startup' | 'smb' | 'mid-market' | 'enterprise'
// @param techContext      - Extra Exa research, only populated for 'technical-deep-dive'

export function buildFollowupPrompt(
  presetType: string,
  yourCompany: string,
  competitor: string,
  battlecardSummary: string,
  analysisContext?: string,
  companySize?: string,
  techContext?: string,
): string {
  const sizeCtx = companySize ? ` The rep sells to ${companySize} accounts.` : ''
  const dealCtx = analysisContext ? `\nDeal angle: "${analysisContext}"` : ''
  const base = `Competitive intelligence for ${yourCompany} vs ${competitor}:\n${battlecardSummary}${dealCtx}${sizeCtx}`

  switch (presetType) {

    // ── Positioning Playbook ───────────────────────────────────────────────────
    // Gives the rep a memorable frame, technical advantages, reframes, and an
    // opening line for when the competitor comes up on a call.
    case 'positioning-playbook':
      return `${base}

Generate a positioning playbook for ${yourCompany} reps who encounter ${competitor} in a deal:

1. **The Frame** — name a memorable 3-5 word frame that captures why ${yourCompany} wins (e.g. "Built for scale from day one"). Explain it in 1-2 sentences.${analysisContext ? ` Open this section by directly naming the rep's problem: "${analysisContext}" — then show how ${yourCompany}'s position answers it.` : ''}

2. **Technical Advantages** — 3-4 specific capabilities where ${yourCompany} has a clear edge. For each: name the exact feature, explain what ${competitor} does instead, and state why it matters to the buyer.

3. **Reframing Their Strengths** — for each of ${competitor}'s top 2-3 strengths, write one sentence that reframes it as a limitation or trade-off without dismissing it.

4. **Opening Move** — write the 1-2 sentences a rep says when ${competitor} comes up on a call. Conversational, confident, not defensive.
${companySize ? `\nTone: calibrated for a ${companySize} deal.` : ''}`

    // ── Technical Deep-Dive ────────────────────────────────────────────────────
    // Architecture, capability comparison, hidden constraints, and integration story.
    // Receives extra Exa-sourced developer docs via techContext.
    case 'technical-deep-dive':
      return `${base}
${techContext ? `\n## Additional Technical Research\n${techContext}\n` : ''}
Generate a technical deep-dive comparing ${yourCompany} and ${competitor}. Ground every claim in the research above — do not add architectural details not stated in the context.

1. **Architecture & Design** — explain how the underlying architecture of each product differs and what that means for scalability, flexibility, or maintenance. Be specific about technical choices (APIs, data models, infrastructure patterns).

2. **Capability Comparison** — for 4-5 specific technical capabilities, show exactly what each product does. Use feature names. Assign a clear advantage to one side.

3. **Hidden Constraints** — 3 technical limitations in ${competitor} that prospects typically only discover post-sale. Explain why each constraint exists architecturally, not just that it exists.

4. **Integration & Extensibility** — how does each platform handle custom integrations, APIs, and developer extensibility? Where does ${yourCompany} have the edge?

5. **Proof Points** — 2-3 specific technical claims ${yourCompany} reps can make with confidence, grounded in the product capabilities above.
${companySize ? `\nTone: calibrated for a ${companySize} technical buyer.` : ''}`

    // ── Discovery Questions ────────────────────────────────────────────────────
    // 5 questions the rep asks to expose gaps without naming the competitor directly.
    case 'discovery-questions':
      return `${base}

Generate 5 discovery questions a ${yourCompany} rep asks when ${competitor} is in the deal. The questions should feel natural in a sales conversation — not interrogative.

Format each as:
**Question:** [the question, written conversationally]
**Surfaces:** [one sentence — what gap or pain point this is designed to expose]

Rules:
- Do not name ${competitor} directly in the questions
- Questions should lead the prospect to articulate the pain themselves
- Mix technical, operational, and business-level questions
${analysisContext ? `\nFocus the questions toward: "${analysisContext}"` : ''}${companySize ? `\nMatch the vocabulary and framing to a ${companySize} buyer.` : ''}`

    // ── Champion Brief ─────────────────────────────────────────────────────────
    // Talking points for an internal champion to advocate in their buying committee.
    case 'champion-brief':
      return `${base}

Write a champion enablement brief — talking points an internal champion uses to advocate for ${yourCompany} over ${competitor} in their buying committee. The champion may not be technical.

**The Business Case** (3 bullet points)
Each bullet: one concrete business reason to choose ${yourCompany}. Lead with outcomes, not features. Do not invent specific percentages or dollar figures — use directional language instead.

**Anticipated Committee Objections**
List the 3 most likely objections the buying committee raises about switching to or choosing ${yourCompany}, and the 1-2 sentence response the champion gives.

**The Risk of Choosing ${competitor}**
One paragraph (3-4 sentences). Frame the risk in business terms: cost of the wrong choice, what they give up, what future pain looks like. Not fear-mongering — honest and factual.

**The Close**
One sentence the champion says to close the internal conversation. Memorable and confident.
${companySize ? `\nTone: written for a ${companySize} buying committee.` : ''}`

    default:
      return base
  }
}
