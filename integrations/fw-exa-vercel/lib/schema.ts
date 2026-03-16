import { z } from 'zod'

export const BattlecardSchema = z.object({
  competitor: z
    .string()
    .describe('The competitor company name'),

  tagline: z
    .string()
    .describe('A one-line positioning statement describing how the competitor markets themselves'),

  overview: z
    .string()
    .describe(
      'A 3-4 sentence strategic overview: what they sell, who they target, their market position, and why they win deals',
    ),

  recentDevelopments: z
    .array(z.string())
    .describe(
      'Recent developments in the last 6-12 months: product launches, funding, acquisitions, leadership changes, major partnerships, or strategic pivots',
    ),

  strengths: z
    .array(z.string())
    .describe(
      'Genuine competitive strengths — be specific, not generic. What do they actually do well that causes us to lose deals?',
    ),

  weaknesses: z
    .array(z.string())
    .describe(
      'Real weaknesses and gaps that prospects complain about. These are exploitable angles in competitive deals',
    ),

  differentiators: z
    .array(
      z.object({
        point: z
          .string()
          .describe('The specific capability or advantage your company has over this competitor'),
        talkTrack: z
          .string()
          .describe(
            'The exact language a sales rep should use when prospects bring up this competitor — conversational, specific, persuasive',
          ),
      }),
    )
    .describe(
      'Key reasons why prospects should choose your company over this competitor, each with an actionable talk track',
    ),

  objections: z
    .array(
      z.object({
        objection: z
          .string()
          .describe('The actual objection a prospect raises, quoted naturally as they would say it'),
        rebuttal: z
          .string()
          .describe('A direct, confident, and factual rebuttal that acknowledges the concern before redirecting'),
      }),
    )
    .describe('Most common competitive objections and how to handle them effectively'),

  pricing: z
    .string()
    .describe(
      'Known pricing model, published tiers, and approximate costs. Note if pricing is opaque or if competitors typically discount heavily',
    ),

  idealCustomerProfile: z
    .string()
    .describe(
      'The type of customer this competitor typically wins: company size, industry, use case, and buyer persona',
    ),

  winThemes: z
    .array(z.string())
    .describe(
      'Short, punchy themes that capture WHY your company wins competitive deals against this competitor — 3-6 words each',
    ),

  contextualAnalysis: z
    .object({
      headline: z
        .string()
        .describe('One-line summary of the core competitive dynamic in this deal context'),
      dimensions: z
        .array(
          z.object({
            aspect: z.string().describe('The dimension being compared, e.g. "Pricing model", "Onboarding speed", "AI capabilities"'),
            yourPosition: z.string().describe("Brief description of your company's position on this dimension (1-2 sentences)"),
            theirPosition: z.string().describe("Brief description of the competitor's position on this dimension (1-2 sentences)"),
            verdict: z
              .enum(['win', 'lose', 'neutral'])
              .describe('win = your company has the advantage, lose = competitor has the advantage, neutral = roughly equal'),
          }),
        )
        .describe('3-5 head-to-head comparison dimensions most relevant to the deal context'),
      playbook: z
        .string()
        .describe('2-3 sentences: what to lead with and how to position given this specific competitive angle'),
    })
    .describe('Head-to-head comparison and positioning playbook — always populate with the most relevant dimensions'),
})

export type Battlecard = z.infer<typeof BattlecardSchema>
