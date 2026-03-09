import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Live Battlecards — AI Competitive Intelligence',
  description:
    'Real-time competitive intelligence for sales teams, powered by Exa and Fireworks AI.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={inter.className}>
      <body className="min-h-screen bg-slate-950">{children}</body>
    </html>
  )
}
