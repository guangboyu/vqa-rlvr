import { useState } from 'react'
import Arena from './components/Arena'
import Dashboard from './components/Dashboard'
import Gallery from './components/Gallery'

type View = 'gallery' | 'results' | 'arena'

const VIEWS: { key: View; label: string }[] = [
  { key: 'gallery', label: 'Examples' },
  { key: 'results', label: 'Results' },
  { key: 'arena', label: 'Live comparison' },
]

const LINKS = [
  { href: 'https://guangboyu.github.io/vqa-rlvr/report/', label: 'Report' },
  { href: 'https://github.com/guangboyu/vqa-rlvr', label: 'Code' },
]

export default function App() {
  const [view, setView] = useState<View>('gallery')

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 border-b border-rule bg-paper/95 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-8 px-6">
          <span className="hidden font-serif text-[19px] font-medium tracking-[-0.005em] md:inline">
            What RL adds to VQA
          </span>
          <nav className="flex h-full gap-6 whitespace-nowrap text-sm" aria-label="Views">
            {VIEWS.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setView(key)}
                aria-current={view === key ? 'page' : undefined}
                className={`-mb-px border-b-2 transition-colors ${
                  view === key
                    ? 'border-ink text-ink'
                    : 'border-transparent text-ink-3 hover:text-ink'
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
          <div className="ml-auto hidden gap-5 text-sm text-ink-3 sm:flex">
            {LINKS.map(({ href, label }) => (
              <a key={href} href={href} target="_blank" rel="noreferrer" className="hover:text-ink">
                {label} ↗
              </a>
            ))}
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 pb-24 pt-12">
        {view === 'arena' ? <Arena /> : view === 'results' ? <Dashboard /> : <Gallery />}
      </main>
    </div>
  )
}
