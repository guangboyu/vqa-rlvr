import { useState } from 'react'
import Arena from './components/Arena'
import Dashboard from './components/Dashboard'

type View = 'arena' | 'dashboard'

export default function App() {
  const [view, setView] = useState<View>('arena')

  return (
    <div className="min-h-screen bg-ink-950 text-ink-100">
      <header className="border-b border-ink-800">
        <div className="mx-auto flex max-w-7xl items-center gap-8 px-6 py-4">
          <div>
            <h1 className="font-display text-2xl tracking-tight text-ink-50">
              VQA <span className="text-ember-500">Arena</span>
            </h1>
            <p className="text-xs text-ink-400">
              base vs SFT vs RLVR — Qwen3-VL post-training, live on one RTX 4090
            </p>
          </div>
          <nav className="ml-auto flex gap-1 font-mono text-sm">
            {(['arena', 'dashboard'] as const).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`rounded-md px-4 py-1.5 transition-colors ${
                  view === v ? 'bg-ink-700 text-ink-50' : 'text-ink-400 hover:text-ink-100'
                }`}
              >
                {v}
              </button>
            ))}
          </nav>
          <a
            href="https://github.com/guangboyu/vqa-rlvr"
            target="_blank"
            rel="noreferrer"
            className="font-mono text-xs text-ink-400 hover:text-ember-400"
          >
            github ↗
          </a>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-6 py-8">
        {view === 'arena' ? <Arena /> : <Dashboard />}
      </main>
    </div>
  )
}
