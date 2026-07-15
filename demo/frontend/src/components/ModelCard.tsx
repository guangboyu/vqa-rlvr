import type { ModelSpec } from '../lib/api'

export interface CardState {
  text: string
  status: 'idle' | 'streaming' | 'done'
  startedAt: number | null
  finishedAt: number | null
}

const ANSWER_RE = /<answer>([\s\S]*?)<\/answer>/g

function splitAnswer(text: string): { reasoning: string; answer: string | null } {
  const matches = [...text.matchAll(ANSWER_RE)]
  if (matches.length === 0) return { reasoning: text, answer: null }
  const last = matches[matches.length - 1]
  return { reasoning: text.slice(0, last.index).trim(), answer: last[1].trim() }
}

export default function ModelCard({ spec, state }: { spec: ModelSpec; state: CardState }) {
  const { reasoning, answer } = splitAnswer(state.text)
  const seconds =
    state.startedAt && (state.finishedAt ?? Date.now())
      ? (((state.finishedAt ?? Date.now()) - state.startedAt) / 1000).toFixed(1)
      : null

  return (
    <div className="flex min-h-[16rem] flex-col rounded-lg border border-ink-700 bg-ink-900">
      <div className="flex items-baseline justify-between border-b border-ink-700 px-4 py-3">
        <div>
          <h3 className="font-display text-lg text-ink-50">{spec.label}</h3>
          <p className="text-xs text-ink-400">{spec.description}</p>
        </div>
        {state.status === 'streaming' && (
          <span className="streaming-dot h-2.5 w-2.5 shrink-0 rounded-full bg-ember-500" />
        )}
        {state.status === 'done' && seconds && (
          <span className="shrink-0 font-mono text-xs text-ink-400">{seconds}s</span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3">
        {state.status === 'idle' && !state.text ? (
          <p className="text-sm italic text-ink-600">Waiting for a question…</p>
        ) : (
          <>
            {reasoning && (
              <p className="whitespace-pre-wrap font-mono text-[13px] leading-relaxed text-ink-300">
                {reasoning}
              </p>
            )}
            {answer !== null && (
              <div className="mt-3 rounded-md border border-ember-600/40 bg-ember-500/10 px-3 py-2">
                <span className="mr-2 font-mono text-[10px] uppercase tracking-widest text-ember-400">
                  answer
                </span>
                <span className="font-display text-xl text-ink-50">{answer}</span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
