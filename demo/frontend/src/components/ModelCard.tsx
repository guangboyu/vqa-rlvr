import type { ModelSpec } from '../lib/api'
import { methodColor, methodOf, responseParts } from '../lib/methods'
import { ModelName } from './ui'

export interface CardState {
  text: string
  status: 'idle' | 'streaming' | 'done'
  startedAt: number | null
  finishedAt: number | null
}

export default function ModelCard({ spec, state }: { spec: ModelSpec; state: CardState }) {
  const { reasoning, answer } = responseParts(state.text, state.status === 'done')
  const seconds = state.startedAt
    ? (((state.finishedAt ?? Date.now()) - state.startedAt) / 1000).toFixed(1)
    : null
  const color = methodColor(methodOf(spec.key))

  return (
    <div className={`flex min-h-[18rem] flex-col rounded-lg border border-rule bg-paper ${spec.loaded ? '' : 'opacity-60'}`}>
      <div className="border-b border-rule px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <ModelName name={spec.label} className="text-[15px] font-medium" />
          {state.status === 'streaming' && (
            <span className="streaming-dot h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />
          )}
          {state.status === 'done' && seconds && (
            <span className="shrink-0 text-xs tabular-nums text-ink-3">{seconds} s</span>
          )}
        </div>
        <p className="mt-1 text-xs leading-relaxed text-ink-3">{spec.description}</p>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        {state.status === 'idle' && !state.text ? (
          <p className="text-sm text-ink-4">{spec.loaded ? 'The answer appears here.' : 'Not loaded'}</p>
        ) : (
          <>
            {reasoning && (
              <p className="whitespace-pre-wrap text-[13.5px] leading-relaxed text-ink-2">{reasoning}</p>
            )}
            {answer !== null && (
              <div className="mt-4 border-l-2 bg-wash px-3 py-2" style={{ borderColor: color }}>
                <span className="eyebrow mr-2">Answer</span>
                <span className="font-serif text-xl">{answer}</span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
