import type { ReactNode } from 'react'
import { methodColor, methodOf } from '../lib/methods'

/** A small filled dot in the model's method color; the base model gets a ring. */
export function MethodDot({ name }: { name: string }) {
  const method = methodOf(name)
  const color = methodColor(method)
  return (
    <span
      aria-hidden
      className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
      style={method === 'base' ? { boxShadow: `inset 0 0 0 1.6px ${color}` } : { background: color }}
    />
  )
}

export function ModelName({ name, className = '' }: { name: string; className?: string }) {
  return (
    <span className={`inline-flex items-center gap-2 ${className}`}>
      <MethodDot name={name} />
      {name}
    </span>
  )
}

/** A row of mutually exclusive options. */
export function Segmented<T extends string>({ label, value, options, onChange, disabled }: {
  label?: string
  value: T
  options: { value: T; label: string }[]
  onChange: (value: T) => void
  disabled?: boolean
}) {
  return (
    <div className="flex items-center gap-3 text-sm">
      {label && <span className="text-ink-3">{label}</span>}
      <div role="radiogroup" aria-label={label} className="inline-flex rounded-md border border-line bg-paper p-0.5">
        {options.map((option) => {
          const active = option.value === value
          return (
            <button
              key={option.value}
              role="radio"
              aria-checked={active}
              disabled={disabled}
              onClick={() => onChange(option.value)}
              className={`rounded-[5px] px-3 py-1 transition-colors disabled:opacity-50 ${
                active ? 'bg-ink text-paper' : 'text-ink-2 hover:text-ink'
              }`}
            >
              {option.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}

/** A view's opening: serif title and a short lede. */
export function PageIntro({ title, children }: { title: string; children: ReactNode }) {
  return (
    <header className="max-w-3xl">
      <h1 className="font-serif text-[34px] font-medium leading-tight tracking-[-0.01em]">{title}</h1>
      <div className="mt-3 text-[15px] leading-relaxed text-ink-2">{children}</div>
    </header>
  )
}

/** A titled block with an optional note under the title. */
export function Section({ title, note, children }: {
  title: string
  note?: ReactNode
  children: ReactNode
}) {
  return (
    <section>
      <h2 className="font-serif text-[22px] font-medium">{title}</h2>
      {note && <p className="mt-1 max-w-3xl text-[13px] leading-relaxed text-ink-3">{note}</p>}
      <div className="mt-4">{children}</div>
    </section>
  )
}

/** A score from a recorded evaluation: 1 is a hit, below 1 partial credit, 0 a miss. */
export function Score({ label, value }: { label: string; value: number }) {
  const hit = value >= 1
  const miss = value === 0
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs tabular-nums ${miss ? 'text-miss' : 'text-ink-2'}`}>
      <span aria-hidden className="font-medium">{hit ? '✓' : miss ? '✕' : '◐'}</span>
      {label} {value.toFixed(2)}
    </span>
  )
}
