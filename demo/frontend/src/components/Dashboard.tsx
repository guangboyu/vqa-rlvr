import { Fragment, useEffect, useState } from 'react'
import { fetchResults, type ResultsPayload } from '../lib/api'

const DATASETS = ['vqav2', 'gqa', 'clevr', 'textvqa'] as const
const DATASET_LABELS: Record<string, string> = {
  vqav2: 'VQAv2',
  gqa: 'GQA',
  clevr: 'CLEVR',
  textvqa: 'TextVQA (OOD)',
}
const BASELINE = 'zero_shot_2b_reasoning'

const MAIN_ROWS: [string, string][] = [
  ['2B zero-shot', 'zero_shot_2b_reasoning'],
  ['2B + SFT', 'sft_2b_reasoning'],
  ['2B + GRPO (RLVR)', 'grpo_2b_main_base_reasoning'],
  ['2B + SFT + GRPO', 'grpo_2b_main_sft_reasoning'],
  ['8B zero-shot', 'zero_shot_8b_reasoning'],
  ['8B + SFT', 'sft_8b_reasoning'],
]

const TRANSFER_ROWS: [string, string][] = [
  ['CLEVR only', 'grpo_2b_clevr_base_reasoning'],
  ['VQAv2 only', 'grpo_2b_vqav2only_reasoning'],
  ['GQA only', 'grpo_2b_gqaonly_reasoning'],
  ['VQAv2 + GQA', 'grpo_2b_main_base_reasoning'],
]

// Diverging encoding for deltas — validated poles on the dark surface
// (scripts/validate_palette.js: #5b8dd6 / #b87d26, neutral #3a4250 midpoint).
function deltaColor(delta: number, maxAbs: number): string {
  const t = Math.min(Math.abs(delta) / maxAbs, 1)
  const from = [0x3a, 0x42, 0x50]
  const to = delta >= 0 ? [0xb8, 0x7d, 0x26] : [0x5b, 0x8d, 0xd6]
  const mix = from.map((f, i) => Math.round(f + (to[i] - f) * t))
  return `rgb(${mix[0]},${mix[1]},${mix[2]})`
}

function Section({ title, note, children }: {
  title: string; note?: string; children: React.ReactNode
}) {
  return (
    <section className="rounded-lg border border-ink-700 bg-ink-900 p-5">
      <h2 className="font-display text-lg text-ink-50">{title}</h2>
      {note && <p className="mt-0.5 text-xs text-ink-400">{note}</p>}
      <div className="mt-4">{children}</div>
    </section>
  )
}

export default function Dashboard() {
  const [data, setData] = useState<ResultsPayload & { ablation?: { weight: number; steps: number; retention: number; run_id: string }[] } | null>(null)

  useEffect(() => {
    fetchResults().then(setData)
  }, [])

  if (!data) return <p className="text-sm text-ink-400">Loading results…</p>
  const { runs } = data
  const baseline = runs[BASELINE] ?? {}
  const deltas = TRANSFER_ROWS.map(([, id]) =>
    DATASETS.map((d) => (runs[id]?.[d] ?? 0) - (baseline[d] ?? 0)),
  )
  const maxAbs = Math.max(...deltas.flat().map(Math.abs))

  return (
    <div className="space-y-6">
      <Section
        title="Chain-of-thought accuracy across post-training stages"
        note="Full eval sets, greedy decoding. Every number traces to a results/runs JSON with git SHA."
      >
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-ink-700 text-left font-mono text-xs text-ink-400">
              <th className="py-2 pr-4 font-normal">model</th>
              {DATASETS.map((d) => (
                <th key={d} className="px-3 py-2 text-right font-normal">{DATASET_LABELS[d]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MAIN_ROWS.map(([label, id]) => (
              <tr key={id} className="border-b border-ink-800 last:border-0">
                <td className="py-2 pr-4 text-ink-100">{label}</td>
                {DATASETS.map((d) => {
                  const v = runs[id]?.[d]
                  const best = Math.max(...MAIN_ROWS.map(([, r]) => runs[r]?.[d] ?? 0))
                  return (
                    <td key={d} className={`px-3 py-2 text-right font-mono ${v === best ? 'text-ember-400' : 'text-ink-300'}`}>
                      {v?.toFixed(1) ?? '—'}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      <div className="grid gap-6 lg:grid-cols-2">
        <Section
          title="Transfer matrix — where does RLVR generalize?"
          note="Δ accuracy vs zero-shot (reasoning mode). Rows: GRPO training set. Amber = gain, blue = loss."
        >
          <div className="grid" style={{ gridTemplateColumns: 'auto repeat(4, 1fr)', gap: 2 }}>
            <div />
            {DATASETS.map((d) => (
              <div key={d} className="px-1 pb-1 text-center font-mono text-[10px] text-ink-400">
                {DATASET_LABELS[d]}
              </div>
            ))}
            {TRANSFER_ROWS.map(([label], i) => (
              <Fragment key={label}>
                <div className="flex items-center pr-2 font-mono text-[11px] text-ink-300">
                  {label}
                </div>
                {DATASETS.map((d, j) => {
                  const delta = deltas[i][j]
                  return (
                    <div
                      key={`${label}-${d}`}
                      title={`train ${label} → eval ${DATASET_LABELS[d]}: ${delta >= 0 ? '+' : ''}${delta.toFixed(1)}`}
                      className="flex h-12 items-center justify-center rounded font-mono text-sm text-ink-50"
                      style={{ background: deltaColor(delta, maxAbs) }}
                    >
                      {delta >= 0 ? '+' : ''}{delta.toFixed(1)}
                    </div>
                  )
                })}
              </Fragment>
            ))}
          </div>
          <p className="mt-3 text-xs leading-relaxed text-ink-400">
            Training on synthetic CLEVR alone is the only set that improves everything —
            including the largest out-of-distribution gain (+5.3 on TextVQA).
          </p>
        </Section>

        <Section
          title="Reward design — the price of keeping reasoning"
          note="Format-reward weight vs chain-of-thought retention (share of GQA completions with reasoning)."
        >
          <div className="space-y-3">
            {data.ablation?.map((arm) => (
              <div key={arm.weight}>
                <div className="mb-1 flex justify-between font-mono text-[11px] text-ink-400">
                  <span>weight {arm.weight.toFixed(1)} · {arm.steps} steps</span>
                  <span className="text-ink-100">{arm.retention}% retained · VQAv2 {runs[arm.run_id]?.vqav2?.toFixed(1)}</span>
                </div>
                <div className="h-4 w-full rounded-sm bg-ink-800">
                  <div
                    className="h-4 rounded-sm bg-ember-600"
                    style={{ width: `${arm.retention}%` }}
                    title={`${arm.retention}% of completions keep chain-of-thought`}
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="mt-4 text-xs leading-relaxed text-ink-400">
            With no format reward, GRPO deletes reasoning (17% retention) because it lowers
            exact-match reward. A 0.2 weight fully preserves it — and the judge study shows
            the apparent EM cost is a metric artifact (81.8 vs 81.1 judge-corrected).
          </p>
        </Section>
      </div>

      {data.judge_study && (
        <Section
          title="LLM-judge study — how much does exact-match undercount?"
          note={`Claude Haiku on sampled EM-misses + hit controls. Total spend: $${data.judge_study.judge_spend_usd.toFixed(2)}.`}
        >
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-ink-700 text-left font-mono text-xs text-ink-400">
                <th className="py-2 pr-4 font-normal">run</th>
                <th className="px-3 py-2 text-right font-normal">EM</th>
                <th className="px-3 py-2 text-right font-normal">rescue rate</th>
                <th className="px-3 py-2 text-right font-normal">judge-corrected</th>
                <th className="px-3 py-2 text-right font-normal">agreement on hits</th>
              </tr>
            </thead>
            <tbody>
              {data.judge_study.results.map((r) => (
                <tr key={r.run_id} className="border-b border-ink-800 last:border-0">
                  <td className="py-2 pr-4 font-mono text-xs text-ink-300">{r.run_id}</td>
                  <td className="px-3 py-2 text-right font-mono text-ink-300">{(r.em_accuracy * 100).toFixed(1)}</td>
                  <td className="px-3 py-2 text-right font-mono text-ink-300">{(r.judge_rescue_rate * 100).toFixed(1)}%</td>
                  <td className="px-3 py-2 text-right font-mono text-ember-400">{(r.judge_corrected_accuracy * 100).toFixed(1)}</td>
                  <td className="px-3 py-2 text-right font-mono text-ink-300">{(r.control_agreement_on_em_hits * 100).toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
      )}
    </div>
  )
}
