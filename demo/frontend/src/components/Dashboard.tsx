import { Fragment, useEffect, useState } from 'react'
import { columnLabel, fetchResults, historyScores, type Report } from '../lib/api'
import FreshResults from './FreshResults'

// July runs are named by their historical run ids; the dataset columns come from the report.
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
  const [data, setData] = useState<Report | null>(null)
  const [variant, setVariant] = useState<'official' | 'normalized'>('official')
  const [history, setHistory] = useState(false)

  useEffect(() => {
    fetchResults().then(setData)
  }, [])

  if (!data) return <p className="text-sm text-ink-400">Loading results…</p>
  const datasets = data.datasets.map(d => d.name)
  const labels = Object.fromEntries(data.datasets.map(d => [d.name, columnLabel(d)]))
  const questions = data.datasets.map(d => {
    const n = Math.max(0, ...data.history.filter(h => h.dataset === d.name).map(h => h.metrics.n))
    return `${n.toLocaleString()} ${d.display_name}`
  }).join(', ')
  const corrected = data.cells.length > 0
  if (corrected && !history) return (
    <div className="space-y-5">
      <FreshResults report={data} />
      <button onClick={() => setHistory(true)} className="text-sm text-ink-300 underline">
        Explore the historical experiments and their limitations
      </button>
    </div>
  )
  const official = historyScores(data, 'official')
  const officialAvailable = Object.keys(official).length > 0
  const runs = variant === 'official' && officialAvailable ? official
    : historyScores(data, 'normalized')
  const baseline = runs[BASELINE] ?? {}
  const deltas = TRANSFER_ROWS.map(([, id]) =>
    datasets.map((d) => runs[id]?.[d] != null && baseline[d] != null ? runs[id][d] - baseline[d] : null),
  )
  const maxAbs = Math.max(1, ...deltas.flat().map((value) => Math.abs(value ?? 0)))

  return (
    <div className="space-y-6">
      {corrected && <button onClick={() => setHistory(false)}
        className="text-sm text-ink-300 underline">Back to the corrected comparison</button>}
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <label htmlFor="metric">Scoring protocol</label>
        <select id="metric" value={officialAvailable ? variant : 'normalized'}
          onChange={(event) => setVariant(event.target.value as 'official' | 'normalized')}
          className="rounded border border-ink-700 bg-ink-900 p-2">
          <option value="official" disabled={!officialAvailable}>Dataset-author accuracy</option>
          <option value="normalized">Historical normalized accuracy</option>
        </select>
        <p className="text-xs text-ink-400">Same saved predictions; different scoring rules.</p>
      </div>
      <p className="text-sm text-ink-300">
        Historical experiments: several training runs used a defective VQAv2 reward,
        and the original data mix contained shared images. Rescoring updates evaluation,
        not training. See the repository validation note before interpreting comparisons.
      </p>
      <Section
        title="Accuracy under the reasoning prompt"
        note={`Greedy decoding: ${questions} questions.`}
      >
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-ink-700 text-left font-mono text-xs text-ink-400">
              <th className="py-2 pr-4 font-normal">model</th>
              {datasets.map((d) => (
                <th key={d} className="px-3 py-2 text-right font-normal">{labels[d]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MAIN_ROWS.map(([label, id]) => (
              <tr key={id} className="border-b border-ink-800 last:border-0">
                <td className="py-2 pr-4 text-ink-100">{label}</td>
                {datasets.map((d) => {
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
          <div className="grid" style={{ gridTemplateColumns: `auto repeat(${datasets.length}, 1fr)`, gap: 2 }}>
            <div />
            {datasets.map((d) => (
              <div key={d} className="px-1 pb-1 text-center font-mono text-[10px] text-ink-400">
                {labels[d]}
              </div>
            ))}
            {TRANSFER_ROWS.map(([label], i) => (
              <Fragment key={label}>
                <div className="flex items-center pr-2 font-mono text-[11px] text-ink-300">
                  {label}
                </div>
                {datasets.map((d, j) => {
                  const delta = deltas[i][j]
                  return (
                    <div
                      key={`${label}-${d}`}
                      title={delta == null ? 'Not evaluated' : `train ${label} → eval ${labels[d]}: ${delta.toFixed(1)}`}
                      className="flex h-12 items-center justify-center rounded font-mono text-sm text-ink-50"
                      style={{ background: delta == null ? '#202631' : deltaColor(delta, maxAbs) }}
                    >
                      {delta == null ? '—' : `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}`}
                    </div>
                  )
                })}
              </Fragment>
            ))}
          </div>
          <p className="mt-3 text-xs leading-relaxed text-ink-400">
            A blank cell means no saved evaluation. These runs compare different training
            sources; the VQAv2-only legacy arm received no correctness signal.
          </p>
        </Section>

        <Section
          title="Reward design and response length"
          note="Historical retention proxy: share of GQA completions longer than 100 characters."
        >
          <div className="space-y-3">
            {data.retention_ablation?.map((arm) => (
              <div key={arm.weight}>
                <div className="mb-1 flex justify-between font-mono text-[11px] text-ink-400">
                  <span>weight {arm.weight.toFixed(1)} · {arm.steps} steps</span>
                  <span className="text-ink-100">{arm.retention}% retained · VQAv2 {runs[arm.run_id]?.vqav2?.toFixed(1)}</span>
                </div>
                <div className="h-4 w-full rounded-sm bg-ink-800">
                  <div
                    className="h-4 rounded-sm bg-ember-600"
                    style={{ width: `${arm.retention}%` }}
                    title={`${arm.retention}% of completions exceed 100 characters`}
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="mt-4 text-xs leading-relaxed text-ink-400">
            Response length changes with the reward. Length alone does not establish
            reasoning quality, and these arms also differ in training duration.
          </p>
        </Section>
      </div>

      {data.judge_study && (
        <Section
          title="LLM-judge study — how much does exact-match undercount?"
          note={`Historical exploratory estimate; the follow-up did not reproduce the claimed ranking reversal. Judge spend: $${data.judge_study.judge_spend_usd.toFixed(2)}.`}
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
