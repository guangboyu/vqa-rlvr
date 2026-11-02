import { Fragment, useEffect, useState } from 'react'
import { columnLabel, fetchResults, historyScores, type Report } from '../lib/api'
import FreshResults from './FreshResults'
import { ModelName, PageIntro, Section, Segmented } from './ui'

// July runs are named by their historical run ids; the dataset columns come from the report.
const BASELINE = 'zero_shot_2b_reasoning'

const MAIN_ROWS: [string, string][] = [
  ['Instruct 2B', 'zero_shot_2b_reasoning'],
  ['SFT 2B', 'sft_2b_reasoning'],
  ['GRPO 2B', 'grpo_2b_main_base_reasoning'],
  ['SFT → GRPO 2B', 'grpo_2b_main_sft_reasoning'],
  ['Instruct 8B', 'zero_shot_8b_reasoning'],
  ['SFT 8B', 'sft_8b_reasoning'],
]

const TRANSFER_ROWS: [string, string][] = [
  ['CLEVR only', 'grpo_2b_clevr_base_reasoning'],
  ['VQAv2 only', 'grpo_2b_vqav2only_reasoning'],
  ['GQA only', 'grpo_2b_gqaonly_reasoning'],
  ['VQAv2 + GQA', 'grpo_2b_main_base_reasoning'],
]

/** Diverging fill on the light page: green for a gain, orange for a loss, paper at zero. */
function deltaFill(delta: number, maxAbs: number): string {
  const t = Math.min(Math.abs(delta) / maxAbs, 1)
  const from = [0xf6, 0xf6, 0xf4]
  const to = delta >= 0 ? [0x2f, 0x8a, 0x6b] : [0xc4, 0x62, 0x2d]
  const mix = from.map((f, i) => Math.round(f + (to[i] - f) * t * 0.85))
  return `rgb(${mix[0]},${mix[1]},${mix[2]})`
}

export default function Dashboard() {
  const [data, setData] = useState<Report | null>(null)
  const [variant, setVariant] = useState<'official' | 'normalized'>('official')
  const [history, setHistory] = useState(false)

  useEffect(() => {
    fetchResults().then(setData)
  }, [])

  if (!data) return <p className="text-sm text-ink-3">Loading results…</p>
  const corrected = data.cells.length > 0
  if (corrected && !history) return <FreshResults report={data} onHistory={() => setHistory(true)} />

  const datasets = data.datasets.map(d => d.name)
  const labels = Object.fromEntries(data.datasets.map(d => [d.name, columnLabel(d)]))
  const questions = data.datasets.map(d => {
    const n = Math.max(0, ...data.history.filter(h => h.dataset === d.name).map(h => h.metrics.n))
    return `${n.toLocaleString()} ${d.display_name}`
  }).join(', ')
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
    <div className="space-y-14">
      {corrected && <button onClick={() => setHistory(false)}
        className="text-sm text-ink-3 hover:text-ink">← Current results</button>}
      <PageIntro title="The July 2026 experiments">
        Kept for reference. Several of these runs trained with a VQAv2 reward that paid nothing on
        single-reference questions, and their training pools shared 47 images with the VQAv2 test
        subset. Rescoring changes the evaluation, not the training; the repository's history page
        explains each limitation.
      </PageIntro>
      <Segmented label="Scorer" value={officialAvailable ? variant : 'normalized'} onChange={setVariant}
        options={[{ value: 'official', label: "Dataset authors'" }, { value: 'normalized', label: 'Normalized' }]} />

      <Section title="Accuracy under the reasoning prompt (%)" note={`Greedy decoding on ${questions} questions.`}>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead><tr>
              <th>Model</th>
              {datasets.map((d) => <th key={d}>{labels[d]}</th>)}
            </tr></thead>
            <tbody>{MAIN_ROWS.map(([label, id]) => <tr key={id}>
              <td><ModelName name={label} /></td>
              {datasets.map((d) => {
                const v = runs[id]?.[d]
                const best = Math.max(...MAIN_ROWS.map(([, r]) => runs[r]?.[d] ?? 0))
                return <td key={d} className={v === best ? 'font-semibold' : 'text-ink-2'}>{v?.toFixed(1) ?? '—'}</td>
              })}
            </tr>)}</tbody>
          </table>
        </div>
      </Section>

      <div className="grid gap-14 lg:grid-cols-2">
        <Section title="Where does GRPO transfer?"
          note="Points gained over Instruct under the reasoning prompt; rows name the GRPO training data. Green is a gain, orange a loss.">
          <div className="grid gap-[3px]" style={{ gridTemplateColumns: `auto repeat(${datasets.length}, 1fr)` }}>
            <div />
            {datasets.map((d) => (
              <div key={d} className="px-1 pb-1 text-center text-[11px] text-ink-3">{labels[d]}</div>
            ))}
            {TRANSFER_ROWS.map(([label], i) => (
              <Fragment key={label}>
                <div className="flex items-center pr-3 text-[13px] text-ink-2">{label}</div>
                {datasets.map((d, j) => {
                  const delta = deltas[i][j]
                  const strong = delta != null && Math.abs(delta) / maxAbs > 0.55
                  return (
                    <div
                      key={`${label}-${d}`}
                      title={delta == null ? 'Not evaluated' : `trained on ${label}, tested on ${labels[d]}: ${delta.toFixed(1)}`}
                      className={`flex h-11 items-center justify-center rounded-[3px] text-sm tabular-nums ${strong ? 'text-paper' : 'text-ink'}`}
                      style={{ background: delta == null ? '#fbfbfa' : deltaFill(delta, maxAbs) }}
                    >
                      {delta == null ? '—' : `${delta >= 0 ? '+' : '−'}${Math.abs(delta).toFixed(1)}`}
                    </div>
                  )
                })}
              </Fragment>
            ))}
          </div>
          <p className="mt-3 text-[13px] leading-relaxed text-ink-3">
            A dash means no saved evaluation. The VQAv2-only run received no correctness signal.
          </p>
        </Section>

        <Section title="Reward design and response length"
          note="Share of GQA responses longer than 100 characters under each format-reward weight.">
          <div className="space-y-4">
            {data.retention_ablation?.map((arm) => (
              <div key={arm.weight}>
                <div className="mb-1.5 flex justify-between text-[13px]">
                  <span className="text-ink-2">Format weight {arm.weight.toFixed(1)} · {arm.steps} steps</span>
                  <span className="tabular-nums">{arm.retention}% long · VQAv2 {runs[arm.run_id]?.vqav2?.toFixed(1)}</span>
                </div>
                <div className="h-2 w-full rounded-full bg-wash">
                  <div className="h-2 rounded-full bg-ink-2" style={{ width: `${arm.retention}%` }}
                    title={`${arm.retention}% of responses exceed 100 characters`} />
                </div>
              </div>
            ))}
          </div>
          <p className="mt-4 text-[13px] leading-relaxed text-ink-3">
            Length follows the reward. It does not measure reasoning quality, and these runs also
            differ in training length.
          </p>
        </Section>
      </div>

      {data.judge_study && (
        <Section title="How much does exact match undercount?"
          note={`An exploratory July estimate; the follow-up did not reproduce its claimed ranking reversal. Judge spend: $${data.judge_study.judge_spend_usd.toFixed(2)}.`}>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead><tr>
                <th>Run</th><th>Exact match</th><th>Rescued by the judge</th><th>Judge-corrected</th><th>Agreement on hits</th>
              </tr></thead>
              <tbody>{data.judge_study.results.map((r) => <tr key={r.run_id}>
                <td className="font-mono text-xs text-ink-2">{r.run_id}</td>
                <td>{(r.em_accuracy * 100).toFixed(1)}</td>
                <td>{(r.judge_rescue_rate * 100).toFixed(1)}%</td>
                <td className="font-semibold">{(r.judge_corrected_accuracy * 100).toFixed(1)}</td>
                <td>{(r.control_agreement_on_em_hits * 100).toFixed(0)}%</td>
              </tr>)}</tbody>
            </table>
          </div>
        </Section>
      )}
    </div>
  )
}
