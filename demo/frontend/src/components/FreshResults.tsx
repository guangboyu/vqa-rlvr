import { useState } from 'react'
import { columnLabel, testCell, type Report } from '../lib/api'

const percent = (value: number | null | undefined) =>
  value == null ? '—' : (value * 100).toFixed(2)
// The recorded image interventions cover the first 128 test questions of each dataset.
const INTERVENTION_TIER = 'first-128'
const selectStyle = 'rounded border border-ink-700 bg-ink-900 p-2'
const cellStyle = 'px-3 py-2 text-right font-mono'

export default function FreshResults({ report }: { report: Report }) {
  const [template, setTemplate] = useState('short')
  const [metric, setMetric] = useState<'official' | 'normalized'>('official')
  const [dataset, setDataset] = useState<string>(report.datasets[0]?.name ?? '')
  const armName = (label: string) => report.arms.find(a => a.label === label)?.name ?? label
  const rows = report.arms.map(arm => ({
    arm: arm.label,
    datasets: Object.fromEntries(report.datasets.map(d =>
      [d.name, testCell(report, arm.label, d.name, template)])),
  }))

  return <div className="space-y-6">
    <div>
      <h2 className="font-display text-xl text-ink-50">What changes after post-training?</h2>
      <p className="mt-2 text-sm leading-relaxed text-ink-300">
        Final rerun, one training seed, fixed held-out questions. Every RL arm uses the same
        training pool and 1,000 updates under one frozen recipe. SFT adds an epoch of answer
        supervision, so total training exposure and compute differ.
      </p>
    </div>
    <div className="flex flex-wrap items-center gap-4 text-sm">
      <label>Prompt <select value={template} onChange={e => setTemplate(e.target.value)}
        className={selectStyle}>
        <option value="short">Short answer</option><option value="reasoning">Reasoning</option>
      </select></label>
      <label>Scoring <select value={metric}
        onChange={e => setMetric(e.target.value as 'official' | 'normalized')} className={selectStyle}>
        <option value="official">Dataset-author accuracy</option>
        <option value="normalized">Normalized answer matching</option>
      </select></label>
    </div>
    <section className="overflow-x-auto rounded-lg border border-ink-700 bg-ink-900 p-5">
      <h3 className="font-display text-lg text-ink-50">Answer accuracy (%)</h3>
      <p className="mt-1 text-xs text-ink-400">
        {report.datasets.map(d => {
          const n = rows.map(row => row.datasets[d.name]?.n).find(value => value != null)
          return `${n?.toLocaleString() ?? '—'} ${d.display_name}`
        }).join(' · ')}. Scoring choices reuse the same responses.
      </p>
      <table className="mt-3 w-full text-sm">
        <thead><tr className="border-b border-ink-700 text-ink-400">
          <th className="py-2 text-left font-normal">Model</th>
          {report.datasets.map(d => <th key={d.name} className={`${cellStyle} font-normal`}>
            {columnLabel(d)}</th>)}
        </tr></thead>
        <tbody>{rows.map(row => <tr key={row.arm} className="border-b border-ink-800 last:border-0">
          <td className="py-2 text-ink-100">{armName(row.arm)}</td>
          {report.datasets.map(d => <td key={d.name} className={cellStyle}>
            {percent(row.datasets[d.name]?.[metric]?.overall)}
          </td>)}
        </tr>)}</tbody>
      </table>
    </section>
    {Object.keys(report.exposure).length > 0 && <section className="overflow-x-auto rounded-lg border border-ink-700 bg-ink-900 p-5">
      <h3 className="font-display text-lg text-ink-50">How much GRPO feedback is available?</h3>
      <p className="mt-1 text-xs leading-relaxed text-ink-400">
        Eight responses form a prompt group. Identical rewards give zero group advantages.
        These counts describe sampled training responses, not held-out accuracy.
        {report.same_prompt_order && ' The complete prompt order matches across these runs.'}
      </p>
      <table className="mt-3 w-full text-sm">
        <thead><tr className="border-b border-ink-700 text-ink-400">
          <th className="py-2 text-left font-normal">Model</th>
          {['Distinct questions', 'Groups with varied rewards (%)', 'Updates without reward variation']
            .map(label => <th key={label} className={`${cellStyle} font-normal`}>{label}</th>)}
        </tr></thead>
        <tbody>{Object.entries(report.exposure).map(([path, run]) => {
          const arm = path.split('/').pop() ?? path
          return <tr key={path} className="border-b border-ink-800 last:border-0">
            <td className="py-2">{armName(arm)}</td>
            <td className={cellStyle}>{run.overall.unique_questions.toLocaleString()}</td>
            <td className={cellStyle}>{percent(run.overall.variable_reward_fraction)}</td>
            <td className={cellStyle}>{run.updates_without_reward_variation} / {run.completed_updates}</td>
          </tr>
        })}</tbody>
      </table>
    </section>}
    <label className="flex items-center gap-3 text-sm">Inspect a dataset
      <select value={dataset} onChange={e => setDataset(e.target.value)} className={selectStyle}>
        {report.datasets.map(d => <option key={d.name} value={d.name}>{d.display_name}</option>)}
      </select>
    </label>
    <section className="overflow-x-auto rounded-lg border border-ink-700 bg-ink-900 p-5">
      <h3 className="font-display text-lg text-ink-50">Response behavior</h3>
      <p className="mt-1 text-xs text-ink-400">
        Longer output does not establish better reasoning. Lowercasing before official
        scoring shows how much answer casing affects the result.
      </p>
      <table className="mt-3 w-full text-sm">
        <thead><tr className="border-b border-ink-700 text-ink-400">
          {['Model', 'Mean tokens', 'Truncated (%)', 'Answer tags (%)', 'Lowercase score (%)']
            .map((label, i) => <th key={label} className={i ? `${cellStyle} font-normal` : 'text-left font-normal'}>
              {label}</th>)}
        </tr></thead>
        <tbody>{rows.map(row => {
          const d = row.datasets[dataset]
          return <tr key={row.arm} className="border-b border-ink-800 last:border-0">
            <td className="py-2">{armName(row.arm)}</td>
            <td className={cellStyle}>{d?.mean_output_tokens?.toFixed(1) ?? '—'}</td>
            <td className={cellStyle}>{percent(d?.truncated_fraction)}</td>
            <td className={cellStyle}>{percent(d?.answer_tag_fraction)}</td>
            <td className={cellStyle}>{percent(d?.official_on_lowercased_answers)}</td>
          </tr>
        })}</tbody>
      </table>
    </section>
    <section className="overflow-x-auto rounded-lg border border-ink-700 bg-ink-900 p-5">
      <h3 className="font-display text-lg text-ink-50">Does changing the image change the answer?</h3>
      <p className="mt-1 text-xs leading-relaxed text-ink-400">
        First 128 questions per dataset. All scores below use the original image's
        references and the dataset-author metric. This measures image sensitivity;
        it does not establish accuracy on the changed images or isolate the image encoder.
      </p>
      <table className="mt-3 w-full text-sm">
        <thead><tr className="border-b border-ink-700 text-ink-400">
          <th className="py-2 text-left font-normal">Model</th>
          {['Original (%)', 'Black (%)', 'Swapped (%)'].map(label =>
            <th key={label} className={`${cellStyle} font-normal`}>{label}</th>)}
        </tr></thead>
        <tbody>{rows.filter(row => testCell(report, row.arm, dataset, template, 'black', INTERVENTION_TIER))
          .map(row => {
            const black = testCell(report, row.arm, dataset, template, 'black', INTERVENTION_TIER)
            const swapped = testCell(report, row.arm, dataset, template, 'swapped', INTERVENTION_TIER)
            return <tr key={row.arm} className="border-b border-ink-800 last:border-0">
              <td className="py-2">{armName(row.arm)}</td>
              <td className={cellStyle}>{percent(black?.paired_with_original?.original.official)}</td>
              <td className={cellStyle}>{percent(black?.official?.overall)}</td>
              <td className={cellStyle}>{percent(swapped?.official?.overall)}</td>
            </tr>
          })}</tbody>
      </table>
    </section>
  </div>
}
