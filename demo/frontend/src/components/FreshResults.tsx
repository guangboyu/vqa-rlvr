import { useState } from 'react'
import type { Comparison } from '../lib/api'

const DATASETS = ['vqav2', 'gqa', 'clevr', 'textvqa'] as const
const LABELS: Record<string, string> = {
  vqav2: 'VQAv2', gqa: 'GQA', clevr: 'CLEVR counting', textvqa: 'TextVQA',
  base2: '2B Instruct', sft2: '2B + SFT', 'grpo2-base': '2B + GRPO',
  'grpo2-sft': '2B + SFT + GRPO', base8: '8B Instruct', 'grpo8-base': '8B + GRPO',
  'opd2-self': '2B + gold-conditioned OPD',
}
const ORDER = ['base2', 'sft2', 'grpo2-base', 'grpo2-sft', 'base8', 'grpo8-base', 'opd2-self']
const percent = (value: number | undefined) => value == null ? '—' : (value * 100).toFixed(2)
const selectStyle = 'rounded border border-ink-700 bg-ink-900 p-2'
const cellStyle = 'px-3 py-2 text-right font-mono'

export default function FreshResults({ comparison }: { comparison: Comparison }) {
  const [template, setTemplate] = useState('short')
  const [metric, setMetric] = useState<'overall' | 'normalized'>('overall')
  const [dataset, setDataset] = useState<string>('vqav2')
  const rows = Object.entries(comparison.runs)
    .filter(([, run]) => run.template === template)
    .map(([id, run]) => ({ id, arm: id.replace(`-${template}`, ''), ...run }))
    .sort((a, b) => ORDER.indexOf(a.arm) - ORDER.indexOf(b.arm))

  return <div className="space-y-6">
    <div>
      <h2 className="font-display text-xl text-ink-50">What changes after post-training?</h2>
      <p className="mt-2 text-sm leading-relaxed text-ink-300">
        Corrected pipeline, one training seed, fixed held-out questions. Both 2B GRPO starting
        points use the same training pool and 300 updates. SFT adds an epoch of answer
        supervision, so total training exposure and compute differ.
      </p>
    </div>
    <div className="flex flex-wrap items-center gap-4 text-sm">
      <label>Prompt <select value={template} onChange={e => setTemplate(e.target.value)}
        className={selectStyle}>
        <option value="short">Short answer</option><option value="reasoning">Reasoning</option>
      </select></label>
      <label>Scoring <select value={metric}
        onChange={e => setMetric(e.target.value as 'overall' | 'normalized')} className={selectStyle}>
        <option value="overall">Dataset-author accuracy</option>
        <option value="normalized">Normalized answer matching</option>
      </select></label>
    </div>
    <section className="overflow-x-auto rounded-lg border border-ink-700 bg-ink-900 p-5">
      <h3 className="font-display text-lg text-ink-50">Answer accuracy (%)</h3>
      <p className="mt-1 text-xs text-ink-400">
        5,000 VQAv2 · 12,578 GQA · 3,000 CLEVR counting · 2,000 TextVQA.
        Scoring choices reuse the same responses.
      </p>
      <table className="mt-3 w-full text-sm">
        <thead><tr className="border-b border-ink-700 text-ink-400">
          <th className="py-2 text-left font-normal">Model</th>
          {DATASETS.map(d => <th key={d} className={`${cellStyle} font-normal`}>{LABELS[d]}</th>)}
        </tr></thead>
        <tbody>{rows.map(row => <tr key={row.id} className="border-b border-ink-800 last:border-0">
          <td className="py-2 text-ink-100">{LABELS[row.arm] ?? row.arm}</td>
          {DATASETS.map(d => <td key={d} className={cellStyle}>
            {percent(row.datasets[d]?.[metric])}
          </td>)}
        </tr>)}</tbody>
      </table>
    </section>
    {comparison.training && <section className="overflow-x-auto rounded-lg border border-ink-700 bg-ink-900 p-5">
      <h3 className="font-display text-lg text-ink-50">How much GRPO feedback is available?</h3>
      <p className="mt-1 text-xs leading-relaxed text-ink-400">
        Eight responses form a prompt group. Identical rewards give zero group advantages.
        These counts describe sampled training responses, not held-out accuracy.
        {comparison.training.same_prompt_order && ' The complete prompt order matches across these runs.'}
      </p>
      <table className="mt-3 w-full text-sm">
        <thead><tr className="border-b border-ink-700 text-ink-400">
          <th className="py-2 text-left font-normal">Model</th>
          {['Distinct questions', 'Groups with varied rewards (%)', 'Updates without reward variation']
            .map(label => <th key={label} className={`${cellStyle} font-normal`}>{label}</th>)}
        </tr></thead>
        <tbody>{Object.entries(comparison.training.runs).map(([path, run]) => {
          const arm = path.split('/').pop() ?? path
          return <tr key={path} className="border-b border-ink-800 last:border-0">
            <td className="py-2">{LABELS[arm] ?? arm}</td>
            <td className={cellStyle}>{run.overall.unique_questions.toLocaleString()}</td>
            <td className={cellStyle}>{percent(run.overall.variable_reward_fraction)}</td>
            <td className={cellStyle}>{run.updates_without_reward_variation} / {run.completed_updates}</td>
          </tr>
        })}</tbody>
      </table>
    </section>}
    <label className="flex items-center gap-3 text-sm">Inspect a dataset
      <select value={dataset} onChange={e => setDataset(e.target.value)} className={selectStyle}>
        {DATASETS.map(d => <option key={d} value={d}>{LABELS[d]}</option>)}
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
          return <tr key={row.id} className="border-b border-ink-800 last:border-0">
            <td className="py-2">{LABELS[row.arm] ?? row.arm}</td>
            <td className={cellStyle}>{d?.mean_output_tokens.toFixed(1) ?? '—'}</td>
            <td className={cellStyle}>{percent(d?.truncated_fraction)}</td>
            <td className={cellStyle}>{percent(d?.answer_tag_fraction)}</td>
            <td className={cellStyle}>{percent(d?.lowercase_official)}</td>
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
        <tbody>{rows.map(row => {
          const d = comparison.image_interventions[row.id]?.[dataset]
          return <tr key={row.id} className="border-b border-ink-800 last:border-0">
            <td className="py-2">{LABELS[row.arm] ?? row.arm}</td>
            <td className={cellStyle}>{percent(d?.black?.original_score)}</td>
            <td className={cellStyle}>{percent(d?.black?.changed_image_score)}</td>
            <td className={cellStyle}>{percent(d?.swapped?.changed_image_score)}</td>
          </tr>
        })}</tbody>
      </table>
    </section>
  </div>
}
