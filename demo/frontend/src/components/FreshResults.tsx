import { useState } from 'react'
import { columnLabel, testCell, type CellMetrics, type Report } from '../lib/api'
import { ModelName, PageIntro, Section, Segmented } from './ui'

const percent = (value: number | null | undefined) =>
  value == null ? '—' : (value * 100).toFixed(1)
// The recorded image interventions cover the first 128 test questions of each dataset.
const INTERVENTION_TIER = 'first-128'

type Prompt = 'short' | 'reasoning'
type Metric = 'official' | 'normalized'

interface Row {
  arm: string
  name: string
  size: string
  cells: Record<string, CellMetrics | undefined>
}

/** Arms grouped by model size, in the report's order. */
function groupBySize(rows: Row[]): [string, Row[]][] {
  const groups = new Map<string, Row[]>()
  for (const row of rows) groups.set(row.size, [...(groups.get(row.size) ?? []), row])
  return [...groups]
}

export default function FreshResults({ report, onHistory }: { report: Report; onHistory: () => void }) {
  const [prompt, setPrompt] = useState<Prompt>('short')
  const [metric, setMetric] = useState<Metric>('official')
  const [dataset, setDataset] = useState<string>(report.datasets[0]?.name ?? '')
  const nameOf = (arm: string) => report.arms.find(a => a.label === arm)?.name ?? arm
  const rows: Row[] = report.arms.map(arm => ({
    arm: arm.label,
    name: arm.name,
    size: arm.name.includes('8B') ? '8B' : '2B',
    cells: Object.fromEntries(report.datasets.map(d =>
      [d.name, testCell(report, arm.label, d.name, prompt)])),
  }))
  const best = Object.fromEntries(report.datasets.map(d => [d.name, Math.max(
    ...rows.map(row => row.cells[d.name]?.[metric]?.overall ?? -1))]))
  const sizes = Object.fromEntries(report.datasets.map(d =>
    [d.name, rows.map(row => row.cells[d.name]?.n).find(n => n != null)]))

  return <div className="space-y-14">
    <PageIntro title="Results">
      Photo questions (VQAv2, GQA, CLEVR counting) and TextVQA, which no model trained on. One
      training seed per model, fixed test questions. Every RL model draws the same 4,000 training
      questions over 1,000 updates under one recipe; SFT trains for one epoch on 39,458 answers.
      Chart and medical results are in the{' '}
      <a href="https://guangboyu.github.io/vqa-rlvr/report/" target="_blank" rel="noreferrer"
        className="text-accent underline decoration-accent/30 underline-offset-2 hover:decoration-accent">report</a>.
    </PageIntro>

    <div className="flex flex-wrap gap-x-8 gap-y-3">
      <Segmented label="Prompt" value={prompt} onChange={setPrompt} options={[
        { value: 'short', label: 'Short answer' }, { value: 'reasoning', label: 'Step by step' }]} />
      <Segmented label="Scorer" value={metric} onChange={setMetric} options={[
        { value: 'official', label: "Dataset authors'" }, { value: 'normalized', label: 'Normalized' }]} />
    </div>

    <Section title="Answer accuracy (%)" note={<>
      {report.datasets.map(d => `${sizes[d.name]?.toLocaleString() ?? '—'} ${d.display_name}`).join(' · ')}{' '}
      test questions. Both scorers read the same responses; best in each column in bold.
    </>}>
      <div className="overflow-x-auto">
        <table className="data-table">
          <thead><tr>
            <th>Model</th>
            {report.datasets.map(d => <th key={d.name}>{columnLabel(d)}</th>)}
          </tr></thead>
          <tbody>{groupBySize(rows).map(([size, group]) => [
            <tr key={size} className="group-row"><td colSpan={report.datasets.length + 1}>Qwen3-VL {size}</td></tr>,
            ...group.map(row => <tr key={row.arm}>
              <td><ModelName name={row.name} /></td>
              {report.datasets.map(d => {
                const value = row.cells[d.name]?.[metric]?.overall
                return <td key={d.name} className={value === best[d.name] ? 'font-semibold' : 'text-ink-2'}>
                  {percent(value)}
                </td>
              })}
            </tr>),
          ])}</tbody>
        </table>
      </div>
    </Section>

    {Object.keys(report.exposure).length > 0 && <Section title="How much learning signal GRPO received"
      note={<>Each training question gets eight sampled responses. When all eight earn the same reward, the
        group carries no advantage and teaches nothing.
        {report.same_prompt_order && ' Every run below drew the questions in the same order.'}</>}>
      <div className="overflow-x-auto">
        <table className="data-table">
          <thead><tr>
            <th>Model</th><th>Distinct questions</th><th>Groups with varied rewards (%)</th>
            <th>Updates without any varied group</th>
          </tr></thead>
          <tbody>{Object.entries(report.exposure).map(([path, run]) => {
            const arm = (path.split('/').pop() ?? path).replace(/^vqa-/, '')
            return <tr key={path}>
              <td><ModelName name={nameOf(arm)} /></td>
              <td>{run.overall.unique_questions.toLocaleString()}</td>
              <td>{percent(run.overall.variable_reward_fraction)}</td>
              <td>{run.updates_without_reward_variation} of {run.completed_updates.toLocaleString()}</td>
            </tr>
          })}</tbody>
        </table>
      </div>
    </Section>}

    <div className="space-y-8">
      <Segmented label="Dataset" value={dataset} onChange={setDataset}
        options={report.datasets.map(d => ({ value: d.name, label: d.display_name }))} />

      <Section title="How the models respond"
        note="Longer output is not better reasoning by itself. The lowercase column applies the authors' scorer after lowercasing each answer, which shows how much capitalization costs.">
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead><tr>
              {['Model', 'Mean tokens', 'Truncated (%)', 'With answer tag (%)', 'Lowercased score (%)']
                .map(label => <th key={label}>{label}</th>)}
            </tr></thead>
            <tbody>{rows.map(row => {
              const cell = row.cells[dataset]
              return <tr key={row.arm}>
                <td><ModelName name={row.name} /></td>
                <td>{cell?.mean_output_tokens?.toFixed(0) ?? '—'}</td>
                <td>{percent(cell?.truncated_fraction)}</td>
                <td>{percent(cell?.answer_tag_fraction)}</td>
                <td>{percent(cell?.official_on_lowercased_answers)}</td>
              </tr>
            })}</tbody>
          </table>
        </div>
      </Section>

      <Section title="Does a changed image change the answer?"
        note="Recorded for the 2B Instruct and SFT models on the first 128 test questions of each dataset, scored against the original image's references with the authors' scorer. A drop shows the answer depends on the image; it does not show the answer was right for the right reason.">
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead><tr>
              {['Model', 'Original image (%)', 'Black image (%)', "Another question's image (%)"]
                .map(label => <th key={label}>{label}</th>)}
            </tr></thead>
            <tbody>{rows.filter(row => testCell(report, row.arm, dataset, prompt, 'black', INTERVENTION_TIER))
              .map(row => {
                const black = testCell(report, row.arm, dataset, prompt, 'black', INTERVENTION_TIER)
                const swapped = testCell(report, row.arm, dataset, prompt, 'swapped', INTERVENTION_TIER)
                return <tr key={row.arm}>
                  <td><ModelName name={row.name} /></td>
                  <td>{percent(black?.paired_with_original?.original.official)}</td>
                  <td>{percent(black?.official?.overall)}</td>
                  <td>{percent(swapped?.official?.overall)}</td>
                </tr>
              })}</tbody>
          </table>
        </div>
      </Section>
    </div>

    <button onClick={onHistory} className="text-sm text-ink-3 underline decoration-line underline-offset-4 hover:text-ink">
      The July 2026 experiments and their limitations
    </button>
  </div>
}
