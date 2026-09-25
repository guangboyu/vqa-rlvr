export interface ModelSpec {
  key: string
  label: string
  description: string
  loaded: boolean
}

/** The report builder's export (results/report.json): the same records as every table. */
export interface Report {
  arms: { label: string; name: string }[]
  max_pixels: number
  datasets: { name: string; display_name: string; ladder: string }[]
  cells: Cell[]
  history: { run: string; dataset: string; prompt: string; metrics: CellMetrics }[]
  exposure: Record<string, {
    completed_updates: number
    updates_without_reward_variation: number
    overall: { unique_questions: number; variable_reward_fraction: number }
  }>
  same_prompt_order: boolean | null
  retention_ablation: { weight: number; steps: number; retention: number; run_id: string }[]
  judge_study: {
    results: {
      run_id: string
      em_accuracy: number
      judge_rescue_rate: number
      judge_corrected_accuracy: number
      control_agreement_on_em_hits: number
    }[]
    judge_spend_usd: number
  } | null
}

export interface Cell {
  arm: string
  dataset: string
  role: string
  tier: string
  prompt: string
  image_mode: string
  step: number | null
  temperature: number
  samples: number
  max_pixels: number
  legacy: boolean
  metrics: CellMetrics
}

export interface CellMetrics {
  n: number
  official: { overall: number } | null
  normalized: { overall: number }
  mean_output_tokens: number | null
  truncated_fraction: number | null
  answer_tag_fraction: number | null
  official_on_lowercased_answers?: number
  paired_with_original?: {
    n: number
    original: { official?: number; normalized: number }
    normalized_answer_changed_fraction: number
  }
}

/** A greedy test-role cell of one arm, dataset, prompt, image mode, and tier. */
export function testCell(report: Report, arm: string, dataset: string, prompt: string,
                         imageMode = 'original', tier = 'full'): CellMetrics | undefined {
  return report.cells.find(c => c.arm === arm && c.dataset === dataset && c.prompt === prompt
    && c.image_mode === imageMode && c.tier === tier && c.role === 'test' && c.step === null
    && c.samples === 1 && c.max_pixels === report.max_pixels)?.metrics
}

/** A dataset's column heading: OOD datasets say so. */
export function columnLabel(dataset: Report['datasets'][number]): string {
  return dataset.ladder === 'in-domain' ? dataset.display_name : `${dataset.display_name} (OOD)`
}

/** July scores in percent, by run and dataset, under one scorer variant. */
export function historyScores(report: Report, variant: 'official' | 'normalized') {
  const scores: Record<string, Record<string, number>> = {}
  for (const row of report.history) {
    const value = row.metrics[variant]?.overall
    if (value != null) (scores[row.run] ??= {})[row.dataset] = value * 100
  }
  return scores
}

export async function fetchModels(): Promise<ModelSpec[]> {
  const res = await fetch('/api/models')
  return res.json()
}

export async function fetchResults(): Promise<Report> {
  const res = await fetch('/api/results')
  return res.json()
}

/** POST an arena question; invokes callbacks as SSE tokens arrive per model. */
export async function ask(
  image: File,
  question: string,
  template: 'short' | 'reasoning',
  onToken: (model: string, token: string) => void,
  onDone: (model: string) => void,
  models: string[] = [],
  imageCondition: 'original' | 'black' = 'original',
): Promise<void> {
  const form = new FormData()
  form.append('image', image)
  form.append('question', question)
  form.append('template', template)
  form.append('models', models.join(','))
  form.append('image_condition', imageCondition)

  const res = await fetch('/api/ask', { method: 'POST', body: form })
  if (!res.ok || !res.body) throw new Error(`ask failed: ${res.status}`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      const data = line.replace(/^data: /, '').trim()
      if (!data || data === '[DONE]') continue
      const event = JSON.parse(data)
      if (event.error) throw new Error(event.error)
      if (event.done) onDone(event.model)
      else onToken(event.model, event.token)
    }
  }
}
