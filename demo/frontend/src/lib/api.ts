export interface ModelSpec {
  key: string
  label: string
  description: string
  loaded: boolean
}

export interface ResultsPayload {
  comparison: Comparison | null
  runs: Record<string, Record<string, number>>
  official_runs: Record<string, Record<string, number>>
  ablation: { weight: number; steps: number; retention: number; run_id: string }[]
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

export interface Comparison {
  training?: {
    same_prompt_order: boolean | null
    runs: Record<string, {
      completed_updates: number
      updates_without_reward_variation: number
      overall: { unique_questions: number; variable_reward_fraction: number }
    }>
  }
  runs: Record<string, {
    template: string
    datasets: Record<string, {
      overall: number
      normalized: number
      lowercase_official: number
      n: number
      mean_output_tokens: number
      truncated_fraction: number
      answer_tag_fraction: number
    }>
  }>
  image_interventions: Record<string, Record<string, Record<string, {
    n: number
    original_score: number
    changed_image_score: number
    normalized_answer_changed_fraction: number
  }>>>
}

export async function fetchModels(): Promise<ModelSpec[]> {
  const res = await fetch('/api/models')
  return res.json()
}

export async function fetchResults(): Promise<ResultsPayload> {
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
