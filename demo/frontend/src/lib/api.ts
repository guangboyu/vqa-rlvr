export interface ModelSpec {
  key: string
  label: string
  description: string
  loaded: boolean
}

export interface ResultsPayload {
  runs: Record<string, Record<string, number>>
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
): Promise<void> {
  const form = new FormData()
  form.append('image', image)
  form.append('question', question)
  form.append('template', template)

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
      if (event.done) onDone(event.model)
      else onToken(event.model, event.token)
    }
  }
}
