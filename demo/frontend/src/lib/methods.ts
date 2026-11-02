export type Method = 'base' | 'sft' | 'grpo' | 'sftgrpo' | 'opd'

// The same four method colors as the report page; OPD gets a fifth, quieter hue.
const METHOD_COLORS: Record<Method, string> = {
  base: '#3d424c',
  sft: '#c4622d',
  grpo: '#2c64a8',
  sftgrpo: '#2f8a6b',
  opd: '#7a5aa6',
}

/** Which training method a model name or key describes. */
export function methodOf(name: string): Method {
  const text = name.toLowerCase()
  if (text.includes('opd')) return 'opd'
  const sft = text.includes('sft')
  const rl = text.includes('grpo')
  if (sft && rl) return 'sftgrpo'
  if (rl) return 'grpo'
  if (sft) return 'sft'
  return 'base'
}

export function methodColor(method: Method): string {
  return METHOD_COLORS[method]
}

const ANSWER_RE = /<answer>([\s\S]*?)<\/answer>/g

/** Split a response into its reasoning and the last tagged answer, if any. */
export function splitAnswer(text: string): { reasoning: string; answer: string | null } {
  const matches = [...text.matchAll(ANSWER_RE)]
  if (matches.length === 0) return { reasoning: text, answer: null }
  const last = matches[matches.length - 1]
  return { reasoning: text.slice(0, last.index).trim(), answer: last[1].trim() }
}

/** What to show for a response: an untagged one-line reply, once complete, is itself the answer. */
export function responseParts(text: string, finished = true): { reasoning: string; answer: string | null } {
  const parts = splitAnswer(text)
  const bare = parts.answer === null && !parts.reasoning.trim().includes('\n')
    && parts.reasoning.trim().length <= 60
  return finished && bare ? { reasoning: '', answer: parts.reasoning.trim() } : parts
}
