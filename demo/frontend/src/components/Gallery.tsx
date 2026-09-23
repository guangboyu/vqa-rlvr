import { useEffect, useState } from 'react'

interface Example {
  title: string
  lesson: string
  question: string
  qid: string
  dataset?: string
  study?: string
  reference_label?: string
  image: string
  answers: string[]
  responses: { label: string; run_id: string; raw: string; image?: string; official: number | null; normalized: number | null }[]
}

export default function Gallery() {
  const [examples, setExamples] = useState<Example[]>([])
  const [error, setError] = useState('')
  useEffect(() => {
    fetch('/api/gallery').then((response) => {
      if (!response.ok) throw new Error('Could not load examples.')
      return response.json()
    }).then(setExamples).catch((error) => setError(String(error)))
  }, [])

  return (
    <div className="space-y-6">
      <div>
        <h2 className="font-display text-xl">What the model did</h2>
        <p className="mt-2 max-w-3xl text-sm text-ink-300">
          Saved model responses from the same VQA questions. These examples illustrate
          failure modes; the dashboard provides broader accuracy comparisons. No GPU is needed here.
        </p>
      </div>
      {error && <p role="alert">{error}</p>}
      {examples.map((example) => (
        <section key={example.title} className="rounded-lg border border-ink-700 bg-ink-900 p-5">
          <h3 className="font-display text-lg text-ink-50">{example.title}</h3>
          {example.study && <p className="mt-1 text-xs text-ink-400">{example.study}</p>}
          <p className="mt-1 text-sm text-ink-300">{example.lesson}</p>
          <div className="mt-4 grid gap-5 lg:grid-cols-[240px_1fr]">
            <div>
              <img src={example.image} alt="Image used for this VQA question" className="w-full rounded" />
              <p className="mt-3 text-sm">{example.question}</p>
              <p className="mt-2 text-xs text-ink-400">{example.reference_label ?? 'References'}: {[...new Set(example.answers)].join(', ')}</p>
              <p className="mt-1 font-mono text-xs text-ink-400">{example.dataset ?? 'VQAv2'} question {example.qid}</p>
            </div>
            <div className="space-y-4">
              {example.responses.map((response) => (
                <div key={response.run_id} className="rounded border border-ink-700 p-3">
                  <p className="text-sm text-ember-400">{response.label}</p>
                  {response.image && <img src={response.image} alt={response.label} className="mt-2 max-h-36 rounded" />}
                  <p className="mt-1 text-xs text-ink-400">
                    {response.official !== null && response.normalized !== null
                      ? `Original VQA score: ${response.official.toFixed(2)} · Always-normalized score: ${response.normalized.toFixed(2)}`
                      : 'Recorded intervention; altered images have no reference answers.'}
                  </p>
                  <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap font-mono text-xs leading-relaxed text-ink-200">{response.raw}</pre>
                  <p className="mt-2 break-all font-mono text-[10px] text-ink-400">{response.run_id}</p>
                </div>
              ))}
            </div>
          </div>
        </section>
      ))}
    </div>
  )
}
