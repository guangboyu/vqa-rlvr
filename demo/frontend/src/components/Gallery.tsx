import { useEffect, useState } from 'react'
import { responseParts } from '../lib/methods'
import { ModelName, PageIntro, Score } from './ui'

interface Response {
  label: string
  run_id: string
  raw: string
  image?: string
  official: number | null
  normalized: number | null
}

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
  responses: Response[]
}

export default function Gallery() {
  const [examples, setExamples] = useState<Example[]>([])
  const [error, setError] = useState('')
  useEffect(() => {
    fetch('/api/gallery').then((response) => {
      if (!response.ok) throw new Error('Could not load the recorded examples.')
      return response.json()
    }).then(setExamples).catch((error) => setError(String(error)))
  }, [])

  return (
    <div>
      <PageIntro title="Examples">
        Recorded answers from the models on the same test questions, each with its score. They
        show how a scorer, a training method or a changed image moves a single answer; the
        Results view has the aggregate numbers. No GPU is needed.
      </PageIntro>
      {error && <p role="alert" className="mt-8 text-sm text-miss">{error}</p>}
      <div className="mt-10">
        {examples.map((example) => <ExampleView key={example.title} example={example} />)}
      </div>
    </div>
  )
}

function ExampleView({ example }: { example: Example }) {
  const references = [...new Set(example.answers)]
  return (
    <article className="grid gap-8 border-t border-rule py-10 lg:grid-cols-[260px_minmax(0,1fr)]">
      <figure className="space-y-4">
        <img src={example.image} alt="The image this question is about"
          className="w-full rounded-md border border-rule" />
        <figcaption className="space-y-3">
          <p className="font-serif text-[19px] leading-snug">{example.question}</p>
          <div>
            <p className="eyebrow">{example.reference_label ?? 'Reference answers'}</p>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {references.map((answer) => (
                <span key={answer} className="rounded border border-rule bg-wash px-2 py-0.5 text-[13px]">
                  {answer}
                </span>
              ))}
            </div>
          </div>
          <p className="text-xs text-ink-3">{example.dataset ?? 'VQAv2'} · question {example.qid}</p>
        </figcaption>
      </figure>
      <div className="min-w-0">
        {example.study && <p className="eyebrow">{example.study}</p>}
        <h2 className="mt-1 font-serif text-[24px] font-medium leading-tight">{example.title}</h2>
        <p className="mt-2 max-w-2xl text-[15px] leading-relaxed text-ink-2">{example.lesson}</p>
        <div className="mt-6 space-y-4">
          {example.responses.map((response) => <ResponseView key={response.run_id} response={response} />)}
        </div>
      </div>
    </article>
  )
}

function ResponseView({ response }: { response: Response }) {
  const { reasoning, answer } = responseParts(response.raw)
  const scored = response.official !== null && response.normalized !== null
  return (
    <div className="rounded-lg border border-rule">
      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-1 border-b border-rule px-4 py-2.5">
        <ModelName name={response.label} className="text-sm font-medium" />
        {scored ? (
          <div className="flex gap-4">
            <Score label="VQA score" value={response.official!} />
            <Score label="Normalized" value={response.normalized!} />
          </div>
        ) : (
          <span className="text-xs text-ink-3">Changed image, so no reference answer</span>
        )}
      </div>
      <div className="flex gap-4 px-4 py-3">
        {response.image && (
          <img src={response.image} alt={response.label} className="h-24 w-24 shrink-0 rounded border border-rule object-cover" />
        )}
        <div className="min-w-0 flex-1">
          {reasoning && (
            <p className="max-h-56 overflow-auto whitespace-pre-wrap font-mono text-[12.5px] leading-relaxed text-ink-2">
              {reasoning}
            </p>
          )}
          {answer !== null && (
            <p className={`${reasoning ? 'mt-3' : ''} text-sm`}>
              <span className="eyebrow mr-2">Answer</span>
              <span className="font-serif text-[18px]">{answer}</span>
            </p>
          )}
        </div>
      </div>
      <p className="border-t border-rule px-4 py-1.5 font-mono text-[10.5px] text-ink-4">{response.run_id}</p>
    </div>
  )
}
