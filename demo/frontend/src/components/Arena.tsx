import { useCallback, useEffect, useRef, useState } from 'react'
import { ask, fetchModels, type ModelSpec } from '../lib/api'
import ModelCard, { type CardState } from './ModelCard'
import { MethodDot, PageIntro, Segmented } from './ui'

const EMPTY: CardState = { text: '', status: 'idle', startedAt: null, finishedAt: null }

type Template = 'short' | 'reasoning'
type ImageCondition = 'original' | 'black'

export default function Arena() {
  const [models, setModels] = useState<ModelSpec[]>([])
  const [image, setImage] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [question, setQuestion] = useState('')
  const [template, setTemplate] = useState<Template>('reasoning')
  const [selected, setSelected] = useState<string[]>([])
  const [imageCondition, setImageCondition] = useState<ImageCondition>('original')
  const [cards, setCards] = useState<Record<string, CardState>>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetchModels().then((items) => {
      setModels(items)
      setSelected(items.filter((item) => item.loaded).map((item) => item.key))
    }).catch((error) => setError(String(error)))
  }, [])

  const acceptFile = useCallback((file: File | undefined) => {
    if (!file || !file.type.startsWith('image/')) return
    setImage(file)
    setPreview(URL.createObjectURL(file))
  }, [])

  const live = models.some((model) => model.loaded)
  const ready = Boolean(image && question.trim() && selected.length) && !busy

  const submit = async () => {
    if (!ready || !image) return
    setBusy(true)
    setError('')
    const now = Date.now()
    setCards(Object.fromEntries(
      selected.map((key) => [key, { ...EMPTY, status: 'streaming', startedAt: now }]),
    ))
    try {
      await ask(
        image,
        question,
        template,
        (model, token) => setCards((c) => ({ ...c, [model]: { ...c[model], text: c[model].text + token } })),
        (model) => setCards((c) => ({ ...c, [model]: { ...c[model], status: 'done', finishedAt: Date.now() } })),
        selected,
        imageCondition,
      )
    } catch (error) {
      setError(String(error))
      setCards((cards) => Object.fromEntries(Object.entries(cards).map(([key, card]) =>
        [key, { ...card, status: 'done', finishedAt: Date.now() }],
      )))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <PageIntro title="Live comparison">
        Ask the 2B models the same question about your own image and read their answers side by
        side. Try the step-by-step prompt on a counting question, then repeat it with a black
        image to see which answers depend on the picture.
      </PageIntro>

      {models.length > 0 && !live && (
        <p className="mt-6 max-w-3xl rounded-md border border-rule bg-wash px-4 py-3 text-sm text-ink-2">
          No models are loaded on this server. Start it with <code className="font-mono text-[13px]">VQA_LIVE_MODELS=1</code>{' '}
          on a 24 GB GPU (see the demo notes), or browse the recorded examples, which need no GPU.
        </p>
      )}
      {error && <p role="alert" className="mt-4 text-sm text-miss">{error}</p>}

      <div className="mt-10 grid gap-10 lg:grid-cols-[320px_minmax(0,1fr)]">
        <div className="space-y-6">
          <div
            role="button"
            tabIndex={0}
            aria-label="Choose an image"
            onClick={() => fileInput.current?.click()}
            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && fileInput.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault()
              acceptFile(e.dataTransfer.files[0])
            }}
            className="flex aspect-[4/3] cursor-pointer items-center justify-center overflow-hidden rounded-lg border border-dashed border-line bg-wash transition-colors hover:border-ink-3"
          >
            {preview ? (
              <img src={preview} alt="Your image" className="h-full w-full object-contain" />
            ) : (
              <div className="p-6 text-center">
                <svg aria-hidden viewBox="0 0 24 24" className="mx-auto h-7 w-7 text-ink-4" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <rect x="3" y="4" width="18" height="16" rx="2" />
                  <circle cx="9" cy="10" r="2" />
                  <path d="m21 16-5-5-9 9" />
                </svg>
                <p className="mt-3 text-sm text-ink-2">Drop an image, or click to choose one</p>
              </div>
            )}
            <input ref={fileInput} type="file" accept="image/*" className="hidden"
              onChange={(e) => acceptFile(e.target.files?.[0])} />
          </div>

          <label className="block">
            <span className="eyebrow">Question</span>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), submit())}
              placeholder="How many people are in the picture?"
              rows={2}
              className="mt-1.5 w-full resize-none rounded-md border border-line bg-paper px-3 py-2 text-[15px] placeholder-ink-4 outline-none transition-shadow focus:border-accent focus:ring-2 focus:ring-accent/15"
            />
          </label>

          <div className="space-y-3">
            <Segmented label="Prompt" value={template} onChange={setTemplate} disabled={busy} options={[
              { value: 'short', label: 'Short answer' }, { value: 'reasoning', label: 'Step by step' }]} />
            <Segmented label="Image" value={imageCondition} onChange={setImageCondition} disabled={busy} options={[
              { value: 'original', label: 'As given' }, { value: 'black', label: 'Black, same size' }]} />
          </div>

          <fieldset disabled={busy} className="space-y-2">
            <legend className="eyebrow mb-2">Models</legend>
            {models.map((model) => (
              <label key={model.key} className={`flex items-center gap-3 text-sm ${model.loaded ? '' : 'text-ink-4'}`}>
                <input type="checkbox" className="h-4 w-4 accent-[#1c1d20]" disabled={!model.loaded}
                  checked={selected.includes(model.key)}
                  onChange={(event) => setSelected((keys) => event.target.checked
                    ? [...keys, model.key] : keys.filter((key) => key !== model.key))} />
                <MethodDot name={model.key} />
                {model.label}
              </label>
            ))}
          </fieldset>

          <button
            onClick={submit}
            disabled={!ready}
            className="w-full rounded-md bg-ink py-2.5 text-sm font-medium text-paper transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-25"
          >
            {busy ? 'Generating…' : 'Compare answers'}
          </button>
        </div>

        <div className="grid content-start gap-4 md:grid-cols-3">
          {models.map((model) => (
            <ModelCard key={model.key} spec={model} state={cards[model.key] ?? EMPTY} />
          ))}
        </div>
      </div>
    </div>
  )
}
