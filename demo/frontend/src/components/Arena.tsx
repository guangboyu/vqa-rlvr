import { useCallback, useEffect, useRef, useState } from 'react'
import { ask, fetchModels, type ModelSpec } from '../lib/api'
import ModelCard, { type CardState } from './ModelCard'

const EMPTY: CardState = { text: '', status: 'idle', startedAt: null, finishedAt: null }

export default function Arena() {
  const [models, setModels] = useState<ModelSpec[]>([])
  const [image, setImage] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [question, setQuestion] = useState('')
  const [template, setTemplate] = useState<'short' | 'reasoning'>('reasoning')
  const [selected, setSelected] = useState<string[]>([])
  const [imageCondition, setImageCondition] = useState<'original' | 'black'>('original')
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

  const submit = async () => {
    if (!image || !question.trim() || busy || !selected.length) return
    setBusy(true)
    setError('')
    const now = Date.now()
    setCards(
      Object.fromEntries(
        selected.map((key) => [key, { ...EMPTY, status: 'streaming', startedAt: now }]),
      ),
    )
    try {
      await ask(
        image,
        question,
        template,
        (model, token) =>
          setCards((c) => ({
            ...c,
            [model]: { ...c[model], text: c[model].text + token },
          })),
        (model) =>
          setCards((c) => ({
            ...c,
            [model]: { ...c[model], status: 'done', finishedAt: Date.now() },
          })),
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
    <div className="grid gap-6 lg:grid-cols-[minmax(280px,340px)_1fr]">
      {/* Left: inputs */}
      <div className="space-y-4">
        {models.length > 0 && !models.some((model) => model.loaded) && (
          <p className="text-sm text-ink-300">Live models are unavailable. Open the gallery to explore saved responses without a GPU.</p>
        )}
        {error && <p role="alert" className="text-sm text-ink-300">{error}</p>}
        <div
          onClick={() => fileInput.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            acceptFile(e.dataTransfer.files[0])
          }}
          className="flex aspect-square cursor-pointer items-center justify-center overflow-hidden rounded-lg border-2 border-dashed border-ink-600 bg-ink-900 transition-colors hover:border-ember-500"
        >
          {preview ? (
            <img src={preview} alt="query" className="h-full w-full object-contain" />
          ) : (
            <div className="p-6 text-center text-sm text-ink-400">
              <p className="font-display text-3xl text-ink-600">⌒◡⌒</p>
              <p className="mt-2">Drop an image here or click to browse</p>
            </div>
          )}
          <input
            ref={fileInput}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => acceptFile(e.target.files?.[0])}
          />
        </div>

        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), submit())}
          placeholder="Ask something about the image…"
          rows={2}
          className="w-full resize-none rounded-lg border border-ink-700 bg-ink-900 px-3 py-2 text-sm text-ink-50 placeholder-ink-600 outline-none focus:border-ember-500"
        />

        <div className="flex items-center gap-2">
          <div className="flex overflow-hidden rounded-md border border-ink-700 text-xs">
            {(['short', 'reasoning'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTemplate(t)}
                className={`px-3 py-1.5 font-mono transition-colors ${
                  template === t ? 'bg-ember-500 text-ink-950' : 'bg-ink-900 text-ink-300'
                }`}
              >
                {t === 'short' ? 'direct answer' : 'chain-of-thought'}
              </button>
            ))}
          </div>
          <button
            onClick={submit}
            disabled={!image || !question.trim() || busy || !selected.length}
            className="ml-auto rounded-md bg-ember-500 px-5 py-1.5 text-sm font-semibold text-ink-950 transition-opacity disabled:opacity-30"
          >
            {busy ? 'Generating…' : 'Compare'}
          </button>
        </div>

        <fieldset className="space-y-2 text-xs text-ink-300" disabled={busy}>
          <legend className="mb-2">Models</legend>
          {models.map((model) => <label key={model.key} className="flex gap-2">
            <input type="checkbox" disabled={!model.loaded} checked={selected.includes(model.key)}
              onChange={(event) => setSelected((keys) => event.target.checked
                ? [...keys, model.key] : keys.filter((key) => key !== model.key))} />
            {model.label}
          </label>)}
        </fieldset>
        <label className="block text-xs text-ink-300">
          Image input
          <select value={imageCondition} disabled={busy}
            onChange={(event) => setImageCondition(event.target.value as 'original' | 'black')}
            className="ml-3 rounded border border-ink-700 bg-ink-900 p-1">
            <option value="original">Original image</option>
            <option value="black">Black image, same dimensions</option>
          </select>
        </label>

        <p className="text-xs leading-relaxed text-ink-400">
          Three checkpoints of the same 2B model answer side by side: the base model, its
          supervised fine-tune, and the RLVR (GRPO) policy. Try chain-of-thought mode on a
          counting question to see the training differences. Repeat with a black image to
          inspect sensitivity; a changed answer alone does not establish correct grounding.
        </p>
      </div>

      {/* Right: model cards */}
      <div className="grid gap-4 md:grid-cols-3">
        {models.map((m) => (
          <ModelCard key={m.key} spec={m} state={cards[m.key] ?? EMPTY} />
        ))}
      </div>
    </div>
  )
}
