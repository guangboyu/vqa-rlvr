"""VQA metrics.

`vqa_accuracy` ports the official VQA v2 evaluation (GT-Vision-Lab/VQA,
vqaEval.py): answer normalization (punctuation, number words, articles,
contractions) plus the 10-annotator leave-one-out accuracy
``mean_over_annotators(min(#matches_among_other_9 / 3, 1))``.

One deliberate deviation, matching lmms-eval: normalization is applied
*unconditionally*. The 2016 code only normalized when annotators disagreed, which
zeroes e.g. prediction "Yes" against a unanimous ["yes"]*10 — an artifact for
zero-shot instruct models that capitalize. Modern harnesses (lmms-eval's vqav2
task) always normalize both sides; we follow them for comparability.

These functions are scorers; `vqar.registry` assigns each dataset its official scorer
(the dataset authors' evaluator) and its normalized scorer (this unconditional rule, also
the RL reward).
"""

import re

# Bump when any scorer's behavior changes: it is part of every evaluation protocol's identity.
SCORING_VERSION = 1

CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd",
    "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all",
    "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
    "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll",
    "youre": "you're", "youve": "you've",
}

NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

ARTICLES = {"a", "an", "the"}

# Verbatim from the official eval, including its well-known lookbehind typo
# ("(?!<=" instead of "(?<!"): kept so our scores match every published number.
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_PUNCT = [
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!",
]


def _process_punctuation(text: str) -> str:
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or _COMMA_STRIP.search(text):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return _PERIOD_STRIP.sub("", out)


def _process_digit_article(text: str) -> str:
    words = []
    for word in text.lower().split():
        word = NUMBER_MAP.get(word, word)
        if word not in ARTICLES:
            words.append(word)
    return " ".join(CONTRACTIONS.get(w, w) for w in words)


def normalize_answer(text: str) -> str:
    """Official VQA answer normalization, applied to a single answer string."""
    text = text.replace("\n", " ").replace("\t", " ").strip()
    return _process_digit_article(_process_punctuation(text))


def vqa_accuracy(prediction: str, gt_answers: list[str]) -> float:
    """VQA v2 accuracy for one question against its 10 annotator answers.

    Leave-one-out requires multiple annotators: with a single reference answer
    ``others`` is empty and EVERY prediction scores 0.0 — which silently zeroed
    the correctness reward for all single-gold vqav2_rl training examples in v1
    runs. Single-reference inputs therefore fall back to normalized exact match.
    """
    if len(gt_answers) < 2:
        return exact_match(prediction, gt_answers)
    prediction = normalize_answer(prediction)
    gt_answers = [normalize_answer(a) for a in gt_answers]
    accs = []
    for i in range(len(gt_answers)):
        others = gt_answers[:i] + gt_answers[i + 1 :]
        matches = sum(a == prediction for a in others)
        accs.append(min(1.0, matches / 3))
    return sum(accs) / len(accs)


def exact_match(prediction: str, gt_answers: list[str]) -> float:
    """Normalized exact match, including single-reference training examples."""
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(a) for a in gt_answers))


def official_vqa_accuracy(prediction: str, gt_answers: list[str]) -> float:
    """GT-Vision-Lab's evaluator, including its conditional normalization."""
    if len(gt_answers) != 10:
        raise ValueError("Official VQA evaluation needs all 10 annotator answers.")
    def clean(text):
        return text.replace("\n", " ").replace("\t", " ").strip()

    prediction, answers = clean(prediction), [clean(answer) for answer in gt_answers]
    if len(set(answers)) > 1:
        prediction = normalize_answer(prediction)
        answers = [normalize_answer(answer) for answer in answers]
    matches = sum(answer == prediction for answer in answers)
    return sum(min(1.0, (matches - (answer == prediction)) / 3) for answer in answers) / 10


def textvqa_accuracy(prediction: str, gt_answers: list[str]) -> float:
    """MMF TextVQA evaluator: EvalAI tokenization, then annotator agreement."""
    if len(gt_answers) != 10:
        raise ValueError("TextVQA evaluation needs all 10 annotator answers.")

    def tokenize(text):
        return text.lower().replace(",", "").replace("?", "").replace("'s", " 's").strip()

    return vqa_accuracy(tokenize(prediction), [tokenize(answer) for answer in gt_answers])


def exact_string_match(prediction: str, gt_answers: list[str]) -> float:
    """GQA's author accuracy: the raw prediction string must equal a reference."""
    return float(prediction in gt_answers)


def casefold_exact_match(prediction: str, gt_answers: list[str]) -> float:
    """Strict project scorer: the case-folded, whitespace-collapsed prediction equals a
    reference; no punctuation, article or number normalization."""
    pred = " ".join(prediction.casefold().split())
    return float(bool(pred) and any(pred == " ".join(a.casefold().split()) for a in gt_answers))


def _relaxed_float(text: str) -> float | None:
    try:
        return float(text.rstrip("%")) / 100.0 if text.endswith("%") else float(text)
    except ValueError:
        return None


def chartqa_relaxed_accuracy(prediction: str, gt_answers: list[str]) -> float:
    """ChartQA's relaxed accuracy (Masry et al. 2022, sec. 5.1), as lmms-eval and Qwen-VL
    implement it: a number within 5% of a numeric reference, else a case-insensitive match."""
    def correct(target: str) -> bool:
        predicted, expected = _relaxed_float(prediction), _relaxed_float(target)
        if predicted is not None and expected:
            return abs(predicted - expected) / abs(expected) <= 0.05
        return prediction.lower() == target.lower()

    return float(any(correct(answer) for answer in gt_answers))


def _number(text: str) -> float | None:
    """A number written with thousands separators, a currency sign or a percent sign."""
    cleaned = text.strip().replace(",", "").removeprefix("$").removesuffix(".").strip()
    return _relaxed_float(cleaned)


def chartqa_normalized_accuracy(prediction: str, gt_answers: list[str]) -> float:
    """The relaxed rule after VQA normalization: '1,234' and '$1234' read as 1234, and text
    answers compare normalized (case, punctuation, articles, number words)."""
    def correct(target: str) -> bool:
        predicted, expected = _number(prediction), _number(target)
        if predicted is not None and expected is not None:
            if expected == 0:
                return predicted == 0
            return abs(predicted - expected) / abs(expected) <= 0.05
        return normalize_answer(prediction) == normalize_answer(target)

    return float(any(correct(answer) for answer in gt_answers))


# Unicode hyphens and dashes models write where ChartQA labels use an ASCII hyphen.
_DASHES = str.maketrans(dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2212", "-"))


def chartqa_list_accuracy(prediction: str, gt_answers: list[str]) -> float:
    """The normalized rule with Unicode dashes read as hyphens, and bracketed list references
    ('[Gambia, Niger]') matched as unordered sets of items: 'Niger, Gambia' counts."""
    prediction = prediction.translate(_DASHES)
    gt_answers = [answer.translate(_DASHES) for answer in gt_answers]
    if chartqa_normalized_accuracy(prediction, gt_answers):
        return 1.0

    def items(text: str) -> list[str]:
        return sorted(normalize_answer(part) for part in text.strip().strip("[]").split(","))

    return float(any(answer.strip().startswith("[") and items(prediction) == items(answer)
                     for answer in gt_answers))


_CHOICE = re.compile(r"^\s*(?:option\s+|answer\s*:?\s*)?\(?([A-Ea-e])\)?(?:[.):,\s]|$)",
                     re.IGNORECASE)


def choice_letter_accuracy(prediction: str, gt_answers: list[str]) -> float:
    """Multiple choice, strict: the prediction is the reference letter alone ('C' or 'C.')."""
    return float(prediction.strip().rstrip(".").upper() == gt_answers[0].upper())


def choice_accuracy(prediction: str, gt_answers: list[str]) -> float:
    """Multiple choice, lenient: a leading option letter ('C', '(C)', 'C. Bossing', 'Option C')
    must be the reference letter; without one, the text must match the reference option's
    text (answers hold the letter, then the option text) after VQA normalization."""
    match = _CHOICE.match(prediction)
    if match:
        return float(match.group(1).upper() == gt_answers[0].upper())
    return float(any(normalize_answer(prediction) == normalize_answer(text)
                     for text in gt_answers[1:]))


# Kvasir-VQA labels: values from a closed vocabulary, several joined by "; " ("pink; red").
# Two label typos are fixed by normalization only; the stored labels stay as released.
_LABEL_SEPARATORS = re.compile(r";|,|\band\b")


def _kvasir_fix(text: str) -> str:
    text = re.sub(r"(\d)\s+mm\b", r"\1mm", text.casefold().replace("rigth", "right"))
    return re.sub(r">\s*20(?!mm)", ">20mm", text)


def label_set_match(prediction: str, gt_answers: list[str]) -> float:
    """Strict: the values, split on ";" and case-folded, equal a reference's values as a set."""
    def values(text):
        return {value.strip() for value in text.casefold().split(";") if value.strip()}

    return float(any(values(prediction) == values(answer) for answer in gt_answers))


def normalized_label_set_match(prediction: str, gt_answers: list[str]) -> float:
    """Lenient: values split on ";", "," or "and", VQA-normalized, the two label typos fixed,
    and compared as sets, so order and separators do not matter."""
    def values(text):
        parts = _LABEL_SEPARATORS.split(_kvasir_fix(text))
        return {normalize_answer(part) for part in parts if normalize_answer(part)}

    return float(any(values(prediction) == values(answer) for answer in gt_answers))


# DeepTumorVQA free-form answers (2026-11-01): numbers scored by mean relative accuracy, as
# the dataset's authors score them (their Appendix ab, after VSI-Bench), text by exact match.
_SIGNED_DECIMAL = re.compile(r"-?\d+(?:\.\d+)?")
MRA_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def mean_relative_accuracy(prediction: str, gt_answers: list[str]) -> float:
    """A numeric reference scores the share of thresholds t in 0.50, 0.55, ..., 0.95 with
    |prediction − reference| / |reference| < 1 − t, reading the prediction's first signed
    decimal (none scores 0; a zero reference needs a zero). A text reference needs a normalized
    exact match."""
    reference = gt_answers[0].strip()
    try:
        expected = float(reference)
    except ValueError:
        return float(normalize_answer(prediction) == normalize_answer(reference))
    found = _SIGNED_DECIMAL.search(prediction.replace(",", ""))
    if found is None:
        return 0.0
    predicted = float(found.group())
    if expected == 0:
        return float(predicted == 0)
    error = abs(predicted - expected) / abs(expected)
    return sum(error < 1 - t for t in MRA_THRESHOLDS) / len(MRA_THRESHOLDS)
