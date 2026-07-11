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

GQA / CLEVR / TextVQA-style single-answer datasets use normalized exact match
with the same normalizer.
"""

import re

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
    """VQA v2 accuracy for one question against its 10 annotator answers."""
    prediction = normalize_answer(prediction)
    gt_answers = [normalize_answer(a) for a in gt_answers]
    accs = []
    for i in range(len(gt_answers)):
        others = gt_answers[:i] + gt_answers[i + 1 :]
        matches = sum(a == prediction for a in others)
        accs.append(min(1.0, matches / 3))
    return sum(accs) / len(accs)


def exact_match(prediction: str, gt_answers: list[str]) -> float:
    """Normalized exact match for single-answer datasets (GQA, CLEVR, TextVQA)."""
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(a) for a in gt_answers))


def score(dataset: str, prediction: str, gt_answers: list[str]) -> float:
    """Dispatch to the right metric for a dataset. Returns a value in [0, 1]."""
    if dataset == "vqav2":
        return vqa_accuracy(prediction, gt_answers)
    return exact_match(prediction, gt_answers)
