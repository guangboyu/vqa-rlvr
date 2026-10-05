"""Frozen OOD scoring rules, independent of the existing four datasets' scorers.

VizWiz ports the authors' prediction-only normalizer and raw-reference agreement.
Source: https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/API.zip
Native vqaEval.py SHA256: d26ef7ac2db06351444c3531f8d95fd9c454423fbcb9fd5dd984f12281146694.
RealWorldQA uses declared project exact match; no native evaluator is claimed.
"""

import re

VIZWIZ_CONTRACTIONS = {'aint': "ain't",
 'arent': "aren't",
 'cant': "can't",
 'couldve': "could've",
 'couldnt': "couldn't",
 "couldn'tve": 'couldn’t’ve',
 'couldnt’ve': 'couldn’t’ve',
 'didnt': 'didn’t',
 'doesnt': 'doesn’t',
 'dont': 'don’t',
 'hadnt': 'hadn’t',
 'hadnt’ve': 'hadn’t’ve',
 "hadn'tve": 'hadn’t’ve',
 'hasnt': 'hasn’t',
 'havent': 'haven’t',
 'hed': 'he’d',
 'hed’ve': 'he’d’ve',
 'he’dve': 'he’d’ve',
 'hes': 'he’s',
 'howd': 'how’d',
 'howll': 'how’ll',
 'hows': 'how’s',
 'Id’ve': 'I’d’ve',
 'I’dve': 'I’d’ve',
 'Im': 'I’m',
 'Ive': 'I’ve',
 'isnt': 'isn’t',
 'itd': 'it’d',
 'itd’ve': 'it’d’ve',
 'it’dve': 'it’d’ve',
 'itll': 'it’ll',
 'let’s': 'let’s',
 'maam': 'ma’am',
 'mightnt': 'mightn’t',
 'mightnt’ve': 'mightn’t’ve',
 'mightn’tve': 'mightn’t’ve',
 'mightve': 'might’ve',
 'mustnt': 'mustn’t',
 'mustve': 'must’ve',
 'neednt': 'needn’t',
 'notve': 'not’ve',
 'oclock': 'o’clock',
 'oughtnt': 'oughtn’t',
 'ow’s’at': '’ow’s’at',
 '’ows’at': '’ow’s’at',
 '’ow’sat': '’ow’s’at',
 'shant': 'shan’t',
 'shed’ve': 'she’d’ve',
 'she’dve': 'she’d’ve',
 'she’s': 'she’s',
 'shouldve': 'should’ve',
 'shouldnt': 'shouldn’t',
 'shouldnt’ve': 'shouldn’t’ve',
 'shouldn’tve': 'shouldn’t’ve',
 'somebody’d': 'somebodyd',
 'somebodyd’ve': 'somebody’d’ve',
 'somebody’dve': 'somebody’d’ve',
 'somebodyll': 'somebody’ll',
 'somebodys': 'somebody’s',
 'someoned': 'someone’d',
 'someoned’ve': 'someone’d’ve',
 'someone’dve': 'someone’d’ve',
 'someonell': 'someone’ll',
 'someones': 'someone’s',
 'somethingd': 'something’d',
 'somethingd’ve': 'something’d’ve',
 'something’dve': 'something’d’ve',
 'somethingll': 'something’ll',
 'thats': 'that’s',
 'thered': 'there’d',
 'thered’ve': 'there’d’ve',
 'there’dve': 'there’d’ve',
 'therere': 'there’re',
 'theres': 'there’s',
 'theyd': 'they’d',
 'theyd’ve': 'they’d’ve',
 'they’dve': 'they’d’ve',
 'theyll': 'they’ll',
 'theyre': 'they’re',
 'theyve': 'they’ve',
 'twas': '’twas',
 'wasnt': 'wasn’t',
 'wed’ve': 'we’d’ve',
 'we’dve': 'we’d’ve',
 'weve': "we've",
 'werent': 'weren’t',
 'whatll': 'what’ll',
 'whatre': 'what’re',
 'whats': 'what’s',
 'whatve': 'what’ve',
 'whens': 'when’s',
 'whered': 'where’d',
 'wheres': "where's",
 'whereve': 'where’ve',
 'whod': 'who’d',
 'whod’ve': 'who’d’ve',
 'who’dve': 'who’d’ve',
 'wholl': 'who’ll',
 'whos': 'who’s',
 'whove': "who've",
 'whyll': 'why’ll',
 'whyre': 'why’re',
 'whys': 'why’s',
 'wont': 'won’t',
 'wouldve': 'would’ve',
 'wouldnt': 'wouldn’t',
 'wouldnt’ve': 'wouldn’t’ve',
 'wouldn’tve': 'wouldn’t’ve',
 'yall': 'y’all',
 'yall’ll': 'y’all’ll',
 'y’allll': 'y’all’ll',
 'yall’d’ve': 'y’all’d’ve',
 'y’alld’ve': 'y’all’d’ve',
 'y’all’dve': 'y’all’d’ve',
 'youd': 'you’d',
 'youd’ve': 'you’d’ve',
 'you’dve': 'you’d’ve',
 'youll': 'you’ll',
 'youre': 'you’re',
 'youve': 'you’ve'}

VIZWIZ_NUMBERS = {'none': '0',
 'zero': '0',
 'one': '1',
 'two': '2',
 'three': '3',
 'four': '4',
 'five': '5',
 'six': '6',
 'seven': '7',
 'eight': '8',
 'nine': '9',
 'ten': '10'}

VIZWIZ_ARTICLES = frozenset({'the', 'an', 'a'})
VIZWIZ_PUNCTUATION = [';',
 '/',
 '[',
 ']',
 '"',
 '{',
 '}',
 '(',
 ')',
 '=',
 '+',
 '\\',
 '_',
 '-',
 '>',
 '<',
 '@',
 '`',
 ',',
 '?',
 '!']

_PERIOD = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA = re.compile(r"(\d)(,)(\d)")


def vizwiz_normalize_prediction(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ").strip()
    out = text
    for punctuation in VIZWIZ_PUNCTUATION:
        if (punctuation + " " in text or " " + punctuation in text) or _COMMA.search(text):
            out = out.replace(punctuation, "")
        else:
            out = out.replace(punctuation, " ")
    # The native evaluator passes re.UNICODE as the replacement count (32), not flags.
    out = _PERIOD.sub("", out, re.UNICODE)
    words = [VIZWIZ_NUMBERS.get(word, word) for word in out.lower().split()]
    return " ".join(VIZWIZ_CONTRACTIONS.get(word, word) for word in words
                    if word not in VIZWIZ_ARTICLES)


def official_vizwiz_accuracy(prediction: str, answers: list[str]) -> float:
    """Native ten-annotator leave-one-out accuracy; ground-truth strings stay raw."""
    if len(answers) != 10:
        raise ValueError("VizWiz evaluation needs all 10 annotator answers")
    prediction = vizwiz_normalize_prediction(prediction)
    matches = sum(answer == prediction for answer in answers)
    return sum(min(1, (matches - (answer == prediction)) / 3) for answer in answers) / 10


def project_case_whitespace_em(prediction: str, answers: list[str]) -> float:
    """Declared RealWorldQA project metric: casefold and collapse whitespace only."""
    if len(answers) != 1:
        raise ValueError("RealWorldQA project exact match needs one reference")
    pred = " ".join(prediction.casefold().split())
    gold = " ".join(answers[0].casefold().split())
    return float(bool(pred) and pred == gold)
