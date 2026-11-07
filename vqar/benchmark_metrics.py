"""Scoring rules of the public benchmarks: each a port of its published rule, or declared ours.

Every rule takes (prediction, references) like the scorers in `vqar.metrics`. Multiple-choice
references hold the letter, the correct option's text, then every option in letter order, because
the rule-based option readers need the whole list.

Sources, read 2026-11-06; the registry's metric metadata names each one per dataset:
- MMStar: MMStar-Benchmark/MMStar eval/vlmeval/evaluate/mmstar.py
- MMMU: MMMU-Benchmark/MMMU mmmu/utils/eval_utils.py
- OCRBench: Yuliang-Liu/MultimodalOCR OCRBench/example.py
- DocVQA: QwenLM/Qwen-VL eval_mm/infographicsvqa_eval.py, a copy of the RRC script
- POPE: RUCAIBox/POPE evaluate.py
- MathVista: normalization from lupantech/MathVista evaluation/calculate_score.py
- Option reading: open-compass/VLMEvalKit vlmeval/utils/matching_util.py (can_infer)
"""

import ast
import math
import operator
import random
import re
import unicodedata

from vqar.metrics import exact_match, normalize_answer


def is_choice(answers: list[str]) -> bool:
    """Whether references hold a letter, its option's text and the option list. MMMU keeps open
    questions whose answer is a bare letter ('C'); those have a single reference."""
    return len(answers) > 2 and bool(re.fullmatch(r"[A-Z]", answers[0]))


def options_of(answers: list[str]) -> dict[str, str]:
    return {chr(ord("A") + i): text for i, text in enumerate(answers[2:])}


# MMStar's authors read only the start of a response.
def mmstar_accuracy(prediction: str, answers: list[str]) -> float:
    """The answer letter opens the response, alone or after '(', 'option ' or 'the answer is '."""
    answer = answers[0].lower().strip().replace("\n", " ")
    predict = prediction.lower().strip().replace("\n", " ")
    return float(predict[:1] == answer
                 or (predict[:1] == "(" and predict[1:2] == answer)
                 or (predict[:7] == "option " and predict[7:8] == answer)
                 or (predict[:14] == "the answer is " and predict[14:15] == answer))


# VLMEvalKit's can_infer under its exact_matching policy; HR-Bench's repository carries the same
# code. Its list of refusal phrases is left out: those answers name no option and score zero anyway.
_OPTION_PUNCTUATION = str.maketrans(dict.fromkeys(".()[],:;!*#{}", " "))


def infer_option(response: str, options: dict[str, str]) -> str | None:
    """The one option letter the response names as a token, else the one option whose text it
    contains, else nothing."""
    tokens = response.translate(_OPTION_PUNCTUATION).split()
    letters = [letter for letter in options if letter in tokens]
    if len(letters) == 1:
        return letters[0]
    texts = [letter for letter, text in options.items() if text.lower() in response.lower()]
    return texts[0] if len(texts) == 1 else None


def option_accuracy(prediction: str, answers: list[str]) -> float:
    """Compared by option text, so either copy of a repeated option counts (MathVista has some)."""
    options = options_of(answers)
    letter = infer_option(prediction, options)
    return float(letter is not None and options[letter] == answers[1])


_LEADING_LETTER = re.compile(r"^\s*(?:option\s+|answer\s*:?\s*)?\(?([A-Z])\)?(?:[.):,\s]|$)",
                             re.IGNORECASE)


def normalized_choice_accuracy(prediction: str, answers: list[str]) -> float:
    """`metrics.choice_accuracy` for rows carrying their options: any letter the row offers can
    lead (MathVista runs to H, MMMU to I), and text is compared with the reference option only."""
    options = options_of(answers)
    match = _LEADING_LETTER.match(prediction)
    if match and match.group(1).upper() in options:
        return float(options[match.group(1).upper()] == answers[1])
    return float(bool(prediction.strip())
                 and normalize_answer(prediction) == normalize_answer(answers[1]))


def normalized_mixed_accuracy(prediction: str, answers: list[str]) -> float:
    """Multiple choice where the row has options, normalized exact match where it does not."""
    if is_choice(answers):
        return normalized_choice_accuracy(prediction, answers)
    return exact_match(prediction, answers)


# Numbers as a project scorer reads them from a short answer.
_NUMBER_TOKEN = re.compile(r"(?<![\w.^])-?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
                           r"(?:[eE][+-]?\d+)?(?:\s*(?:\^|\*\*)\s*\{?-?\d+\}?)?")
_POWER = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:\^|\*\*)\s*\{?(-?\d+)\}?")
_RATIO = re.compile(r"(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")


def read_number(text: str) -> float | None:
    """A whole string read as one number: 40,000, 1e+06, 10^-1, 1/64, 94%, $5."""
    text = text.strip().removeprefix("$").replace(",", "").strip().rstrip(".").removesuffix("%")
    try:
        if power := _POWER.fullmatch(text):
            value = float(power.group(1)) ** int(power.group(2))
        elif ratio := _RATIO.fullmatch(text):
            value = float(ratio.group(1)) / float(ratio.group(2))
        else:
            value = float(text)
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) else None


def numbers_in(text: str) -> list[float]:
    """Every number token in order; an exponent on a unit ('cm^2') is not a number."""
    values = [read_number(token) for token in _NUMBER_TOKEN.findall(text)]
    return [value for value in values if value is not None]


def last_number(text: str) -> float | None:
    """The whole answer as a number, else the last number written in it."""
    whole = read_number(text)
    if whole is not None:
        return whole
    found = numbers_in(text)
    return found[-1] if found else None


def mathvista_accuracy(prediction: str, answers: list[str]) -> float:
    """Project rule. The authors extract answers with GPT; here the option reader replaces that for
    multiple choice and the last number for free form, and the authors' normalization follows:
    integers truncate, floats round to the reference's decimals (always its `precision` field, the
    loader checks), and a list must hold the same numbers in order."""
    if is_choice(answers):
        return option_accuracy(prediction, answers)
    reference = answers[0]
    if reference.startswith("["):
        return float(numbers_in(prediction) == [float(v) for v in ast.literal_eval(reference)])
    number = last_number(prediction)
    if number is None:
        return 0.0
    if "." in reference:
        return float(round(number, len(reference.split(".")[1])) == float(reference))
    return float(int(number) == int(float(reference)))


# MMMU's parser, kept branch for branch.
def _mmmu_choice(response: str, options: dict[str, str]) -> str:
    """The authors' parse_multi_choice_response. They guess with a global RNG when nothing parses;
    this guess is seeded by the response and the letters, so a saved prediction always rescores
    the same, at the same expected credit."""
    letters = list(options)
    guess = random.Random(f"{response}|{''.join(letters)}").choice(letters)
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = f" {response} "
    bracketed = [letter for letter in letters if f"({letter})" in response]
    bare = [letter for letter in letters if f" {letter} " in response]
    by_text = []
    if not bracketed and not bare and len(response.split()) > 5:
        by_text = [letter for letter, text in options.items() if text.lower() in response.lower()]
    candidates = bracketed or bare or by_text
    if not candidates:
        return guess
    if len(candidates) == 1:
        return candidates[0]
    if bracketed:
        where = [response.rfind(f"({letter})") for letter in candidates]
    elif bare:
        where = [response.rfind(f" {letter} ") for letter in candidates]
    else:
        where = [response.lower().rfind(options[letter].lower()) for letter in candidates]
    return candidates[where.index(max(where))]


_MMMU_KEYS = ["could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result "]
_MMMU_COMMAS = re.compile(r"-?\b\d{1,3}(?:,\d{3})+\b")
_MMMU_SCIENTIFIC = re.compile(r"-?\d+(?:\.\d+)?[eE][+-]?\d+")
_MMMU_SIMPLE = re.compile(r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])")


def _mmmu_key_responses(response: str) -> list[str]:
    """The tail of each sentence after its answer indicator; the last may be an equation."""
    response = response.strip().strip(".").lower()
    sentences = re.split(r"\.\s(?=[A-Z])|\n", response)
    keys = []
    for index, sentence in enumerate(sentences):
        indicators = _MMMU_KEYS + ["="] if index == len(sentences) - 1 else _MMMU_KEYS
        shortest = None
        for indicator in indicators:
            if indicator in sentence:
                tail = sentence.split(indicator)[-1].strip()
                if not shortest or len(tail) < len(shortest):
                    shortest = tail
        if shortest and shortest.strip() not in (":", ",", ".", "!", "?", ";", "'"):
            keys.append(shortest)
    return keys or [response]


def _mmmu_normalize(text: str) -> list:
    """A number rounded to two places, else lowercase text; single characters must stand alone."""
    text = text.strip()
    try:
        return [round(float(text.replace(",", "")), 2)]
    except ValueError:
        text = text.lower()
        return [f" {text}", f"{text} "] if len(text) == 1 else [text]


def _mmmu_open(response: str) -> list:
    """The authors' parse_open_response: candidate strings and numbers from the key responses."""
    keys = _mmmu_key_responses(response)
    found = list(keys)
    for key in keys:
        found += (_MMMU_COMMAS.findall(key) + _MMMU_SCIENTIFIC.findall(key)
                  + _MMMU_SIMPLE.findall(key))
    return [value for item in found for value in _mmmu_normalize(item)]


def mmmu_accuracy(prediction: str, answers: list[str]) -> float:
    """MMMU's rule parser, multiple choice and open: an open reference counts when its text is
    inside a candidate string, or its number equals a candidate number after rounding."""
    if is_choice(answers):
        return float(_mmmu_choice(prediction, options_of(answers)) == answers[0])
    golds = [value for answer in answers for value in _mmmu_normalize(answer)]
    for candidate in _mmmu_open(prediction):
        if isinstance(candidate, str):
            if any(isinstance(gold, str) and gold in candidate for gold in golds):
                return 1.0
        elif candidate in golds:
            return 1.0
    return 0.0


def _no_spaces(text: str) -> str:
    return text.strip().replace("\n", " ").replace(" ", "")


def _ocr_form(text: str) -> str:
    return text.lower().strip().replace("\n", " ")


def ocrbench_accuracy(prediction: str, answers: list[str]) -> float:
    """OCRBench's rule: some reference is a substring of the response. Formula references
    (HME100k) compare without spaces and with case kept, the rest in lowercase. HME100k references
    keep their released trailing newline, which is how a row is recognized (the loader checks)."""
    if any(answer.endswith("\n") for answer in answers):
        predicted = _no_spaces(prediction)
        return float(any(_no_spaces(answer) in predicted for answer in answers))
    predicted = _ocr_form(prediction)
    return float(any(_ocr_form(answer) in predicted for answer in answers))


def levenshtein(first: str, second: str) -> int:
    if len(first) > len(second):
        first, second = second, first
    previous = list(range(len(first) + 1))
    for i, char in enumerate(second):
        current = [i + 1]
        for j, other in enumerate(first):
            current.append(previous[j] if char == other
                           else 1 + min(previous[j], previous[j + 1], current[-1]))
        previous = current
    return previous[-1]


def docvqa_anls(prediction: str, answers: list[str], threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity, per question: one minus the best normalized
    distance to any reference, zero below the threshold. As in the RRC script the distance reads
    tidied strings and the denominator the raw lengths."""
    def tidy(text):
        return " ".join(text.strip().lower().split())

    distances = []
    for answer in answers:
        length = max(len(answer.upper()), len(prediction.upper()))
        distances.append(levenshtein(tidy(answer), tidy(prediction)) / length if length else 0.0)
    score = 1 - min(distances)
    return score if score >= threshold else 0.0


def pope_reading(prediction: str) -> str:
    """POPE's evaluate.py: within the first sentence, 'No', 'not' or 'no' reads as no and anything
    else, a blank answer included, reads as yes."""
    words = prediction.split(".")[0].replace(",", "").split(" ")
    return "no" if {"No", "not", "no"} & set(words) else "yes"


def pope_accuracy(prediction: str, answers: list[str]) -> float:
    return float(pope_reading(prediction) == answers[0])


def pope_summary(lines: list[dict]) -> dict:
    """The rest of POPE's report over a record's prediction lines; yes is the positive class."""
    pairs = [(pope_reading(line["prediction"]), line["answers"][0]) for line in lines]
    true_yes = sum(pair == ("yes", "yes") for pair in pairs)
    said_yes = sum(said == "yes" for said, _ in pairs)
    are_yes = sum(gold == "yes" for _, gold in pairs)
    precision = true_yes / said_yes if said_yes else 0.0
    recall = true_yes / are_yes if are_yes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"yes_no": {"precision": precision, "recall": recall, "f1": f1,
                       "yes_ratio": said_yes / len(pairs)}}


_YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def yes_no_accuracy(prediction: str, answers: list[str]) -> float:
    """Project rule: the first stand-alone yes or no in the answer; without either, it is wrong."""
    found = _YES_NO.search(prediction)
    return float(found is not None and found.group(1).lower() == answers[0])


def _same_terms(prediction: str, reference: str) -> bool:
    def terms(text):
        return sorted(normalize_answer(term) for term in text.split(","))

    return "," in reference and terms(prediction) == terms(reference)


def charxiv_accuracy(prediction: str, answers: list[str]) -> float:
    """Project rule for CharXiv's reasoning questions; the authors grade with GPT. A numeric
    reference needs the same value, notation aside, rounded to six places (their grading rule says
    exactly equal, and lmms-eval's rule path has no wider tolerance). A text reference needs a
    normalized match, either string inside the other at 90% of its length (lmms-eval's relaxed
    match), or the same comma-separated terms in any order. Their allowance for answers with the
    same meaning in other words needs a judge, so this rule undercounts those."""
    reference = answers[0]
    expected = read_number(reference)
    if expected is not None:
        number = last_number(prediction)
        return float(number is not None and round(number, 6) == round(expected, 6))
    if not prediction.strip():
        return 0.0
    predicted, gold = normalize_answer(prediction), normalize_answer(reference)
    relaxed = ((gold and gold in predicted and len(gold) >= 0.9 * len(predicted))
               or (predicted and predicted in gold and len(predicted) >= 0.9 * len(gold)))
    return float(predicted == gold or bool(relaxed) or _same_terms(prediction, reference))


# Training rewards of the candidate sources: 0 or 1, so a GRPO group is right or wrong.
def anls_match(prediction: str, answers: list[str]) -> float:
    """DocVQA's ANLS at the authors' threshold, as a reward: any credit counts as right. Spans a
    little off pass by design, and so does a number one digit off: '2020' for '2021'."""
    return float(docvqa_anls(prediction, answers) > 0)


# TeX commands that read the same as a mark, or as nothing, in a final answer.
_TEX_COMMANDS = {"dfrac": r"\frac", "tfrac": r"\frac", "pi": "π", "times": "*", "cdot": "*",
                 "circ": "°", "degree": "°", "sqrt": "√", "leqslant": "≤", "leq": "≤", "le": "≤",
                 "geqslant": "≥", "geq": "≥", "ge": "≥", "left": "", "right": "", "quad": "",
                 "qquad": "", "mathrm": "", "text": "", "textbf": "", "mathbf": "", "rm": "",
                 "unit": "", "number": "", "quantity": "", "angle": "∠", "triangle": "△"}
_TEX_COMMAND = re.compile(r"\\([a-zA-Z]+)")
_TEX_SPACING = re.compile(r"\\[,;:! ()\[\]]")
_MATH_MARKS = str.maketrans({"$": None, "~": None, "×": "*", "·": "*", "−": "-", "–": "-",
                             "或": "or"})
_DEGREE_POWER = re.compile(r"(?:\{\})?\^\{?°\}?")
_MIXED_NUMBER = re.compile(r"(\d+)\\frac\{(\d+)\}\{(\d+)\}")
_ROOT_ATOM = re.compile(r"√(\d+(?:\.\d+)?|[a-zA-Z])")
_ROOT_GROUP = re.compile(r"√\{([^{}]*)\}")
_POWER_BRACES = re.compile(r"\^\{([-\w.]+)\}")
_PLAIN_BRACES = re.compile(r"(?<![a-zA-Z}√^_])\{([^{}]*)\}")
_FRACTION = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_SIMPLE_TERM = re.compile(r"[\w.]+")


def _fraction(found: re.Match) -> str:
    top, bottom = (term if _SIMPLE_TERM.fullmatch(term) else f"({term})" for term in found.groups())
    return f"{top}/{bottom}"


def _without_groups(text: str) -> str:
    """Roots, powers, fractions and plain braces undone from the inside out."""
    while True:
        simpler = _ROOT_GROUP.sub(r"√(\1)", text)
        simpler = _POWER_BRACES.sub(r"^\1", simpler)
        simpler = _PLAIN_BRACES.sub(r"\1", simpler)
        simpler = _FRACTION.sub(_fraction, simpler)
        if simpler == text:
            return text
        text = simpler


def math_form(text: str) -> str:
    """A final answer in one spelling: no dollar signs, spaces or sizing commands, degrees as °,
    roots as √(x), fractions as a/b, lowercase. '$$\\dfrac{1}{2}$$' and '1/2' read the same."""
    text = unicodedata.normalize("NFKC", text.replace("˚", "°")).translate(_MATH_MARKS)
    text = _TEX_SPACING.sub("", text).replace(r"\%", "%")
    text = _TEX_COMMAND.sub(lambda found: _TEX_COMMANDS.get(found[1], found[0]), text)
    text = _DEGREE_POWER.sub("°", "".join(text.split()))
    text = _MIXED_NUMBER.sub(r"(\1+\2/\3)", text)
    text = _without_groups(_ROOT_ATOM.sub(r"√(\1)", text))
    return text.lower().rstrip(".;,。")


# Marks stripped from the end of a number: degrees, percent, and common units of length, area,
# volume, time, mass and money, in English and in the Chinese some references use.
_UNIT_NAMES = ("°", "度", "%", "km/h", "m/s", "cm/s", "cm", "mm", "dm", "km", "m", "meters?",
               "metres?", "centimeters?", "millimeters?", "kilometers?", "inch(?:es)?", "in", "ft",
               "feet", "foot", "yards?", "miles?", "degrees?", "seconds?", "sec", "s", "minutes?",
               "mins?", "hours?", "hrs?", "h", "days?", "years?", "kg", "g", "grams?",
               "kilograms?", "yuan", "dollars?", "cents?", "units?", "米", "厘米", "分米", "毫米",
               "千米", "公里", "平方米", "平方厘米", "立方米", "海里", "小时", "分钟", "秒", "天",
               "克", "千克", "元", "个", "o")  # 'o' is how 35º reads once normalized
_UNIT_TAIL = re.compile(r"(?<=[\d)π}])\(?(?:square|cubic|sq\.?)?(?:" + "|".join(_UNIT_NAMES)
                        + r")(?:\^?[23]\^?)?\)?$")
_THOUSANDS = re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?![\d,])")
_CURRENCY = str.maketrans(dict.fromkeys("¥€£"))
# A left side that names its value (x, ∠AOD, cosA) or computes it (7*5). Anything else is an
# equation to state, whose right side is no answer: '(x-2)^2+(y+3)^2=13' does not mean 13.
_VALUE_NAME = re.compile(r"[∠△]?[a-z]{1,8}\d?|[\d.+\-*/()°π√]+")
_LEADING_ZEROS = re.compile(r"(?<![\d.])0+(?=\d)")
_PLAIN_NUMBER = re.compile(r"-?(?:\d+(?:\.\d+)?|\.\d+)")
_MULTIPLIED = re.compile(r"(?<=[\d)i])(?=pi|sqrt|\()|(?<=\))(?=\d)")
_ARITHMETIC = re.compile(r"[\d.+\-*/()pisqrt]+")
_MAX_EXPRESSION = 100  # longer than any numeric answer the sources hold
_WORD = re.compile(r"[^\W\d_]{2,}")
_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow}


def _without_unit(form: str) -> str:
    """The value part of a math form: what 'x=5' names, no unit, no separators."""
    name, equals, value = form.rpartition("=")
    if equals and _VALUE_NAME.fullmatch(name):
        form = value
    form = _UNIT_TAIL.sub("", form.translate(_CURRENCY))
    return _THOUSANDS.sub(lambda found: found[0].replace(",", ""), form)


def plain_number(text: str) -> str | None:
    """The number a final answer writes, marks removed, when it is one plain number: '145' from
    '145°' or '$$145^{\\circ}$$'. Fractions and roots do not count."""
    value = _without_unit(math_form(text))
    return value if _PLAIN_NUMBER.fullmatch(value) else None


def _arithmetic(node: ast.AST) -> float:
    """The value of a parsed expression; anything but arithmetic on numbers raises ValueError."""
    match node:
        case ast.Constant(value=int() | float() as value):
            return float(value)
        case ast.Name(id="pi"):
            return math.pi
        case ast.Call(func=ast.Name(id="sqrt"), args=[argument], keywords=[]):
            return math.sqrt(_arithmetic(argument))
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -_arithmetic(operand)
        case ast.UnaryOp(op=ast.UAdd(), operand=operand):
            return _arithmetic(operand)
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _OPERATORS:
            return _OPERATORS[type(op)](_arithmetic(left), _arithmetic(right))
    raise ValueError("not arithmetic")


def math_value(form: str) -> float | None:
    """The number a math form evaluates to: 2√(3), 8/3π, 20(3+√(3)); None when it has symbols."""
    expression = _without_unit(form).replace("π", "pi").replace("√", "sqrt").replace("^", "**")
    expression = _MULTIPLIED.sub("*", _LEADING_ZEROS.sub("", expression))
    if len(expression) > _MAX_EXPRESSION or not _ARITHMETIC.fullmatch(expression):
        return None
    try:
        value = _arithmetic(ast.parse(expression, mode="eval").body)
    except (ValueError, SyntaxError, TypeError, ArithmeticError, RecursionError):
        return None
    return value if isinstance(value, float) and math.isfinite(value) else None


def _same_math(prediction: str, reference: str) -> bool:
    """Equal as spelled in one form, equal in value, or equal as VQA-normalized words. Words need a
    word: normalization drops articles, which would read the expression 'a-c' as the letter c."""
    predicted, expected = math_form(prediction), math_form(reference)
    if not predicted or not expected:
        return False
    if predicted == expected:
        return True
    values = math_value(predicted), math_value(expected)
    if None not in values:
        return math.isclose(*values, rel_tol=1e-6, abs_tol=1e-12)
    has_word = _WORD.search(prediction) or _WORD.search(reference)
    return bool(has_word) and normalize_answer(prediction) == normalize_answer(reference)


def math_answer_accuracy(prediction: str, answers: list[str]) -> float:
    """Project rule for the math sources, 0 or 1. Numbers compare by value after degree, percent
    and unit marks are stripped, so '10 m' passes for '10cm'. A row with options reads a bare letter
    as that letter, then an option's text, so 'A cyclin ...' is not the letter A, then a leading
    letter."""
    if not is_choice(answers):
        return float(any(_same_math(prediction, answer) for answer in answers))
    options = options_of(answers)
    letter = _LEADING_LETTER.match(prediction)
    by_letter = float(bool(letter) and options.get(letter.group(1).upper()) == answers[1])
    if letter and not prediction[letter.end():].strip():
        return by_letter
    named = [text for text in options.values() if _same_math(prediction, text)]
    return float(answers[1] in named) if named else by_letter
