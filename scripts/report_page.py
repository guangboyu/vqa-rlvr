"""Build the report page (docs/report/index.html) from the saved results.

    python -m scripts.report_page

Every number comes from the evaluation records and result files under results/. Paired
differences are recomputed from the per-question predictions (10,000 bootstrap resamples)
when they sit beside the records; the predictions stay local, so each difference is also
kept in results/report-deltas.json and read from there in a clean clone. The prose lives in
scripts/report_page.template.html with {{key}} slots filled from the values below.

The same values go to results/numbers.json, the one source of every number quoted outside a
generated table. In the markdown pages a quoted number is written <!--n:key-->80.1<!--/n-->;
this script refreshes those and a test fails when one drifts.
"""

import datetime
import html
import json
import math
import re
import sys
from functools import cache
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from vqar.config import MAX_PIXELS  # noqa: E402
from vqar.eval.paired import paired_bootstrap, question_scores  # noqa: E402
from vqar.eval.report import Cell, load_cells  # noqa: E402
from vqar.registry import DATASETS  # noqa: E402

RESULTS = REPO / "results"
CELLS = load_cells(RESULTS / "records", REPO / "data")
DELTAS_PATH = RESULTS / "report-deltas.json"
DELTAS = json.loads(DELTAS_PATH.read_text()) if DELTAS_PATH.exists() else {}
NUMBERS_PATH = RESULTS / "numbers.json"
QUOTED_NUMBER = re.compile(r"<!--n:(?P<key>[\w.]+)-->.*?<!--/n-->")
METHODS = ("base", "sft", "grpo", "sftgrpo")
METHOD_NAMES = {"base": "Instruct", "sft": "SFT", "grpo": "GRPO", "sftgrpo": "SFT → GRPO"}


def load(name):
    return json.loads((RESULTS / name).read_text())


def record(arm, dataset, prompt="short", image_mode="original", max_pixels=MAX_PIXELS):
    key = Cell(arm, dataset, "test", "full", prompt, image_mode, None, 0.0, 1, max_pixels)
    found = CELLS.get(key)
    if found is None:
        raise KeyError(f"no record for {key}")
    return found


def acc(arm, dataset, prompt="short", variant="official", image_mode="original",
        max_pixels=MAX_PIXELS):
    return 100 * record(arm, dataset, prompt, image_mode, max_pixels).metrics[variant]["overall"]


def tokens(arm, dataset, prompt="reasoning"):
    return record(arm, dataset, prompt).metrics["mean_output_tokens"]


@cache
def delta(reference, candidate, dataset, prompt="short", variant="official",
          modes=("original", "original")):
    """candidate − reference with a paired 95% bootstrap interval over the test questions;
    `modes` gives each side's image mode. Recomputed from the predictions when present, else
    read from the saved differences."""
    key = "|".join((reference, candidate, dataset, prompt, variant, *modes))
    paths = [record(arm, dataset, prompt, mode).predictions_path()
             for arm, mode in zip((reference, candidate), modes, strict=True)]
    if all(path.exists() for path in paths):
        DELTAS[key] = paired_bootstrap(*(question_scores(path, variant) for path in paths))
    if key not in DELTAS:
        raise SystemExit(f"no predictions and no saved difference for {key}")
    return DELTAS[key]


def f1(x):
    return f"{x:.1f}"


def signed(x, digits=1):
    return f"{x:+.{digits}f}".replace("-", "−")


def ci(d, digits=1):
    return (f'{signed(d["delta"], digits)} <span class="ci">[{signed(d["low"], digits)}, '
            f'{signed(d["high"], digits)}]</span>')


def esc(text):
    return html.escape(str(text), quote=True)


# ---------------------------------------------------------------- numbers for the prose

N = {}


def put(key, value):
    N[key] = value


# ChartQA, 2B, 1,250 human-written test questions.
CQ = {"base": "base2", "sft": "cqa-sft2", "grpo": "cqa-grpo2-base-e1",
      "short": "cqa-short2-base-e1", "sftgrpo": "cqa-grpo2-sft-e1"}
judge = {(c["arm"], c["prompt"]): c for c in load("judge-chartqa.json")["cells"]}
for name, arm in CQ.items():
    put(f"cq.{name}.r", f1(acc(arm, "chartqa", "reasoning")))
    put(f"cq.{name}.s", f1(acc(arm, "chartqa", "short")))
    put(f"cq.{name}.tok", f"{tokens(arm, 'chartqa'):.0f}")
    put(f"cq.{name}.aud", f1(judge[(arm, "reasoning")]["audited"]))
d = delta(CQ["base"], CQ["grpo"], "chartqa", "reasoning")
put("cq.gain", f1(d["delta"]))
put("cq.gain.ci", ci(d))
aud = next(p for p in load("judge-chartqa.json")["paired_deltas"]
           if p["reference"] == "base2" and p["candidate"] == CQ["grpo"]
           and p["prompt"] == "reasoning" and p["scorer"] == "audited")
put("cq.audgain", f1(aud["delta"]))
put("cq.audgain.ci", ci(aud))
best_short = max(acc(arm, "chartqa", "short") for arm in CQ.values())
put("cq.bestshort", f1(best_short))
put("cq.overshort", f1(acc(CQ["grpo"], "chartqa", "reasoning") - best_short))
best_sft = max(acc(CQ["sft"], "chartqa", prompt) for prompt in ("short", "reasoning"))
put("cq.oversft", f1(acc(CQ["grpo"], "chartqa", "reasoning") - best_sft))
best_short_aud = max(judge[(arm, "short")]["audited"] for arm in CQ.values())
put("cq.overshort.aud", f1(judge[(CQ["grpo"], "reasoning")]["audited"] - best_short_aud))
# SFT on reasoning traces: correct samples from Instruct 2B and from Instruct 8B.
for name, arm in (("self", "card-reasoning-sft-self"), ("8b", "card-reasoning-sft-8b")):
    put(f"cq.traces.{name}", f1(acc(arm, "chartqa", "reasoning")))
    lead = delta(arm, CQ["grpo"], "chartqa", "reasoning")
    put(f"cq.traces.{name}.lead", f1(lead["delta"]))
    put(f"cq.traces.{name}.lead.ci", ci(lead))
d = delta(CQ["sft"], CQ["sftgrpo"], "chartqa", "reasoning")
put("cq.sftgrpo.gain", f1(d["delta"]))

passk = {(r["model"], r["prompt"]): r["pass_at_k"] for r in load("passk-chartqa.json")["records"]}
for name, arm, prompt in (("base.r", "base2", "reasoning"), ("grpo", CQ["grpo"], "reasoning"),
                          ("sftgrpo", CQ["sftgrpo"], "reasoning"), ("base.s", "base2", "short"),
                          ("short", CQ["short"], "short"), ("sft", "cqa-sft2", "short")):
    put(f"pk.{name}.1", f1(passk[(arm, prompt)]["1"]))
    put(f"pk.{name}.32", f1(passk[(arm, prompt)]["32"]))

# Photo questions, unseen sets.
PH = {"base": "base2", "sft": "sft2", "grpo": "p1-f-short2-base", "sftgrpo": "p1-f-short2-sft",
      "sft4k": "sft4k2", "grpor": "p1-f-grpo2-base", "pool": "p1-fp-short2-base"}
for name in ("base", "sft", "grpo", "sftgrpo", "sft4k"):
    put(f"tv.{name}", f"{acc(PH[name], 'textvqa'):.1f}")
ood = {c["arm"]: c["score"] for c in load("ood.json")["cells"] if c["dataset"] == "vizwiz"}
for name in ("base", "sft", "grpo", "sftgrpo", "sft4k"):
    put(f"vw.{name}", f1(ood[PH[name]]))
put("tv.rl_sft", ci(delta(PH["sft"], PH["grpo"], "textvqa")))
put("tv.rl_sft4k", ci(delta(PH["sft4k"], PH["grpo"], "textvqa")))
put("tv.rl_sft.v", signed(delta(PH["sft"], PH["grpo"], "textvqa")["delta"]))
vw = {(c["reference"], c["candidate"]): c for c in load("ood.json")["comparisons"]
      if c["dataset"] == "vizwiz"}
put("vw.rl_sft", ci(vw[(PH["sft"], PH["grpo"])]))
put("vw.rl_sft.v", signed(vw[(PH["sft"], PH["grpo"])]["delta"]))
vw_rl_4k = vw[(PH["grpo"], PH["sft4k"])]
put("vw.rl_sft4k", ci({k: -vw_rl_4k[k] for k in ("delta", "high", "low")}
                      | {"low": -vw_rl_4k["high"], "high": -vw_rl_4k["low"]}))

for ds in ("vqav2", "gqa", "textvqa"):
    put(f"ph.{ds}.base.s", f1(acc("base2", ds)))
    put(f"ph.{ds}.base.r", f1(acc("base2", ds, "reasoning")))
    put(f"ph.{ds}.base.tok", f"{tokens('base2', ds):.0f}")
    put(f"ph.{ds}.sft.s", f1(acc("sft2", ds)))
    put(f"ph.{ds}.grpo.s", f1(acc(PH["grpo"], ds)))
    put(f"ph.{ds}.pool.s", f1(acc(PH["pool"], ds)))
    put(f"ph.{ds}.grpor.r", f1(acc(PH["grpor"], ds, "reasoning")))
    put(f"ph.{ds}.grpor.tok", f"{tokens(PH['grpor'], ds):.0f}")
    put(f"ph.{ds}.cq.r", f1(acc(CQ["grpo"], ds, "reasoning")))
    put(f"ph.{ds}.cq.s", f1(acc(CQ["grpo"], ds)))
    put(f"ph.{ds}.cq.tok", f"{tokens(CQ['grpo'], ds):.0f}")
put("ph.cq.vqa.r", f1(acc(PH["grpor"], "chartqa", "reasoning")))
put("ph.cq.vqa.tok", f"{tokens(PH['grpor'], 'chartqa'):.0f}")
put("ph.cq.mix.r", f1(acc("mix-grpo2-base", "chartqa", "reasoning")))
put("ph.cq.mix.tok", f"{tokens('mix-grpo2-base', 'chartqa'):.0f}")
judge_photo = {(c["arm"], c["dataset"], c["prompt"]): c for c in load("judge-final.json")["cells"]}
photo_gain = {p["scorer"]: p for p in load("judge-final.json")["paired_deltas"]
              if (p["reference"], p["candidate"], p["dataset"], p["prompt"])
              == ("base2", PH["grpor"], "vqav2", "reasoning")}
put("ph.vqav2.gain", f1(photo_gain["official"]["delta"]))
put("ph.vqav2.audgain", f1(photo_gain["audited"]["delta"]))
put("ph.vqav2.base.r.aud", f1(judge_photo[("base2", "vqav2", "reasoning")]["audited"]))
put("ph.vqav2.grpor.r.aud", f1(judge_photo[(PH["grpor"], "vqav2", "reasoning")]["audited"]))
put("ph.vqav2.base.s.aud", f1(judge_photo[("base2", "vqav2", "short")]["audited"]))
heldout = {(r["model"], r["prompt"], r["dataset"]): r["pass_at_k"]
           for r in load("passk-heldout.json")["records"]}
for name, arm in (("base", "base2"), ("grpo", PH["grpo"]), ("sft", "sft2"), ("pool", PH["pool"])):
    for ds in ("vqav2", "gqa"):
        put(f"hk.{name}.{ds}.1", f1(heldout[(arm, "short", ds)]["1"]))
        put(f"hk.{name}.{ds}.32", f1(heldout[(arm, "short", ds)]["32"]))

# Medical: SLAKE and PathVQA official, OmniMedVQA normalized (its official scorer counts only
# option letters).
MED = [("SLAKE, 2B", "slake", "official", "base2",
        {"sft": "slk-sft2", "grpo": "slk-short2-base-e1", "sftgrpo": "slk-short2-sft-e1"}),
       ("SLAKE, 8B", "slake", "official", "base8",
        {"sft": "slk-sft8", "grpo": "slk-short8-base-e1", "sftgrpo": "slk-short8-sft-e1"}),
       ("OmniMedVQA, held-out collections, 2B", "omnimedx", "normalized", "base2",
        {"sft": "omx-sft2", "grpo": "omx-short2-base", "sftgrpo": "omx-short2-sft"})]
for _label, ds, variant, base, arms in MED:
    key = {"slake": "slk2" if base == "base2" else "slk8", "omnimedx": "omx"}[ds]
    put(f"med.{key}.base", f1(acc(base, ds, variant=variant)))
    for name, arm in arms.items():
        put(f"med.{key}.{name}", f1(acc(arm, ds, variant=variant)))
        put(f"med.{key}.{name}.path", f1(acc(arm, "pathvqa")))
    put(f"med.{key}.base.path", f1(acc(base, "pathvqa")))
    d = delta(arms["sft"], arms["sftgrpo"], ds, variant=variant)
    put(f"med.{key}.sftgrpo_sft", ci(d))
    put(f"med.{key}.sftgrpo_sft.v", f1(d["delta"]))
    put(f"med.{key}.grpo_sft.path", ci(delta(arms["sft"], arms["grpo"], "pathvqa")))
put("med.omx.sftm", f1(acc("omx-sft2-matched", "omnimedx", variant="normalized")))
put("med.omx.sftm.path", f1(acc("omx-sft2-matched", "pathvqa")))

# Beyond accuracy.
beyond = load("beyond-accuracy.json")["sets"]
BY = {}
for spec in beyond:
    if "group" in spec:
        continue
    for row in spec["rows"]:
        if not row.get("missing"):
            BY[(spec["dataset"], spec["prompt"], spec["base"], row["model"])] = row


def by(dataset, base, model, prompt="short"):
    return BY[(dataset, prompt, base, model)]


put("ba.slk8.base.black", f1(by("slake", "base8", "base8")["black_accuracy"]))
put("ba.slk8.sft.black", f1(by("slake", "base8", "slk-sft8")["black_accuracy"]))
put("ba.slk8.grpo.black", f1(by("slake", "base8", "slk-short8-base-e1")["black_accuracy"]))
BEYOND_SETTINGS = (("slk2", "slake", "base2", "slk-sft2", "slk-short2-base-e1"),
                   ("slk8", "slake", "base8", "slk-sft8", "slk-short8-base-e1"),
                   ("omx", "omnimedx", "base2", "omx-sft2-matched", "omx-short2-base"),
                   ("vqa", "vqav2", "base2", "sft4k2", "p1-f-short2-base"),
                   ("cqa", "chartqa", "base2", "cqa-sft2", "cqa-short2-base-e1"))
for key, ds, base, sft, grpo in BEYOND_SETTINGS:
    put(f"kl.{key}.sft", f"{by(ds, base, sft)['kl']['kl']:.2f}")
    put(f"kl.{key}.grpo", f"{by(ds, base, grpo)['kl']['kl']:.2f}")
    put(f"cal.{key}.sft", f1(by(ds, base, sft)["calibration"]["calibration_error"]))
    put(f"cal.{key}.grpo", f1(by(ds, base, grpo)["calibration"]["calibration_error"]))
    put(f"cal.{key}.grpo.conf", f1(by(ds, base, grpo)["calibration"]["mean_confidence"]))
    put(f"cal.{key}.grpo.acc", f1(by(ds, base, grpo)["calibration"]["majority_accuracy"]))
put("cal.slk2.base", f1(by("slake", "base2", "base2")["calibration"]["calibration_error"]))


# Calibration error over the short-answer settings, each method's every arm.
SHORT_CALIBRATION = {
    "sft": (("slake", "base2", "slk-sft2"), ("slake", "base8", "slk-sft8"),
            ("omnimedx", "base2", "omx-sft2"), ("omnimedx", "base2", "omx-sft2-matched"),
            ("vqav2", "base2", "sft2"), ("vqav2", "base2", "sft4k2"),
            ("chartqa", "base2", "cqa-sft2")),
    "grpo": (("slake", "base2", "slk-short2-base-e1"), ("slake", "base8", "slk-short8-base-e1"),
             ("omnimedx", "base2", "omx-short2-base"), ("vqav2", "base2", "p1-f-short2-base"),
             ("chartqa", "base2", "cqa-short2-base-e1"))}
for name, settings in SHORT_CALIBRATION.items():
    errors = [by(*setting)["calibration"]["calibration_error"] for setting in settings]
    put(f"cal.{name}.lo", f1(min(errors)))
    put(f"cal.{name}.hi", f1(max(errors)))

# What survives a black image, at 8B on SLAKE: the two methods on the same training questions.
slk8_base = by("slake", "base8", "base8")["accuracy"]
for name, arm in (("sft", "slk-sft8"), ("grpo", "slk-short8-base-e1")):
    row = by("slake", "base8", arm)
    put(f"ba.slk8.{name}.gain", f1(row["accuracy"] - slk8_base))
    put(f"ba.slk8.{name}.blackgain", f1(row["black_gain"]["delta"]))
for key, ds, base, arm in (("slk2", "slake", "base2", "slk-short2-sft-e1"),
                           ("slk8", "slake", "base8", "slk-short8-sft-e1"),
                           ("omx", "omnimedx", "base2", "omx-short2-sft")):
    put(f"ba.{key}.sftgrpo.blackgain", signed(by(ds, base, arm)["black_gain"]["delta"]))
put("ba.omx.sft.blackgain", signed(by("omnimedx", "base2", "omx-sft2")["black_gain"]["delta"]))
put("ba.omx.sftm.blackgain",
    signed(by("omnimedx", "base2", "omx-sft2-matched")["black_gain"]["delta"]))
RL_BLIND = [by(ds, base, arm, prompt)["black_gain"] for ds, base, arm, prompt in (
    ("slake", "base2", "slk-short2-base-e1", "short"),
    ("slake", "base8", "slk-short8-base-e1", "short"),
    ("omnimedx", "base2", "omx-short2-base", "short"),
    ("vqav2", "base2", "p1-f-short2-base", "short"),
    ("chartqa", "base2", "cqa-short2-base-e1", "short"),
    ("chartqa", "base2", "cqa-grpo2-base-e1", "reasoning"))]
put("ba.rl.settings", str(len(RL_BLIND)))
put("ba.rl.level", str(sum(d["low"] <= 0 <= d["high"] for d in RL_BLIND)))
put("ba.rl.most", signed(max(d["delta"] for d in RL_BLIND)))
medical_kl = [float(N[f"kl.{key}.sft"]) / float(N[f"kl.{key}.grpo"])
              for key in ("slk2", "slk8", "omx")]
put("kl.med.lo", str(round(min(medical_kl))))
put("kl.med.hi", str(round(max(medical_kl))))

# Reach: Instruct's pass@8 on open-answer questions against each method's greedy end point.
reach = load("reach.json")
for row in reach["rows"]:
    if row["dataset"] == "slake":
        key = "slk2" if "2B" in row["label"] else "slk8"
        put(f"reach.{key}.reach", f1(row["reach"]))
        put(f"reach.{key}.rl", f1(row["greedy"]["rl"]))
        put(f"reach.{key}.sft", f1(row["greedy"]["sft"]))
        put(f"reach.{key}.sft.over", f1(row["sft_minus_reach"]["delta"]))
    if (row["dataset"], row["prompt"]) == ("chartqa", "reasoning"):
        put("reach.cq.reach", f1(row["reach"]))
        put("reach.cq.instruct", f1(row["greedy"]["instruct"]))
        put("reach.cq.rl", f1(row["greedy"]["rl"]))
        put("reach.cq.sft", f1(row["greedy"]["sft"]))
    if (row["dataset"], row["prompt"]) == ("gqa", "short"):
        put("reach.gqa.reach", f1(row["reach"]))
        put("reach.gqa.rl", f1(row["greedy"]["rl"]))
# The chart-only control at the mixed model's length (results/ledger.json, dev rows).
control = next(card for card in load("ledger.json")["cards"]
               if card["id"] == "chart-length-control")["final"]
put("ctrl.acc", f1(control["accuracy"]))
put("ctrl.base.acc", f1(control["baseline"]))
put("ctrl.tok", f"{control['tokens']:.0f}")
put("reach.sets", str(len(reach["rows"])))

# The category table: where Instruct's reasoning beats its short answers.
table = load("categories.json")
payoff = {row["dataset"]: row["cells"]["base2|reasoning"]["score"]
          - row["cells"]["base2|short"]["score"] for row in table["rows"]}
put("cat.benchmarks", str(len(table["rows"])))
put("cat.reasoning.pays", str(sum(value > 0.5 for value in payoff.values())))
put("cat.payoff.mathvista", signed(payoff["mathvista"]))
put("cat.payoff.chartqa", signed(payoff["chartqa"]))
put("cat.payoff.vqav2", signed(payoff["vqav2"]))
put("reach.rl.most", f1(max(reach["verdict"]["rl_above_reach"].values())))
vision = {row["model"]: row for spec in beyond if spec.get("group") == "vision"
          for row in spec["rows"] if not row.get("missing")}
put("vis.papo.black", f1(vision["slk-short2-base-e1-papo"]["black_accuracy"]))
put("vis.papo.kl", f1(vision["slk-short2-base-e1-papo"]["kl"]["kl"]))
for name, arm in (("vis", "slk-sft2-vis"), ("upsft", "slk-sft2-up"),
                  ("upgrpo", "slk-short2-base-e1-up"), ("papo", "slk-short2-base-e1-papo")):
    put(f"vis.{name}", f1(acc(arm, "slake")))

# Training-setting screen on ChartQA.
knobs = load("knobs-chartqa.json")
spread = [k["reasoning"]["delta"] for k in knobs["knobs"] if "reasoning" in k]
put("knob.count", str(len(spread) - 1))
put("knob.lo", signed(min(spread)))
put("knob.hi", signed(max(spread)))

# Harness counts.
# Published records only: in-progress batches and licence-restricted datasets stay local.
record_files = [p for p in (RESULTS / "records").glob("*/*.json")
                if DATASETS[p.name.split("-")[0]].public_records]
put("h.records", f"{len(record_files):,}")
test_sets = ["vqav2", "gqa", "clevr", "textvqa", "chartqa", "slake", "omnimedx", "pathvqa",
             "vizwiz"]
VIZWIZ_QUESTIONS = 8000  # scored by the authors' evaluator outside the record cells (ood.json)
N_questions = sum(record("base2", ds).metrics["n"] if ds != "vizwiz" else VIZWIZ_QUESTIONS
                  for ds in test_sets)
put("h.questions", f"{N_questions:,}")
put("h.sets", str(len(test_sets)))
put("h.pathvqa", f"{record('base2', 'pathvqa').metrics['n']:,}")
# Measured once, outside the records: docs/harness/validation.md, results/judge-*.json.
api_spend = (load("judge-chartqa.json")["spend_usd"]  # the judge's running total
             + load("judge-audit.json")["cost_usd"] + load("judge-audit-chartqa.json")["cost_usd"])
put("h.api", f"${api_spend:.2f}")
put("h.parity", "626,074")
put("h.misses", "54,060")
put("h.audited", "800")
put("h.sampler.merged", "0.132")
put("h.sampler.unmerged", "0.008")
# Updates of the chart-only and the photos + charts runs (experiments.toml, run configs).
put("cq.updates", "1,844")
put("mix.updates", "2,844")
put("date", datetime.date.today().strftime("%-d %B %Y"))


# Shortcut audit of 3D CT benchmarks.
PUBLISHED = {"deeptumor_mc": 64.6, "deeptumor_ff": 63.4, "m3d": 75.78}  # papers' fine-tuned
prior = {name: load(f"screening/{name}-option-prior.json")["benchmark"]["all"]
         for name in ("deeptumor", "deeptumorx", "m3dvqa", "deeptumorff")}
put("au.mc.rule", f1(prior["deeptumor"]["prior_accuracy"]))
put("au.mc.chance", f1(prior["deeptumor"]["chance"]))
put("au.mc.sft", f1(acc("dtv-probe-sft2", "deeptumor", variant="normalized")))
put("au.mc.blind", f1(acc("dtv-probe-sft2", "deeptumor", variant="normalized",
                          image_mode="black")))
put("au.mc.base", f1(acc("base2", "deeptumor", variant="normalized")))
put("au.ff.rule", f1(prior["deeptumorff"]["prior_score"]))
put("au.ff.base", f1(acc("base2", "deeptumorff")))
put("au.ff.sft", f1(acc("dtff-sft2", "deeptumorff")))
put("au.ff.blind", f1(acc("dtff-blind-sft2", "deeptumorff", image_mode="black")))
gap = delta("dtff-blind-sft2", "dtff-sft2", "deeptumorff", modes=("black", "original"))
put("au.ff.gap", ci(gap))
put("au.ff.gap.v", f1(gap["delta"]))
put("au.m3d.rule", f1(prior["m3dvqa"]["letter_prior_accuracy"]))
put("au.m3d.pub", f1(PUBLISHED["m3d"]))
put("au.m3d.base", f1(acc("base2", "m3dvqa")))
m3d_types = record("m3d-blind-sft2", "m3dvqa").metrics["official"]["by_answer_type"]
put("au.m3d.blind", f1(acc("m3d-blind-sft2", "m3dvqa")))
put("au.m3d.blind.mean", f1(100 * sum(m3d_types.values()) / len(m3d_types)))
for name, value in m3d_types.items():
    put(f"au.m3d.{name}", f1(100 * value))
put("au.x.rule", f1(prior["deeptumorx"]["prior_accuracy"]))
put("au.x.chance", f1(prior["deeptumorx"]["chance"]))
put("au.x.base", f1(acc("base2", "deeptumorx", variant="normalized")))
put("au.x.old", f1(acc("dtv-probe-sft2", "deeptumorx")))
X = {"sft": "dtvx-probe-sft2", "grpo": "dtvx-grpo2-base", "sftgrpo": "dtvx-grpo2-sft"}
for name, arm in X.items():
    put(f"au.x.{name}", f1(acc(arm, "deeptumorx")))
    put(f"au.x.{name}.black", f1(acc(arm, "deeptumorx", image_mode="black")))
    types = record(arm, "deeptumorx").metrics["official"]["by_answer_type"]
    black_types = record(arm, "deeptumorx", image_mode="black").metrics["official"][
        "by_answer_type"]
    put(f"au.x.{name}.recog", f1(100 * types["recognition"]))
    put(f"au.x.{name}.recog.black", f1(100 * black_types["recognition"]))
    put(f"au.x.{name}.meas", f1(100 * types["measurement"]))
GRID = 2_097_152
put("au.v.sft", f1(acc("dtvv-probe-sft2", "deeptumorv", max_pixels=GRID)))
put("au.v.sft.black", f1(acc("dtvv-probe-sft2", "deeptumorv", image_mode="black",
                             max_pixels=GRID)))
put("au.x.grpo_sft", ci(delta(X["sft"], X["grpo"], "deeptumorx")))
put("au.x.sftgrpo_sft", ci(delta(X["sft"], X["sftgrpo"], "deeptumorx")))


# ---------------------------------------------------------------- figures (inline SVG)

def tip(text):
    return f'data-tip="{esc(text)}"'


def marker(cx, cy, method, r=5.5, title=""):
    """Instruct is a hollow ring; trained models are filled dots in their method color."""
    if method == "base":
        return (f'<circle class="mk base" cx="{cx:.1f}" cy="{cy:.1f}" r="{r - 1:.1f}" '
                f'{tip(title)}/>')
    return f'<circle class="mk {method}" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" {tip(title)}/>'


def strip(values, lo, hi, width=190, height=26):
    """One row of the overview: the four models on the row's own scale."""
    pad = 10
    def x(v):
        return pad + (v - lo) / (hi - lo) * (width - 2 * pad)
    parts = [f'<line class="axis" x1="{pad}" y1="{height / 2}" x2="{width - pad}" '
             f'y2="{height / 2}"/>']
    for method in METHODS:
        v = values[method]
        parts.append(marker(x(v), height / 2, method, 5, f"{METHOD_NAMES[method]}: {v:.1f}%"))
    return (f'<svg class="strip" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}" role="img" aria-label="positions of the four models">'
            + "".join(parts) + "</svg>")


def axis_ticks(x, lo, hi, step, y0, y1, fmt="{:g}"):
    out = []
    v = lo
    while v <= hi + 1e-9:
        xx = x(v)
        out.append(f'<line class="grid" x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y1}"/>'
                   f'<text class="tick" x="{xx:.1f}" y="{y1 + 16}" text-anchor="middle">'
                   f'{fmt.format(v)}</text>')
        v += step
    return "".join(out)


def chartqa_figure():
    rows = [("Instruct", "base", "base2"), ("SFT", "sft", "cqa-sft2"),
            ("GRPO, short-answer rollouts", "grpo", "cqa-short2-base-e1"),
            ("GRPO, reasoning rollouts", "grpo", "cqa-grpo2-base-e1"),
            ("SFT → GRPO, reasoning rollouts", "sftgrpo", "cqa-grpo2-sft-e1")]
    width, left, right, top, row_h = 720, 214, 70, 14, 40
    lo, hi = 60, 85
    plot_w = width - left - right
    def x(v):
        return left + (v - lo) / (hi - lo) * plot_w
    height = top + row_h * len(rows) + 44
    parts = [axis_ticks(x, lo, hi, 5, top - 4, top + row_h * len(rows))]
    for i, (name, method, arm) in enumerate(rows):
        y = top + row_h * i + row_h / 2
        off = acc(arm, "chartqa", "reasoning")
        audited = judge[(arm, "reasoning")]["audited"]
        tok = tokens(arm, "chartqa")
        parts.append(f'<text class="rowlabel" x="{left - 12}" y="{y + 4}" '
                     f'text-anchor="end">{esc(name)}</text>')
        parts.append(f'<line class="link" x1="{x(off):.1f}" y1="{y}" x2="{x(audited):.1f}" '
                     f'y2="{y}"/>')
        dx, dy = x(audited), y
        parts.append(f'<path class="mk-aud {method}" d="M{dx:.1f},{dy - 6} L{dx + 6:.1f},{dy} '
                     f'L{dx:.1f},{dy + 6} L{dx - 6:.1f},{dy} Z" '
                     f'{tip(f"{name}: audited judge {audited:.1f}%")}/>')
        parts.append(marker(x(off), y, method, 6, f"{name}: official {off:.1f}%"))
        parts.append(f'<text class="val" x="{x(off) - 10:.1f}" y="{y + 4}" '
                     f'text-anchor="end">{off:.1f}</text>')
        parts.append(f'<text class="val muted" x="{dx + 10:.1f}" y="{y + 4}">{audited:.1f}</text>')
        parts.append(f'<text class="tok" x="{width - 4}" y="{y + 4}" '
                     f'text-anchor="end">{tok:.0f} tok</text>')
    parts.append(f'<text class="axislabel" x="{left + plot_w / 2}" y="{height - 2}" '
                 f'text-anchor="middle">Accuracy, reasoning prompt (%)</text>')
    return (f'<svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="ChartQA accuracy by model, official and audited">' + "".join(parts)
            + "</svg>")


def passk_panel(title, series, lo, hi):
    width, height, left, right, top, bottom = 340, 230, 40, 92, 26, 34
    ks = [1, 2, 4, 8, 16, 32]
    plot_w, plot_h = width - left - right, height - top - bottom
    def x(k):
        return left + math.log2(k) / 5 * plot_w
    def y(v):
        return top + (hi - v) / (hi - lo) * plot_h
    parts = [f'<text class="paneltitle" x="{left}" y="14">{esc(title)}</text>']
    v = lo
    while v <= hi + 1e-9:
        parts.append(f'<line class="grid" x1="{left}" y1="{y(v):.1f}" x2="{left + plot_w}" '
                     f'y2="{y(v):.1f}"/><text class="tick" x="{left - 6}" y="{y(v) + 4:.1f}" '
                     f'text-anchor="end">{v:g}</text>')
        v += 10
    for k in ks:
        parts.append(f'<text class="tick" x="{x(k):.1f}" y="{top + plot_h + 16}" '
                     f'text-anchor="middle">{k}</text>')
    parts.append(f'<text class="axislabel" x="{left + plot_w / 2}" y="{height - 2}" '
                 f'text-anchor="middle">samples per question, k</text>')
    ends = []
    for name, method, values in series:
        pts = " ".join(f"{x(k):.1f},{y(values[str(k)]):.1f}" for k in ks)
        parts.append(f'<polyline class="ln {method}" points="{pts}"/>')
        for k in ks:
            parts.append(marker(x(k), y(values[str(k)]), method, 3.5,
                                f"{name}, pass@{k}: {values[str(k)]:.1f}%"))
        ends.append([y(values["32"]), name, method, values["32"]])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 14)
    for yy, name, _method, v32 in ends:
        parts.append(f'<text class="endlabel" x="{left + plot_w + 8}" y="{yy + 4:.1f}">'
                     f'{esc(name)} {v32:.1f}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">'
            + "".join(parts) + "</svg>")


def passk_figure():
    reasoning = passk_panel("Reasoning prompt", [
        ("Instruct", "base", passk[("base2", "reasoning")]),
        ("GRPO", "grpo", passk[(CQ["grpo"], "reasoning")]),
        ("SFT → GRPO", "sftgrpo", passk[(CQ["sftgrpo"], "reasoning")])], 60, 100)
    short = passk_panel("Short-answer prompt", [
        ("Instruct", "base", passk[("base2", "short")]),
        ("SFT", "sft", passk[("cqa-sft2", "short")]),
        ("GRPO", "grpo", passk[(CQ["short"], "short")])], 60, 100)
    return f'<div class="panels">{reasoning}{short}</div>'


def black_figure():
    rows = [("SLAKE, 2B", "slake", "base2", "slk-sft2", "slk-short2-base-e1"),
            ("SLAKE, 8B", "slake", "base8", "slk-sft8", "slk-short8-base-e1"),
            ("OmniMedVQA, held-out", "omnimedx", "base2", "omx-sft2-matched", "omx-short2-base"),
            ("VQAv2", "vqav2", "base2", "sft4k2", "p1-f-short2-base"),
            ("ChartQA", "chartqa", "base2", "cqa-sft2", "cqa-short2-base-e1")]
    width, left, right, top, row_h = 720, 160, 150, 14, 36
    lo, hi = 0, 70
    plot_w = width - left - right
    def x(v):
        return left + (v - lo) / (hi - lo) * plot_w
    height = top + row_h * len(rows) + 44
    parts = [axis_ticks(x, lo, hi, 10, top - 4, top + row_h * len(rows))]
    for i, (name, ds, base, sft, grpo) in enumerate(rows):
        y = top + row_h * i + row_h / 2
        b, s, g = (by(ds, base, m) for m in (base, sft, grpo))
        parts.append(f'<text class="rowlabel" x="{left - 12}" y="{y + 4}" '
                     f'text-anchor="end">{esc(name)}</text>')
        xs = [x(r["black_accuracy"]) for r in (b, s, g)]
        parts.append(f'<line class="link" x1="{min(xs):.1f}" y1="{y}" x2="{max(xs):.1f}" '
                     f'y2="{y}"/>')
        for method, row in (("base", b), ("sft", s), ("grpo", g)):
            parts.append(marker(x(row["black_accuracy"]), y, method, 6,
                                f"{name}, {METHOD_NAMES[method]}: {row['black_accuracy']:.1f}% "
                                f"with a black image"))
        parts.append(f'<text class="val" x="{width - 4}" y="{y + 4}" text-anchor="end">'
                     f'SFT {signed(s["black_gain"]["delta"])} · GRPO '
                     f'{signed(g["black_gain"]["delta"])}</text>')
    parts.append(f'<text class="axislabel" x="{left + plot_w / 2}" y="{height - 2}" '
                 f'text-anchor="middle">Accuracy with a black image (%)</text>')
    return (f'<svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="Accuracy with a black image by model">' + "".join(parts) + "</svg>")


def audit_figure():
    """Per benchmark: the text-only rule, a model given no image, the same recipe with the
    image, and the published fine-tuned model, plotted and listed."""
    rows = [("DeepTumorVQA, multiple choice", "au.mc", PUBLISHED["deeptumor_mc"]),
            ("DeepTumorVQA, free form", "au.ff", PUBLISHED["deeptumor_ff"]),
            ("M3D-VQA, multiple choice", "au.m3d", PUBLISHED["m3d"])]
    columns = (("rule", "Rule"), ("blind", "No image"), ("sighted", "Image"),
               ("published", "Published"))
    width, left, right, top, row_h, col_w = 860, 200, 250, 34, 44, 60
    lo, hi = 40, 80
    plot_w = width - left - right
    def x(v):
        return left + (v - lo) / (hi - lo) * plot_w
    height = top + row_h * len(rows) + 44
    parts = [axis_ticks(x, lo, hi, 10, top - 4, top + row_h * len(rows))]
    for j, (_, title) in enumerate(columns):
        parts.append(f'<text class="tok" x="{width - right + 34 + col_w * (j + 1)}" '
                     f'y="{top - 14}" text-anchor="end">{title}</text>')
    for i, (name, key, published) in enumerate(rows):
        y = top + row_h * i + row_h / 2
        values = {"rule": float(N[f"{key}.rule"]), "blind": float(N[f"{key}.blind"]),
                  "published": published}
        if key != "au.m3d":
            values["sighted"] = float(N[f"{key}.sft"])
        parts.append(f'<text class="rowlabel" x="{left - 12}" y="{y + 4}" '
                     f'text-anchor="end">{esc(name)}</text>')
        xs = [x(v) for v in values.values()]
        parts.append(f'<line class="link" x1="{min(xs):.1f}" y1="{y}" x2="{max(xs):.1f}" '
                     f'y2="{y}"/>')
        v = values["rule"]
        parts.append(f'<rect class="mk-rule" x="{x(v) - 5:.1f}" y="{y - 5}" width="10" '
                     f'height="10" {tip(f"Text-only rule: {v:.1f}%")}/>')
        v = values["blind"]
        parts.append(f'<circle class="mk-blind" cx="{x(v):.1f}" cy="{y}" r="6" '
                     f'{tip(f"Model without the image: {v:.1f}%")}/>')
        if "sighted" in values:
            v = values["sighted"]
            parts.append(f'<circle class="mk-sighted" cx="{x(v):.1f}" cy="{y}" r="6" '
                         f'{tip(f"Same recipe with the image: {v:.1f}%")}/>')
        v = published
        parts.append(f'<path class="mk-pub" d="M{x(v):.1f},{y - 7} L{x(v) + 7:.1f},{y} '
                     f'L{x(v):.1f},{y + 7} L{x(v) - 7:.1f},{y} Z" '
                     f'{tip(f"Published fine-tuned model: {v:.1f}%")}/>')
        for j, (field, _) in enumerate(columns):
            text = f"{values[field]:.1f}" if field in values else "—"
            style = "val" if field == "blind" else "val muted"
            parts.append(f'<text class="{style}" x="{width - right + 34 + col_w * (j + 1)}" '
                         f'y="{y + 4}" text-anchor="end">{text}</text>')
    parts.append(f'<text class="axislabel" x="{left + plot_w / 2}" y="{height - 4}" '
                 f'text-anchor="middle">Accuracy (%)</text>')
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Accuracy without '
            f'the image against published fine-tuned models">' + "".join(parts) + "</svg>")


# ---------------------------------------------------------------- tables

def method_cells(values, best):
    out = []
    for method in METHODS:
        v = values.get(method)
        if v is None:
            out.append('<td class="n">—</td>')
            continue
        if method == best:
            out.append(f'<td class="n best"><b>{v:.1f}</b></td>')
        else:
            out.append(f'<td class="n">{v:.1f}</td>')
    return "".join(out)


def overview_table():
    rows = [
        ("Charts", "ChartQA", "reasoning prompt",
         {m: acc(CQ[m], "chartqa", "reasoning") for m in METHODS}),
        ("Charts", "ChartQA, audited judge", "reasoning prompt",
         {m: judge[(CQ[m], "reasoning")]["audited"] for m in METHODS}),
    ]
    for label, ds, variant, base, arms in MED:
        rows.append(("Medical", label, "short answer" + (", normalized" if variant != "official"
                                                          else ""),
                     {"base": acc(base, ds, variant=variant),
                      **{m: acc(a, ds, variant=variant) for m, a in arms.items()}}))
    rows.append(("Unseen", "PathVQA (trained on SLAKE, 2B)", "short answer",
                 {"base": acc("base2", "pathvqa"), "sft": acc("slk-sft2", "pathvqa"),
                  "grpo": acc("slk-short2-base-e1", "pathvqa"),
                  "sftgrpo": acc("slk-short2-sft-e1", "pathvqa")}))
    rows.append(("Unseen", "TextVQA (trained on VQAv2 + GQA)", "short answer",
                 {m: acc(PH[m], "textvqa") for m in METHODS}))
    rows.append(("Unseen", "VizWiz (trained on VQAv2 + GQA)", "short answer",
                 {m: ood[PH[m]] for m in METHODS}))
    body = []
    for group, name, setting, values in rows:
        best = max(values, key=values.get)
        lo = math.floor(min(values.values()) / 5) * 5
        hi = math.ceil(max(values.values()) / 5) * 5
        if hi - lo < 10:
            hi = lo + 10
        body.append(f'<tr><td class="grp">{group}</td><td><b>{esc(name)}</b>'
                    f'<span class="sub">{esc(setting)}</span></td>{method_cells(values, best)}'
                    f'<td class="stripcell">{strip(values, lo, hi)}'
                    f'<span class="range">{lo}–{hi}</span></td></tr>')
    return "".join(body)


def medical_table():
    out = []
    for label, ds, variant, base, arms in MED:
        own = {"base": acc(base, ds, variant=variant),
               **{m: acc(a, ds, variant=variant) for m, a in arms.items()}}
        path = {"base": acc(base, "pathvqa"), **{m: acc(a, "pathvqa") for m, a in arms.items()}}
        d = delta(arms["sft"], arms["sftgrpo"], ds, variant=variant)
        p = delta(arms["sft"], arms["grpo"], "pathvqa")
        out.append(f'<tr><td><b>{esc(label)}</b></td>'
                   f'{method_cells(own, max(own, key=own.get))}<td class="n">{ci(d)}</td>'
                   f'<td class="n">{path["base"]:.1f} · {path["sft"]:.1f} · '
                   f'<b>{path["grpo"]:.1f}</b> · {path["sftgrpo"]:.1f}</td>'
                   f'<td class="n">{ci(p)}</td></tr>')
    return "".join(out)


def kl_table():
    rows = [("SLAKE, 2B", "slk2"), ("SLAKE, 8B", "slk8"), ("OmniMedVQA, held-out", "omx"),
            ("VQAv2", "vqa"), ("ChartQA, short answer", "cqa")]
    out = []
    for name, key in rows:
        sft, grpo = float(N[f"kl.{key}.sft"]), float(N[f"kl.{key}.grpo"])
        ratio = sft / grpo
        cls = "pos" if ratio > 1 else "neg"
        out.append(f'<tr><td>{name}</td><td class="n">{sft:.2f}</td><td class="n">{grpo:.2f}</td>'
                   f'<td class="n {cls}">{ratio:.1f}×</td></tr>')
    return "".join(out)


def whole(x):
    """Rounded to a whole number, with a typographic minus and no negative zero."""
    return str(round(x)).replace("-", "\u2212")


GATE_REASONS = {"image needed": "image barely used", "not saturated": "near the ceiling"}


def screening_table():
    screening = load("screening/screening.json")
    rows = {r["dataset"]: r for r in screening["rows"] if r["model"] == "base2"}
    order = ["chartqa", "slake", "omnimedvqa", "vqav2", "gqa", "clevrcomp", "kvasirx1",
             "pathvqa", "rexvqa", "medthink", "deeptumor", "deeptumorx", "textvqa", "clevr"]
    out = []
    for r in (rows[d] for d in order if d in rows):
        passed = r["status"] == "passed"
        if passed:
            verdict = '<span class="tag ok">passes</span>'
        else:
            failed = ", ".join(GATE_REASONS.get(g, g) for g in r["failed_gates"]
                               if g != "training questions")
            verdict = (f'<span class="tag">fails</span> '
                       f'<span class="sub-n">· {esc(failed or "no training pool")}</span>')
        payoff = ci(r["matched_payoff"]) if r.get("matched_payoff") else "—"
        out.append(f'<tr><td>{esc(r["display_name"])}</td>'
                   f'<td class="n">{100 * r["short_accuracy"]:.1f}</td>'
                   f'<td class="n">{payoff}</td>'
                   f'<td class="n">{whole(r["image_dependence"]["delta"])} / '
                   f'{whole(r["wrong_image_dependence"]["delta"])}</td>'
                   f'<td class="n">{100 * r["learnable_share"]:.0f}%</td><td>{verdict}</td></tr>')
    return "".join(out)


FRAGMENTS = {
    "AUDIT_FIG": audit_figure(),
    "OVERVIEW_ROWS": overview_table(),
    "CHARTQA_FIG": chartqa_figure(),
    "PASSK_FIG": passk_figure(),
    "BLACK_FIG": black_figure(),
    "MEDICAL_ROWS": medical_table(),
    "KL_ROWS": kl_table(),
    "SCREENING_ROWS": screening_table(),
}


def plain(value):
    """A value without its markup, for the markdown pages and the numbers file."""
    return html.unescape(re.sub(r"<[^>]+>", "", value))


def numbers():
    return {key: plain(value) for key, value in sorted(N.items()) if key != "date"}


def fill_numbers(text, values):
    """Refresh every <!--n:key-->value<!--/n--> in a markdown page."""
    return QUOTED_NUMBER.sub(lambda m: f"<!--n:{m['key']}-->{values[m['key']]}<!--/n-->", text)


def numbered_pages():
    pages = [REPO / "README.md", *sorted((REPO / "docs").rglob("*.md"))]
    talk = REPO / "interview" / "talk.md"  # local only, absent from a clean clone
    return [*pages, talk] if talk.exists() else pages


def main():
    values = numbers()
    NUMBERS_PATH.write_text(json.dumps(values, indent=1, ensure_ascii=False) + "\n")
    for path in numbered_pages():
        text = path.read_text()
        if fill_numbers(text, values) != text:
            path.write_text(fill_numbers(text, values))
            print(f"updated numbers in {path.relative_to(REPO)}")
    page = (HERE / "report_page.template.html").read_text()
    for key, value in FRAGMENTS.items():
        page = page.replace("{{" + key + "}}", value)
    for key, value in N.items():
        page = page.replace("{{" + key + "}}", value)
    if "{{" in page:
        at = page.index("{{")
        raise SystemExit(f"unfilled slot: {page[at:at + 60]}")
    out = REPO / "docs" / "report" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    DELTAS_PATH.write_text(json.dumps(dict(sorted(DELTAS.items())), indent=1) + "\n")
    print(f"{out} {len(page) // 1024} KiB")


if __name__ == "__main__":
    main()
