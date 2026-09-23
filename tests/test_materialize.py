"""Every registry entry materializes its declared subsets from pinned sources (tiny fixtures)."""

import io

import datasets
import pytest
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from vqar import data
from vqar.registry import DATASETS, SOURCE_REVISIONS


def png(shade: int) -> dict:
    buffer = io.BytesIO()
    PILImage.new("RGB", (2, 2), (shade, shade, shade)).save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": None}


def cauldron():
    rows = [{"images": [png(i)], "texts": [{"user": f"q{i}-{j}?\nAnswer briefly.",
                                             "assistant": "Yes."} for j in range(2)]}
            for i in range(6)]
    features = Features({"images": Sequence(Image()),
                         "texts": [{"user": Value("string"), "assistant": Value("string")}]})
    return Dataset.from_list(rows, features=features)


def annotated(n: int, answers):
    rows = [{"question_id": i, "image": png(i), "question": f"q{i}?", "answers": answers,
             "answer_type": "other"} for i in range(n)]
    return Dataset.from_list(rows).cast_column("image", Image())


def gqa(config: str):
    if config.endswith("_images"):
        return Dataset.from_list([{"id": f"img{i}", "image": png(i)} for i in range(6)]
                                 ).cast_column("image", Image())
    return Dataset.from_list([{"id": f"q{i}", "imageId": f"img{i}", "question": f"q{i}?",
                               "answer": "left", "types": {"structural": "choose"}}
                              for i in range(6)])


def clevr():
    rows = [{"image": PILImage.new("RGBA", (2, 2)), "problem": f"How many? {i}",
             "solution": "<answer>3</answer>"} for i in range(6)]
    return Dataset.from_list(rows).cast_column("image", Image())


@pytest.fixture
def pinned_sources(monkeypatch, tmp_path):
    """Serve fixtures in place of the Hub and record the revision of every request."""
    requests = []

    def load_dataset(repo, config=None, split=None, revision=None):
        requests.append((repo, revision))
        if repo == "HuggingFaceM4/the_cauldron":
            return cauldron()
        if repo == "lmms-lab/VQAv2":
            return annotated(3, [{"answer": "yes"}] * 10)
        if repo == "lmms-lab/GQA":
            return gqa(config)
        if repo == "leonardPKU/clevr_cogen_a_train":
            return clevr()
        if repo == "lmms-lab/textvqa":
            return annotated(3, ["yes"] * 10)
        raise AssertionError(f"unexpected source {repo}")

    monkeypatch.setattr(data, "load_dataset", load_dataset)
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(tmp_path))
    return requests


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_entry_materializes_declared_subsets_in_the_unified_schema(name, pinned_sources):
    spec = DATASETS[name]
    small = {key: 0 if "seed" in key else 2 for key in spec.sizes}
    subsets = spec.materialize(small)
    assert set(spec.subsets.values()) <= set(subsets)
    for subset in subsets.values():
        assert subset.features == data.FEATURES
        assert len(subset) and set(subset["dataset"]) == {name}
    assert {repo for repo, _ in pinned_sources} == set(spec.sources)
    assert all(revision == SOURCE_REVISIONS[repo] for repo, revision in pinned_sources)
