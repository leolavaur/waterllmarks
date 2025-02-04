# ruff: noqa: D100,D103
import pytest
from nltk.translate.meteor_score import meteor_score, single_meteor_score
from ragas.dataset_schema import SingleTurnSample

# Note: the import also downloads the required NLTK data.
from waterllmarks.evaluation import MetaMeteor


# Testing the upstream functions.
# -------------------------------
@pytest.fixture
def upstream_data():
    hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']  # fmt: skip
    hypothesis2 = ['It', 'is', 'to', 'insure', 'the', 'troops', 'forever', 'hearing', 'the', 'activity', 'guidebook', 'that', 'party', 'direct']  # fmt: skip
    reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']  # fmt: skip
    reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which', 'guarantees', 'the', 'military', 'forces', 'always', 'being', 'under', 'the', 'command', 'of', 'the', 'Party']  # fmt: skip
    reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the', 'army', 'always', 'to', 'heed', 'the', 'directions', 'of', 'the', 'party']  # fmt: skip
    return hypothesis1, hypothesis2, reference1, reference2, reference3


def test_meteor_single_reference(upstream_data):
    hypothesis1, _, reference1, _, _ = upstream_data
    score = single_meteor_score(hypothesis=hypothesis1, reference=reference1)
    assert round(score, 4) == 0.6944


def test_meteor_multiple_references(upstream_data):
    hypothesis1, _, reference1, reference2, reference3 = upstream_data
    score = meteor_score(
        hypothesis=hypothesis1, references=[reference1, reference2, reference3]
    )
    assert round(score, 4) == 0.6944


# Testing the MetaMeteor class.
# -----------------------------


@pytest.fixture
def meta_meteor_data():
    hypothesis1 = "It is a guide to action which ensures that the military always obeys the commands of the party."  # fmt: skip
    hypothesis2 = "It is to insure the troops forever hearing the activity guidebook that party direct."  # fmt: skip
    reference1 = "It is a guide to action that ensures that the military will forever heed Party commands."  # fmt: skip
    reference2 = "It is the guiding principle which guarantees the military forces always being under the command of the Party."  # fmt: skip
    reference3 = "It is the practical guide for the army always to heed the directions of the party."  # fmt: skip
    return hypothesis1, hypothesis2, reference1, reference2, reference3


@pytest.fixture
def meta_meteor():
    return MetaMeteor()


def test_meta_meteor_single_reference(meta_meteor, meta_meteor_data):
    hypothesis1, _, reference1, _, _ = meta_meteor_data
    sample = SingleTurnSample(reference=reference1, response=hypothesis1)
    score = meta_meteor.single_turn_score(sample)
    assert abs(0.6944 - score) < 1e-2
