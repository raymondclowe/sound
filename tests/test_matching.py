import numpy as np
from sound_recognition.matching import WordMatcher


def test_add_and_match():
    matcher = WordMatcher(sample_rate=16000)
    # generate a test tone and add as reference
    sr = 16000
    t = np.linspace(0, 1.0, int(sr * 1.0), False)
    tone = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    matcher.add_reference(tone, word_name="test")
    # test with similar tone
    test_tone = 0.08 * np.sin(2 * np.pi * 440.0 * t)
    best, score, all_scores = matcher.best_match(test_tone)
    assert best == "test"
    assert score > 50
