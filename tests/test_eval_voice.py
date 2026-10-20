import pytest

from eval_voice import edit_distance, score_turns


def test_voice_scores_use_only_available_labels():
    report = score_turns([
        {"id": "one", "reference_transcript": "one two three", "transcript": "one three",
         "expected_language": "hi", "actual_language": "hi", "turn_detection_error": False,
         "interrupted_previous_response": True, "time_to_first_audio_ms": 500},
        {"id": "two", "reference_transcript": "hello", "transcript": "yellow",
         "expected_language": "en", "actual_language": "hi", "turn_detection_error": True,
         "time_to_first_audio_ms": 900},
    ])
    assert edit_distance(["one", "two", "three"], ["one", "three"]) == 1
    assert report["word_error_rate"] == 0.5
    assert report["answer_language_accuracy"] == 0.5
    assert report["turn_detection_error_rate"] == 0.5
    assert report["interruptions"] == 1
    assert report["time_to_first_audio_ms_p50"] == 500


def test_missing_ground_truth_is_not_scored():
    report = score_turns([{"id": "one", "time_to_first_audio_ms": 100}])
    assert report["word_error_rate"] is None
    assert report["answer_language_accuracy"] is None
    assert report["turn_detection_error_rate"] is None


@pytest.mark.parametrize("turn", [
    {"id": "x", "transcript": "hello"},
    {"id": "x", "turn_detection_error": "yes"},
    {"id": "x", "time_to_first_audio_ms": -1},
])
def test_invalid_labels_rejected(turn):
    with pytest.raises(ValueError):
        score_turns([turn])
