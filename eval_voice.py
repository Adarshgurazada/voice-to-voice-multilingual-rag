"""Score human-labeled voice turns without calling an API or storing audio."""

import argparse
import json
from pathlib import Path


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(
                current[-1] + 1, previous[j] + 1,
                previous[j - 1] + (expected != actual),
            ))
        previous = current
    return previous[-1]


def score_turns(turns: list[dict]) -> dict:
    if not turns:
        raise ValueError("Provide at least one labeled voice turn.")
    seen = set()
    word_errors = word_count = 0
    language_correct = language_count = 0
    vad_errors = vad_count = 0
    interruptions = 0
    latencies = []
    for turn in turns:
        turn_id = turn.get("id")
        if not isinstance(turn_id, str) or not turn_id or turn_id in seen:
            raise ValueError("Each turn needs a unique nonempty id.")
        seen.add(turn_id)
        if "reference_transcript" in turn or "transcript" in turn:
            if not isinstance(turn.get("reference_transcript"), str) or not isinstance(turn.get("transcript"), str):
                raise ValueError(f"{turn_id}: both reference_transcript and transcript are required.")
            reference = turn["reference_transcript"].casefold().split()
            hypothesis = turn["transcript"].casefold().split()
            word_errors += edit_distance(reference, hypothesis)
            word_count += len(reference)
        if "expected_language" in turn or "actual_language" in turn:
            expected = turn.get("expected_language")
            actual = turn.get("actual_language")
            if not isinstance(expected, str) or not isinstance(actual, str) or not expected or not actual:
                raise ValueError(f"{turn_id}: expected_language and actual_language need human labels.")
            language_count += 1
            language_correct += expected.casefold() == actual.casefold()
        if "turn_detection_error" in turn:
            if not isinstance(turn["turn_detection_error"], bool):
                raise ValueError(f"{turn_id}: turn_detection_error must be a boolean.")
            vad_count += 1
            vad_errors += turn["turn_detection_error"]
        if "interrupted_previous_response" in turn:
            if not isinstance(turn["interrupted_previous_response"], bool):
                raise ValueError(f"{turn_id}: interrupted_previous_response must be a boolean.")
            interruptions += turn["interrupted_previous_response"]
        if "time_to_first_audio_ms" in turn:
            value = turn["time_to_first_audio_ms"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{turn_id}: time_to_first_audio_ms must be nonnegative.")
            latencies.append(value)
    latencies.sort()
    return {
        "turns": len(turns),
        "word_error_rate": word_errors / word_count if word_count else None,
        "transcription_words": word_count,
        "answer_language_accuracy": language_correct / language_count if language_count else None,
        "language_labeled_turns": language_count,
        "turn_detection_error_rate": vad_errors / vad_count if vad_count else None,
        "vad_labeled_turns": vad_count,
        "interruptions": interruptions,
        "time_to_first_audio_ms_p50": latencies[(len(latencies) - 1) // 2] if latencies else None,
        "time_to_first_audio_ms_p95": latencies[max(0, (95 * len(latencies) + 99) // 100 - 1)] if latencies else None,
        "latency_turns": len(latencies),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Human-annotated JSONL turns")
    parser.add_argument("--output", type=Path, default=Path(".local_data/voice_report.json"))
    args = parser.parse_args()
    try:
        with args.dataset.open(encoding="utf-8") as handle:
            turns = [json.loads(line) for line in handle if line.strip()]
        report = score_turns(turns)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Voice evaluation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
