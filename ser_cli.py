import argparse

from ser import load_audio_16k_mono, predict_emotion_windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SER runner (IEMOCAP 4-class).")
    parser.add_argument("audio_path", help="Path to an audio file (wav/mp3/m4a/...)")
    parser.add_argument("--window", type=float, default=4.0, help="Window length (seconds)")
    parser.add_argument("--hop", type=float, default=2.0, help="Hop length (seconds)")
    args = parser.parse_args()

    audio = load_audio_16k_mono(args.audio_path)
    timeline, label, scores = predict_emotion_windows(audio, window_s=args.window, hop_s=args.hop)

    if not label:
        raise SystemExit("No SER result (empty/unreadable audio).")

    print(f"overall: {label} ({scores.get(label, 0.0):.2f})")
    print("mean_scores:", {k: round(v, 4) for k, v in scores.items()})
    if timeline:
        t0, last_label, last_score, _ = timeline[-1]
        print(f"current: {last_label} ({last_score:.2f}) at {t0:.1f}s")


if __name__ == "__main__":
    main()

