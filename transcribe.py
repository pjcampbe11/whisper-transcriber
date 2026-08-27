#!/usr/bin/env python3
"""
transcribe.py - Local speech-to-text for any audio/video file using faster-whisper,
with optional speaker diarization.

Any file ffmpeg can read works: wav mp3 m4a flac ogg opus aac wma aiff
mp4 mov avi mkv webm flv wmv mpeg 3gp ts m4v ... Audio is extracted to
16 kHz mono WAV in a temp dir before transcription.

Usage:
  python transcribe.py meeting.mp4
  python transcribe.py ./recordings/ --model medium --format srt txt json
  python transcribe.py call.avi --diarize --hf-token hf_xxx
"""
import argparse, json, shutil, subprocess, sys, tempfile, time
from pathlib import Path

MEDIA_EXT = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".aiff", ".aif", ".amr", ".caf",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".mpeg", ".mpg", ".3gp", ".ts", ".m4v", ".mts", ".vob",
}

def fmt_ts(sec, srt=False):
    h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    if srt:
        return f"{int(h):02}:{int(m):02}:{int(s):02},{int((s % 1) * 1000):03}"
    return f"{int(h):02}:{int(m):02}:{int(s):02}"

def collect(path, all_files=False):
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.is_file() and (all_files or f.suffix.lower() in MEDIA_EXT))
    return [p]

def extract_audio(src, tmpdir):
    """Convert any media to 16k mono wav via ffmpeg. Returns path."""
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found. Install it: brew install ffmpeg | apt install ffmpeg | winget install ffmpeg")
    dst = Path(tmpdir) / (src.stem + ".wav")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
           "-acodec", "pcm_s16le", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg failed on {src.name}: {r.stderr.strip()[:300]}")
    return dst

def transcribe_file(model, wav, args):
    segs, info = model.transcribe(
        str(wav), language=args.language, beam_size=args.beam_size,
        vad_filter=not args.no_vad, vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=args.words or args.diarize,
    )
    out = []
    for s in segs:
        seg = {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        if s.words:
            seg["words"] = [{"w": w.word, "start": round(w.start, 2), "end": round(w.end, 2)} for w in s.words]
        out.append(seg)
        if not args.quiet and not args.diarize:
            print(f"[{fmt_ts(s.start)} -> {fmt_ts(s.end)}] {seg['text']}")
    return out, info

def diarize(wav, segments, args):
    """Assign speaker labels using pyannote. Requires HF token + accepted model terms."""
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:
        sys.exit("Diarization needs: pip install pyannote.audio torch")
    token = args.hf_token
    if not token:
        sys.exit("--diarize requires --hf-token (or HF_TOKEN env). Accept terms at huggingface.co/pyannote/speaker-diarization-3.1")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    if torch.cuda.is_available() and args.device != "cpu":
        pipe.to(torch.device("cuda"))
    kw = {}
    if args.num_speakers: kw["num_speakers"] = args.num_speakers
    dia = pipe(str(wav), **kw)
    turns = [(t.start, t.end, spk) for t, _, spk in dia.itertracks(yield_label=True)]

    def speaker_at(start, end):
        best, best_ov = None, 0.0
        for ts, te, spk in turns:
            ov = min(end, te) - max(start, ts)
            if ov > best_ov: best, best_ov = spk, ov
        return best or "UNKNOWN"

    # label words, then rebuild segments so a segment never spans two speakers
    rebuilt = []
    for seg in segments:
        words = seg.get("words") or [{"w": seg["text"], "start": seg["start"], "end": seg["end"]}]
        cur = None
        for w in words:
            spk = speaker_at(w["start"], w["end"])
            if cur and cur["speaker"] == spk:
                cur["end"] = w["end"]; cur["text"] += w["w"]; cur["words"].append(w)
            else:
                cur = {"start": w["start"], "end": w["end"], "speaker": spk, "text": w["w"], "words": [w]}
                rebuilt.append(cur)
    for s in rebuilt:
        s["text"] = s["text"].strip()
        if not args.quiet:
            print(f"[{fmt_ts(s['start'])}] {s['speaker']}: {s['text']}")
    return rebuilt

def write_outputs(src, segments, info, args):
    outdir = Path(args.output) if args.output else src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / src.stem
    def label(s): return f"{s['speaker']}: " if "speaker" in s else ""
    if "txt" in args.format:
        with open(f"{base}.txt", "w", encoding="utf-8") as f:
            for s in segments:
                pre = f"[{fmt_ts(s['start'])}] " if args.timestamps or "speaker" in s else ""
                f.write(f"{pre}{label(s)}{s['text']}\n")
    if "srt" in args.format:
        with open(f"{base}.srt", "w", encoding="utf-8") as f:
            for i, s in enumerate(segments, 1):
                f.write(f"{i}\n{fmt_ts(s['start'], True)} --> {fmt_ts(s['end'], True)}\n{label(s)}{s['text']}\n\n")
    if "json" in args.format:
        clean = [dict(s) for s in segments]
        if not args.words:
            for s in clean: s.pop("words", None)
        with open(f"{base}.json", "w", encoding="utf-8") as f:
            json.dump({"file": str(src), "language": info.language,
                       "language_probability": round(info.language_probability, 3),
                       "duration": round(info.duration, 2), "diarized": args.diarize,
                       "segments": clean}, f, indent=2, ensure_ascii=False)
    return base

def main():
    import os
    ap = argparse.ArgumentParser(description="Transcribe any audio/video file locally with faster-whisper")
    ap.add_argument("input", help="Media file or directory (recursive)")
    ap.add_argument("-m", "--model", default="base", help="tiny|base|small|medium|large-v3|distil-large-v3")
    ap.add_argument("-l", "--language", default=None, help="Language code (en, es...); auto-detect if omitted")
    ap.add_argument("-d", "--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("-c", "--compute", default="default", help="int8|int8_float16|float16|float32|default")
    ap.add_argument("-f", "--format", nargs="+", default=["txt"], choices=["txt", "srt", "json"])
    ap.add_argument("-o", "--output", default=None, help="Output directory (default: next to input)")
    ap.add_argument("--diarize", action="store_true", help="Speaker labels via pyannote (needs HF token)")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token for pyannote")
    ap.add_argument("--num-speakers", type=int, default=None, help="Known speaker count (improves diarization)")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--no-vad", action="store_true", help="Disable voice-activity filtering")
    ap.add_argument("--words", action="store_true", help="Word-level timestamps in json")
    ap.add_argument("--timestamps", action="store_true", help="Prefix txt lines with [HH:MM:SS]")
    ap.add_argument("--all-files", action="store_true", help="In directory mode, try every file, not just known extensions")
    ap.add_argument("--keep-wav", action="store_true", help="Keep extracted 16k wav next to outputs")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper not installed: pip install faster-whisper")

    files = collect(args.input, args.all_files)
    if not files:
        sys.exit(f"No media files found at {args.input}")

    print(f"Loading model '{args.model}' on {args.device} ({args.compute})...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute)

    ok = fail = 0
    with tempfile.TemporaryDirectory(prefix="transcribe_") as tmp:
        for src in files:
            print(f"\n=== {src.name} ===")
            t0 = time.time()
            try:
                wav = extract_audio(src, tmp)
                segments, info = transcribe_file(model, wav, args)
                if args.diarize:
                    segments = diarize(wav, segments, args)
                base = write_outputs(src, segments, info, args)
                if args.keep_wav:
                    shutil.copy(wav, f"{base}.16k.wav")
                print(f"Done: {info.duration:.0f}s audio in {time.time()-t0:.1f}s | lang={info.language} | -> {base}.[{','.join(args.format)}]")
                ok += 1
            except Exception as e:
                print(f"FAILED {src.name}: {e}", file=sys.stderr); fail += 1
    print(f"\n{ok} succeeded, {fail} failed")

if __name__ == "__main__":
    main()
