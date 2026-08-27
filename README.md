None selected 

Skip to content
Using Gmail with screen readers
1 of 20,894
(no subject)
Inbox
email notes from P

P <patcampbell82@gmail.com>
Attachments
3:02 PM (1 minute ago)
to me

 3 Attachments
  •  Scanned by Gmail


# whisper-transcriber

Local, offline speech-to-text for **any audio or video file** — calls, meetings, voicemails, screen recordings, videos — using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 port of OpenAI Whisper, ~4x faster than the original). Optional speaker labels via pyannote. Nothing leaves your machine.

## Supported input
Anything ffmpeg can decode. Audio is extracted to 16 kHz mono WAV automatically.

| Audio | Video |
|---|---|
| wav mp3 m4a flac ogg opus aac wma aiff amr caf | mp4 mov avi mkv webm flv wmv mpeg mpg 3gp ts m4v mts vob |

Directory mode recurses and picks up those extensions; add `--all-files` to try everything.

## Features
- Single file or recursive directory batch
- Output: `txt`, `srt` (subtitles), `json` (segments, optional word timestamps)
- `--diarize` → "who said what" speaker labels (SPEAKER_00, SPEAKER_01, ...)
- Auto language detection or force one
- VAD filtering skips silence / hold music
- CPU or CUDA, int8 quantization for low-RAM boxes
- Per-file error handling; one bad file doesn't kill the batch

## Install
```bash
pip install -r requirements.txt
# ffmpeg is required
#   macOS:   brew install ffmpeg
#   Ubuntu:  sudo apt install ffmpeg
#   Windows: winget install ffmpeg
```

**Diarization (optional):**
```bash
pip install pyannote.audio torch
```
1. Create a token at https://huggingface.co/settings/tokens
2. Accept the terms on https://huggingface.co/pyannote/speaker-diarization-3.1 and https://huggingface.co/pyannote/segmentation-3.0
3. Pass `--hf-token hf_xxx` or `export HF_TOKEN=hf_xxx`

GPU: CUDA 12 + cuDNN 9. See faster-whisper README for wheels.

## Usage
```bash
# any file, default model
python transcribe.py meeting.mp4
python transcribe.py call.avi

# folder of recordings, better accuracy, all formats, to a transcripts dir
python transcribe.py ./recordings/ -m medium -f txt srt json -o ./transcripts

# speaker labels, 2 known speakers
python transcribe.py call.wav --diarize --num-speakers 2 -f txt json

# GPU, English only, word-level timestamps
python transcribe.py call.mkv -m large-v3 -d cuda -c float16 -l en -f json --words

# low-RAM CPU
python transcribe.py call.mp3 -m small -d cpu -c int8
```

Diarized txt output looks like:
```
[00:00:03] SPEAKER_00: Thanks for joining, let's get started.
[00:00:07] SPEAKER_01: Sure, I have the numbers from last week.
```

## Model picker
| Model | RAM/VRAM | Speed | Use when |
|---|---|---|---|
| tiny / base | <1 GB | fastest | quick notes, clean audio |
| small | ~2 GB | fast | everyday meetings |
| medium | ~5 GB | moderate | accents, crosstalk |
| large-v3 | ~10 GB | slow on CPU | best quality, non-English |
| distil-large-v3 | ~6 GB | fast | near large-v3 quality, English |

Models download on first run to `~/.cache/huggingface`.

## Options
```
-m/--model        tiny|base|small|medium|large-v3|distil-large-v3
-l/--language     force language (en, es, ...) — omit to auto-detect
-d/--device       auto|cpu|cuda
-c/--compute      int8|int8_float16|float16|float32
-f/--format       txt srt json (one or more)
-o/--output       output dir (default: next to input file)
--diarize         speaker labels (pyannote, needs HF token)
--hf-token        Hugging Face token (or HF_TOKEN env)
--num-speakers N  known speaker count
--words           word-level timestamps in json
--timestamps      prefix txt lines with [HH:MM:SS]
--no-vad          disable voice-activity filter
--beam-size N     decoding beam (default 5; 1 = faster)
--all-files       directory mode: attempt every file
--keep-wav        keep the extracted 16k wav
```

## Notes
- Diarization runs a second pass and is much slower on CPU; use a GPU for anything over ~20 minutes.
- Check consent laws before recording calls (Illinois is all-party consent).
README.md
Displaying README.md.
