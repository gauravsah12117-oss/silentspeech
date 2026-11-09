# Poster layout (A1 or A0, portrait)

```
┌──────────────────────────────────────────────────┐
│  SILENTSPEECH — tagline, team, college, logo     │
├────────────────┬────────────────┬────────────────┤
│  The problem   │  Architecture  │  Results       │
│  (who, why)    │  diagram       │  (accuracy,    │
│                │                │  confusion     │
├────────────────┤                │  matrix)       │
│  Our solution  ├────────────────┼────────────────┤
│  + demo photo  │  How the AI    │  Responsible   │
│                │  works         │  AI            │
├────────────────┴────────────────┼────────────────┤
│  Impact and future roadmap      │  QR codes:     │
│                                 │  demo + GitHub │
└─────────────────────────────────┴────────────────┘
```

- Title text 90–120 pt, headings 40–50 pt, body text 24–32 pt.
- Under about 500 words, so it reads from 2 metres.
- High-resolution images. The charts in `results/` are saved at 130–140 dpi; for print, re-save them at 300 dpi (change `dpi=` in `evaluate.py`).
- One QR code to the demo video, one to the GitHub repo. The repo must be public before you print the code.
- Demo photo: turn on `SAVE_VIDEO` in `config.py` only for a teammate who ticked the video box on the consent form.
