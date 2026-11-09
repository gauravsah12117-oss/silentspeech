# PPT outline (12 slides)

One idea per slide, large visuals, at most about 30 words per slide, the same fonts and colours throughout.

| # | Slide | Content | Where the material comes from |
|---|---|---|---|
| 1 | Title | Project name, tagline, team names, college | |
| 2 | The hook | "Imagine waking up after surgery, in pain, and being unable to tell anyone." One powerful image, very little text. | |
| 3 | The problem | Who loses their voice, how they communicate today, and the gaps in current solutions | README section 1 |
| 4 | Our solution | One sentence plus a simple visual: lips → AI → their own voice | README section 2 |
| 5 | Live demo | Run `python app.py` live; keep the backup video on this slide | Demo script in README section 25 |
| 6 | Architecture | The pipeline diagram | README section 5 |
| 7 | How the AI works | Landmarks, the model, the viseme challenge, how the LLM uses context | README sections 6, 12, 17 |
| 8 | Our dataset | Phrases, speakers, clips, any regional-language data | `python check_setup.py` prints the counts |
| 9 | Results | Accuracy, confusion matrix, latency, learning curve, before/after enrolment | `results/metrics.json`, `results/*.png`, `results/experiments.csv` |
| 10 | Responsible AI | Consent, privacy, voice-clone safeguards, emergency safety design | README section 20 |
| 11 | Impact and scale | ₹0 hardware, who benefits, path to hospitals | README section 26 |
| 12 | Future and thanks | Roadmap, GitHub QR code, thank you, questions | |

Only put numbers on slide 9 that came out of `evaluate.py` on real recordings.
