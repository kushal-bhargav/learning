# Agency Console Frontend

React + Vite single-page demo for `specs/08_demo_ui_spec.md`.

## Run locally

Start the FastAPI backend from the repo root:

```powershell
python -B -m uvicorn src.api:app --reload
```

Then start the frontend:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

The app defaults to `http://127.0.0.1:8000` for the API. Override with:

```powershell
$env:VITE_API_BASE='http://127.0.0.1:8000'
npm.cmd run dev
```

## Screens

- Session Setup: choose fixture persona, occasion, budget hint, default agency, seed.
- Agency Console: stage cards with rationale, output, Accept/Edit/Regenerate/Delegate rest to AI.
- Creative Generation: visible agency slider; dragging is debounced and only calls the backend after a pause.
- Agency Ledger: colored authorship timeline for poster/study export.
- Feedback: Likert and open-ended authorship response submitted to the backend reward signal.

## Replay mode

For unattended poster or large-screen playback with no live backend/GPU:

```powershell
cd frontend
npm.cmd run dev -- --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:5173/?replay=1
```

Replay mode uses `src/replaySession.ts` plus cached assets under `public/replay/`, so a production build can loop the session from static files only.
