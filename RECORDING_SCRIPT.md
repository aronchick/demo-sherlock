# Recording script: a 10 ms Scout decides when a model is worth calling

One take, about 2:00, five beats. The board at `/log` is the recording surface,
full screen at 1920×1080. Every state below comes from the runtime; the
operator presses two console buttons and nothing else.

## Before you roll

```bash
just up      # warm-up: the first uv run resolves dependencies
just down
just up      # the take
```

Open `http://localhost:8000/log` (or your `DASHBOARD_PORT`) in one tab only: the
board holds four streams open, and a second tab of it stalls the feeds.

**Timing.** The synthetic scene loops every 26 s: the cup is raised at 3.4 s
and flagged about 0.6 s later; the knife is raised at 14.9 s and flagged about
0.6 s later. Offsets below are seconds after a person first walks in from the
left with a cup (scene 1.0 s). A slipped take runs longer; it never
desynchronises, because each beat waits for its event on screen.

**Mode.** `just up` is the mock runtime and the masthead says so. Say only what
the board shows: in mock mode the Scout reads scene truth and the models are
canned; never call that take live.

---

## Beat 1: THE GATE · 0:00–0:20

**Screen state before you speak.** `MONITORING · 2 CAMERAS`. Amber particles
leave both cameras and die on the Scout's right edge; Director and VLM are grey.

**Driver**: nothing to type.

> Every frame from both cameras is one of these dots. They all stop at the
> Scout. Behind it, the Director and the vision model are asleep.

---

## Beat 2: THE CUP · 0:20–0:40

**Screen state before you speak.** `IGNORED CUP · 0 VLM CALLS`. Only the
Director lit; chain steps 1–5 skipped; IGNORED counts up.

> A cup crosses the gate. The Director wakes, decides it is benign, and
> reports it. Zero VLM calls.

---

## Beat 3: THE KNIFE · 0:40–1:05

**Screen state before you speak.** `INVESTIGATING KNIFE · STEP n/6`, the gate
bar red, both models lit, then `ALERT · KNIFE · VERIFIED` and the incident card.

> A knife runs the whole chain: zoom, read, zoom to the head, store the review
> crop, verify, report. The review crop is kept for a person. No model ever
> sees it.

---

## Beat 4: BREAK IT · 1:05–1:35

**Screen state before you speak.** Monitoring again, before the next knife.

**Driver**: press **break the next tool call**; after the alert, press
**cover hero lens**.

> I break the next camera call. Step one fails on hero; the Director retries on
> scout1 and the alert still lands. Now I cover the hero lens: hero stops
> sending frames, and the next detection comes from scout1.

---

## Beat 5: CLOSE · 1:35–2:00

**Screen state before you speak.** The Scout's ledger: `VLM calls avoided`
next to the two meter tracks. Uncover hero first.

> Read the number on the Scout card out loud, then stop. Against one VLM call
> per frame, that is how many calls this room did not make.

> **Closing line:** A ten millisecond Scout decides when a model is worth
> calling.

"Ten millisecond" is copy until it is measured: say it only on a live take
whose Scout card shows about 10 ms per frame. Otherwise close with "A Scout
decides when a model is worth calling."

---

## Required lines

- "Zero VLM calls"
- "No model ever sees it"
- the `VLM calls avoided` number as rendered
- the closing line, in the form the take's mode allows

## Prohibited

Never spoken or shown: port numbers, a dollar figure for the avoided calls,
"face recognition" or anything that identifies the person, and "live" over a
mock take. Model names are spoken only when the masthead says LIVE.
