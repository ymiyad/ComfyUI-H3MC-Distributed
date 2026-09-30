# ComfyUI-H3MC-Distributed

Run [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) chain generation across [ComfyUI-Distributed](https://github.com/robertvoy/ComfyUI-Distributed) workers — with **no shared folder**, and **no changes to either upstream project**.

## Purpose

H3 Motion Context generates long video chains one clip at a time: each clip needs the previous clip's latent as context, so clips are inherently sequential. ComfyUI-Distributed, on the other hand, is built to run the *same* workflow on multiple GPUs *in parallel* with different seeds — it explicitly does **not** speed up a single generation.

Put together, the useful combination isn't "split one clip across GPUs" (not possible), it's **candidate racing**: generate several candidate takes of the *next* clip on all your workers simultaneously, compare them, keep the best one, and move on to the clip after that. This turns idle GPUs into faster iteration instead of faster single-clip renders.

This add-on provides the three nodes needed to make that loop work reliably:

- **H3 Motion Context Upload Latent (Distributed)** — sits right after your sampler, in place of H3 Motion Context Save Latent. Passes the latent through unchanged and hands a copy to Master (locally if this *is* Master, over HTTP if this is a Worker), keyed by this run's job id and this participant's own identity.
- **H3 Motion Context Save Video With Latent (Distributed)** — run on Master once per candidate, after Distributed Collector + Image Batch Divider have split the round's results. Saves a video with its H3 AV latent embedded directly in the video's own container metadata (the same mechanism ComfyUI's own Save Video node uses to embed workflow JSON). The video *is* the continuation state — no separate `.safetensors` file to keep in sync, overwrite by accident, or lose track of.
- **H3 Motion Context Load Latent From Video (Distributed)** — loads the latent embedded in a video saved by the node above, via a standard ComfyUI video-upload widget reading from `ComfyUI/input`. Because it's a normal input-folder video, ComfyUI-Distributed's own media-sync ships whichever file you pick to every worker automatically for the next round.

Because candidates end up as ordinary, playable video files, picking the best one is just picking a video — no separate manifest, slot numbering, or bookkeeping to keep straight, and no possibility of silently continuing from the wrong take.

## Requirements

- [ComfyUI-Distributed](https://github.com/robertvoy/ComfyUI-Distributed)
- [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
- [PyAV](https://pyav.org/) (`av`) — usually already present as a ComfyUI dependency (used for video I/O); install with `pip install av` if missing
- `safetensors` and `torch` — already required by ComfyUI

## Installation

1. Make sure ComfyUI-Distributed and ComfyUI-H3-Motion-Context are installed and working (Master + Workers) first — this add-on only connects the two, it doesn't replace either.
2. Clone or copy this repository into `custom_nodes/` on **every** machine (Master and every Worker):
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/ymiyad/ComfyUI-H3MC-Distributed.git
   ```
3. Restart ComfyUI on every machine.

No configuration file is needed. `H3 Motion Context Save Video With Latent` and `H3 Motion Context Load Latent From Video` are only ever used on Master; `H3 Motion Context Upload Latent` is used in every candidate's generation branch, so it runs on both Master and Workers depending on which machine draws that candidate.

## Tested against

Developed and tested against:

- ComfyUI-Distributed — `main` branch, commit `32ac027` (2026-09-30; latest tagged release at time of writing is `v.1.4.0`)
- ComfyUI-H3-Motion-Context — `main` branch, commit `5335715` / tag `v0.6.2`

ComfyUI-Distributed evolves quickly and this add-on relies on a few of its internal, undocumented details (see [How it works](#how-it-works)) — if something breaks after updating Distributed, that internal shape is the first thing to check.

## How it works

This add-on leans on two things ComfyUI-Distributed already does, rather than fighting them:

1. **Dependency-chain pruning.** Distributed only ships a Worker the nodes reachable from/to its `Distributed Collector` node. `H3 Motion Context Upload Latent` passes its `latent` input straight through as its output, so wiring it directly after your sampler (`Sampler → Upload Latent → VAE Decode → … → Distributed Collector`) keeps it on that chain without needing any change to H3 Motion Context's own nodes.
2. **Input-file media sync.** Before dispatching a prompt, Distributed scans for inputs named `image`/`video`/`audio`/`file` with a recognized extension that live under `ComfyUI/input`, and ships those files to every Worker automatically. `H3 Motion Context Load Latent From Video` uses a standard ComfyUI video-upload combo backed by `ComfyUI/input` for exactly this reason — picking a file there is enough for Distributed to make it available on every Worker for the next round, with no shared network folder.

Getting a candidate's latent from wherever it was rendered back to Master (`H3 Motion Context Upload Latent`) uses a small local HTTP endpoint this add-on registers on Master's own ComfyUI server (`POST /h3_distributed/latent`), storing payloads under `ComfyUI/temp/h3_distributed_latents/` keyed by this run's job id. Entries are cleaned up automatically after 6 hours.

**Avoiding wrong-take risk:** each candidate is stored under a key tied to *who produced it* — Master's own candidate under a fixed slot, each Worker's candidate under its own position in the Distributed panel's worker list — never a shared numbered slot multiple machines could race to write. `Save Video With Latent` resolves Image Batch Divider's `batch_N` numbering back to the correct candidate by reading Master's own `delegate_only` (Orchestrator-only) status, which is only ever read reliably on Master, since this node only ever runs there.

## Usage

### Per-round loop

1. Build your usual H3 Motion Context sampling graph for the next clip, using the previous round's winning context (see [Continuing the chain](#continuing-the-chain) below for round 1+).
2. Right after the sampler, insert **H3 Motion Context Upload Latent** before `VAE Decode`:
   ```
   Sampler → H3 Motion Context Upload Latent → VAE Decode → … → Distributed Collector
   ```
3. In ComfyUI-Distributed's panel, use **Parallel Generation** (Distributed Seed, one candidate per worker with a different seed) so every enabled worker renders its own take of this clip simultaneously.
4. On Master, after `Distributed Collector`, add **Image Batch Divider** with `divide_by` = the number of participants (Master + enabled workers, or just enabled workers if Master is Orchestrator-only).
5. For each `batch_N` output: `Create Video`, then **H3 Motion Context Save Video With Latent** with `participant_index = N`.
6. Preview/play the resulting candidate videos and pick the one you like.
7. Copy (or move) the winning candidate's file into `ComfyUI/input/` on Master.
8. Repeat from step 1 for the next clip, using the promoted file as context (below).

### Continuing the chain

Use **H3 Motion Context Load Latent From Video**, with `video` pointing at the winning candidate you copied into `ComfyUI/input/`:

- **First clip in a chain:** set `enabled = False`. The node returns `None` for `context_latent`, matching H3 Motion Context's own "no previous clip" behavior.
- **Every clip after that:** set `enabled = True` and pick the file.

Because the video lives in `ComfyUI/input/`, ComfyUI-Distributed ships it to every Worker automatically before the next round — nothing else to configure.

### Non-Distributed (single-machine) use

All three nodes also work with no Distributed graph at all: `Upload Latent` just stores locally, `Save Video With Latent` defaults to `participant_index = 1` (its own local slot), and `Load Latent From Video` works exactly the same way. This is a convenient way to switch a single-machine chain over to the "one file is the continuation state" model even if you never plan to distribute it.

## Node reference

| Node | Where it runs | Key inputs | Output |
|---|---|---|---|
| H3 Motion Context Upload Latent (Distributed) | Every candidate branch (Master or Worker) | `latent` | `latent` (pass-through) |
| H3 Motion Context Save Video With Latent (Distributed) | Master only | `video`, `participant_index` (matches `batch_N`), `filename_prefix`, `format` | `video` (pass-through), `video_path` (relative to the output directory) |
| H3 Motion Context Load Latent From Video (Distributed) | Master (feeds a graph Distributed will dispatch) | `enabled`, `video` (from `ComfyUI/input`) | `latent` (or `None` if `enabled = False`) |

## Notes and caveats

- **Re-encoding strips the embedded latent.** Anything that re-encodes or re-muxes a saved candidate (uploading to a hosting site and re-downloading, running it through an editor, etc.) will remove the metadata. Keep the original file around for as long as you might want to continue the chain from it, and only pass *that* file to `Load Latent From Video`.
- **`mp4` vs `mkv`/`webm` for large latents.** The latent is embedded as base64 (~1.33× its raw size) in container metadata. This has worked fine in testing, but Matroska (`mkv`) is the more natural fit for arbitrary embedded payloads if you run into issues with very long/high-resolution clips on `mp4`.
- **The upload endpoint is unauthenticated.** `POST /h3_distributed/latent` on Master accepts uploads from any machine that can reach it, same trust assumption as ComfyUI-Distributed's own Master↔Worker traffic. Don't expose Master's ComfyUI port beyond your own trusted network.
- **Temp storage isn't permanent, but this only matters briefly.** Uploaded candidates live under `ComfyUI/temp/h3_distributed_latents/` and are cleaned up after 6 hours. This window is between a round's `Upload Latent` calls and running `Save Video With Latent` for each `batch_N` candidate — once a candidate's latent is embedded in its saved video, the video file is the permanent record and the temp entry is no longer needed. There's no time pressure on picking a winner or promoting it to `ComfyUI/input/` afterward.
- **Video-upload widgets vary.** In testing, VHS (Video Helper Suite)'s "Load Video (Upload)" node did not work reliably through Distributed for reasons that weren't fully identified; ComfyUI's own standard video loader (and this add-on's `Load Latent From Video`, which uses the same underlying widget pattern) worked without issue. If you build additional nodes around this workflow, prefer ComfyUI's built-in video-upload combo pattern.

## License

GPL-3.0. See [LICENSE](LICENSE).

This project depends on ComfyUI-H3-Motion-Context (GPL-3.0) and ComfyUI-Distributed (Apache-2.0) at runtime but doesn't incorporate code from either, so GPL-3.0 for this repository's own code doesn't relicense them.

## Acknowledgments

This project exists entirely thanks to two projects it builds on top of, neither of which needs any modification to work with it:

- [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) by NikoDemon80 — the MiniMax H3 chain-generation nodes this add-on connects to Distributed.
- [ComfyUI-Distributed](https://github.com/robertvoy/ComfyUI-Distributed) by robertvoy — the multi-machine orchestration this add-on relies on for its dependency-chain and media-sync behavior.

Thank you to both authors for the work that made this possible.
