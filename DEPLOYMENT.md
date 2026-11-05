# Local LIBERO-Recovery deployment

This checkout runs real public learned-policy rollouts with the official **openpi π0.5 LIBERO** checkpoint. The author's starVLA custom checkpoint is not distributed in the repository. The transport/backend is adapted; this is a learned-policy continuation recovery baseline, not the author's reported model or a new recovery algorithm.

Source: `liulin815/LIBERO-Recovery` revision `832d54dfc43f0c9e4240bb8d92f5fda7001bb093`. Policy implementation: `Physical-Intelligence/openpi` revision `215abfb217dbac7d5f1273282331b9b1866c0479`. Dataset: ModelScope `ataier/LIBERO_Recovery_FailureScenes`, revision `95acf98dd41a4965018069e7e4f27aac1fc69328`. Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`; cloud object generations and published hashes are pinned in `models/pi05_libero/source-files.json`, locally verified CRC32C/MD5/SHA256 hashes in `models/pi05_libero/verified-hashes.json`.

The simulator uses a project-local Python3.10 venv, MuJoCo3.2.3 and robosuite1.4.0. The policy uses a separate Python3.11 venv and the official `vendor/openpi/uv.lock`. No global Python packages or existing TacAgent environments are changed. All data, cache, checkpoints, output and local configuration stay here. The benchmark does not require Isaac or Docker. Existing `tacagent-rsi` is untouched.

## Commands

From `/mnt/data/Hangxing/benchmarks/LIBERO-Recovery`:

```bash
scripts/bench setup
scripts/bench doctor
scripts/bench replay-check
scripts/bench list --suite libero_goal --level L1
scripts/bench policy --suite libero_goal --task open_the_middle_drawer_of_the_cabinet --trials 1 --seed 7
scripts/bench recovery --suite libero_goal --task open_the_middle_drawer_of_the_cabinet --level L1 --scenes 2 --close-trials 1 --open-trials 1 --seed 7
scripts/bench status
# Native policy1episode + declared recovery4episodes, sharing one policy server:
scripts/bench acceptance
```

`setup` installs pinned environments and the full failure-scene evaluation data; it does not download the optional80GB raw rollouts or training demonstrations. It is resumable. Do not run simultaneous setup invocations. `policy` and `recovery` acquire the host `../.coordination/gpu.lock` before starting a loopback-only policy server, retain it for all simulator/policy work, and stop only their own server before releasing it. The first inference includes JAX compilation. `policy_rng_seed:0` means the server starts a global diffusion RNG stream at seed0; it is not reset per episode. Every inference records its server query index, so the full acceptance sequence is reproducible. Standalone policy/recovery commands each start a fresh stream. Policy command output and server logs are preserved under `logs/`. GPU access requires running outside a filesystem/network restricted execution sandbox.

For the full failure protocol after assets finish downloading:

```bash
scripts/bench recovery --suite any --task any --level any --scenes 0 --close-trials 5 --open-trials 5 --noise 0.02 --seed 7
```

This is48030rollouts and has not been completed as part of a deployment smoke check. `--suite`, `--task`, `--level`, `--scene-id`, `--scenes`, `--close-trials`, `--open-trials`, `--noise`, `--max-steps`, `--replan-steps` and `--seed` select smaller evaluations. `--scenes 0` means all matching downloaded scenes; a positive number samples scenes using a dedicated seeded RNG. Native task evaluation accepts `--task any` and `--suite any`. Original benchmark perturbations and gripper split are implemented, with the corrections below. The acceptance subset was declared in `deployment/experiment-plan.json` before policy execution; all outcomes are retained.

## Protocol and adaptations

- Failure `.bddl` and `_sim_state.npy` come from the public benchmark, not generated substitute tasks. Each episode records scene/state hashes and its source annotation.
- Initialization loads an in-memory copy of the failure state, applies fresh uniform±2cm xy noise only to free-joint objects, restores it, reanchors the OSC goal, then physically drives the gripper closed/open for20steps. Scene files are never changed. The native success predicate is checked before learned control. No target-object state writes occur during recovery.
- The upstream perturb helper indexes flattened MuJoCo state with a qpos address without accounting for the leading simulation time. This adapter uses offset+1. `scripts/bench replay-check` verifies changed entries are exactly the intended object xy entries, all other entries and source state stay unchanged, and the reconstructed state has the expected dimensions.
- The upstream client uses stale observations from before gripper settling. This adapter uses the final actual settle observation.
- Official openpi input processing rotates both cameras180°, resizes/pads224×224, includes8D EEF/finger state and the `.bddl` language for recovery (native policy evaluation uses the official task filename converted to words, matching the official openpi example). Native success requires environment done **and** a fresh native task predicate, so a horizon termination never becomes task success. The returned7D OSC/gripper actions are already in native LIBERO convention and are passed directly to `env.step`; no starVLA gripper inversion or action ensemble is applied. Replanning is every5steps, as in the official openpi LIBERO example.
- The websocket15 keepalive default expires during first JAX compilation because the official server blocks its asyncio loop during synchronous inference. The local client/server transport disables ping timeouts; inference payloads, model weights and learned actions are unchanged. The first failed startup/inference attempt is retained under `runs/20261105T094843Z_acceptance_policy`. Future startup checks use the official `/healthz` endpoint.
- 4338 of4803 official selection index rows omit level fields despite the README description; levels are loaded from matching public per-scene annotations. Missing levels are never invented.
- Reachable L1 failures are evaluated as `learned_policy_continuation`. Closed starts are out of distribution for ordinary LIBERO-trained policies. The public π0.5 checkpoint reports training/evaluation on spatial/object/goal/10; libero90 execution evaluates additional task generalization. Reported selected smoke outcomes must not be presented as a full recovery benchmark score.

## Evidence

`deployment-status.json` distinguishes environment/assets, policy inference/rollout, task success, failure replay, recovery execution and recovery success. `logs/doctor.json` and `logs/doctor.png` establish finite physics, actual actuator steps and nonblank EGL rendering. `logs/scene-validation.json` records failure-state reconstruction/address checks.

Each `runs/<timestamp>_<mode>/` contains effective `config.json`, scene selections, per-episode `result.json`, all predicted action chunks, timing, native outcomes, raw `trajectory.npz` with images/states/actions/full simulator state, initial/final simulator states and camera videos. Failed episodes and failed server startup logs remain available. `initial_native_success` invalidates any recovery success when the scene was already solved; only native predicate success reached through learned action stepping counts.

Current verified outcomes and concrete run paths are recorded after GPU evaluation in `deployment-status.json`. Installation and a full dataset download alone do not satisfy the learned-policy/recovery gates.

## Verified local result (2026-11-05)

The entire public evaluation dataset is installed:4803 scene pairs and4803 annotations. All16 policy checkpoint objects passed published CRC32C verification, MD5 where supplied, and local SHA256 recording.

Actual tested run: `scripts/bench acceptance` (seed7), run stem `20261105T095735Z_acceptance`. All5 simulator/policy episodes completed and reached the native goal through learned actions:

| Run | Initial state | Split | Steps | Native success before→after |
|---|---|---|---:|---|
| acceptance_policy | Native drawer task | native |138|false→true|
| acceptance_recovery/episode_000_000_close | Official L1 episode019/step0080 |closed|86|false→true|
| acceptance_recovery/episode_000_001_open | Official L1 episode019/step0080 |open|75|false→true|
| acceptance_recovery/episode_001_000_close | Official L1 episode020/step0080 |closed|82|false→true|
| acceptance_recovery/episode_001_001_open | Official L1 episode020/step0080 |open|76|false→true|

Selected recovery acceptance is4/4 (closed2/2, open2/2). This is a two-scene L1 smoke result, not a score over4803 scenes or stronger L2–L4 recovery. The source failure annotations, perturbations, physical gripper settling, false native goal before learned control and true goal afterward are preserved per episode. The native episode is at `runs/20261105T095735Z_acceptance_policy`; recovery episodes are at `runs/20261105T095735Z_acceptance_recovery`.

The first failed transport attempt remains under `runs/20261105T094843Z_acceptance_policy` with zero executed actions. The successful5-episode run's launcher then exited2 **after saving all results**, because the developer edited the running shell script and Bash resumed at an invalid file offset. Its policy/recovery episode outcomes are valid; its process exit is separately recorded as a launcher failure in `logs/acceptance-retry-launch.log`. Current scripts pass `bash -n`; stable `status` and `list` commands exit0. Code is frozen for the coordinator's fresh end-to-end rerun. Do not infer process success from saved episode success alone.

Independent compact command, fresh seed11, one source scene ×closed/open:

```bash
/mnt/data/Hangxing/benchmarks/scripts/benchmarks libero recovery \
  --suite libero_goal --task open_the_middle_drawer_of_the_cabinet \
  --level L1 --scenes 1 --close-trials 1 --open-trials 1 --seed 11
```

CPU evidence consistency audit:

```bash
.venv-sim/bin/python scripts/verify_evidence.py
```

`logs/evidence-audit.json` checks finite/nonzero learned actions, saved action chunks matching every applied action, image/state dimensions, nonblank views, continuous0.05s physics steps, original source-state integrity and exact initialization-only xy perturbations. Actual GPU inference used JAX0.5.3 on cuda:0, torch2.7.1, RTX4060Ti16GB. Inference occupied about9.3GB VRAM; the first restore sampled up to~9.16GB host RSS before releasing staging buffers. See `logs/policy-memory_54802.jsonl`, `logs/policy-memory_65005.jsonl` and per-run runtime JSON. Only owned policy processes were stopped before releasing the host GPU lock.

## Independent end-to-end verification

The coordinator reran the unified command above with fresh seed11 after the code freeze and observed **process exit0**. Run `20261105T100727Z_recovery` completed one official L1 scene×closed/open: closed81 actions, open107 actions; both had native goal false at recorded-state reconstruction and after settling, then environment_done=true and a fresh native predicate=true after learned control. The owned server stopped and the GPU lock was released. Permanent receipt: `deployment/independent-review.json`. This resolves the earlier launcher process failure; that failed execution remains retained separately.

Across the declared seed7 acceptance and this independent seed11 check there are7 completed successful outcomes:1 native task and6 L1 recovery trials (two unique failure snapshots, two seeds). There is also1 retained zero-action transport failure. These are smoke acceptance outcomes, not a statistically representative recovery benchmark result. `logs/evidence-audit.json` now covers all8 attempted episodes and verifies the7 completed trajectories. Stronger L2–L4 recovery, the full48030-rollout protocol and new-policy training were not validated.

A tooling-cache exception is retained transparently: the initial uv downloader bootstrap briefly used the default `~/.cache/uv` (~62MB,15 packages); all subsequent environment/model/data/runtime caches are project-local. A preserved copy is under `cache/bootstrap-uv-cache`; the original was restored non-overwriting per the coordinator's no-existing-data-cleanup instruction. No global Python packages were installed and this cache is not a shared runtime environment.
