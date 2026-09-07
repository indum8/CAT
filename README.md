# Installing metadriverse/cat on Ubuntu 26.04 (with a modern Blackwell GPU)

This document walks through installing [CAT](https://github.com/metadriverse/cat) ([CoRL'23] Adversarial Training for Safe End-to-End Driving) on a fresh Ubuntu 26.04 LTS ("resolute") system, including every error hit along the way and how it was fixed. The official README assumes an older Ubuntu/CUDA/conda setup; this guide documents the gap between that and a brand-new system with:

- Ubuntu 26.04 LTS (ships Python 3.14 by default)
- No conda pre-installed
- A very new GPU (NVIDIA RTX PRO 6000 Blackwell, compute capability `sm_120`) that pre-dates CAT's pinned PyTorch build

If you're on an older/more typical Ubuntu LTS (22.04/24.04) with an older NVIDIA GPU, you may be able to follow the official README directly without most of these workarounds.

---

## 0. Official install steps (for reference)

From the [CAT README](https://github.com/metadriverse/cat):

```bash
git clone https://github.com/metadriverse/cat.git
cd cat
# Download (i) the modified MetaDrive fork and (ii) the pre-trained DenseTNT model
# Place densetnt.bin into ./advgen/pretrained/
# Directory should look like:
#   cat/advgen/pretrained/densetnt.bin
#   cat/metadrive/
#   cat/license
#   ...

conda create -n cat python=3.9
conda activate cat
pip install -r requirements.txt
```

`requirements.txt` contents (as of this install):

```
numpy
Panda3D==1.10.13
panda3d-gltf==0.13
panda3d-simplepbr==0.10
pandas==1.5.3
gym==0.22.0
Shapely==1.8.5
seaborn
geopandas==0.12.2
opencv-python==4.7.0.72
pygame==2.1.3
tqdm==4.64.1
pickle5==0.0.11
waymo-open-dataset-tf-2-11-0==1.5.0
torch==1.12.0+cu116
torchvision==0.13.0+cu116
```

---

## 1. No `conda` available

**Symptom:**
```
Command 'conda' not found, did you mean:
  command 'conga' from snap conga (1.5.0)
```

Conda had apparently been removed along with other things during a prior manual cleanup over SSH.

**Fix:** Install Miniconda into the user's home directory (no `sudo` needed):

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
```

> **Gotcha:** `https://repo.anaconda.com/miniconda3/...` (with a `3`) 404s. The correct path is `https://repo.anaconda.com/miniconda/...` (no `3`). If you get a 0-byte file, check with `ls -la miniconda.sh` and `file miniconda.sh` — a failed download typically saves a tiny HTML error page instead of the real ~190MB installer.

```bash
bash miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
conda init bash
# close and reopen your shell, or: source ~/.bashrc
conda --version
```

**Gotcha — Terms of Service error:**
```
CondaToSNonInteractiveError: Terms of Service have not been accepted for the following channels...
```
Fix:
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Then create the environment as per the README:
```bash
cd ~/cat
conda create -n cat python=3.9 -y
conda activate cat
python --version   # should print Python 3.9.x
```

---

## 2. System Python is 3.14, and Ubuntu 26.04 is too new for the deadsnakes PPA to help directly

We initially considered `python3.9-venv` via the `deadsnakes` PPA as an alternative to conda. This does work on Ubuntu 26.04 ("resolute") — the PPA maintains Python 3.7+ for that release. However, we ultimately chose Miniconda instead, since it matches the CAT README exactly and avoids `sudo`/system package changes entirely. If you prefer a lighter-weight `venv`-based setup, deadsnakes is a valid alternative; just be aware Python 3.9 itself is EOL upstream (October 2025), so long-term support isn't guaranteed either way.

---

## 3. `Shapely` / `geopandas` need a `geos` library

`requirements.txt` pins `Shapely==1.8.5` and `geopandas==0.12.2`, both of which require the GEOS C library to build/link against.

**Fix (conda-based, no sudo):**
```bash
conda install -c conda-forge geos -y
```

(If using a plain venv/apt-based system instead, the equivalent is `sudo apt install libgeos-dev -y`.)

---

## 4. `torch==1.12.0+cu116` isn't a normal PyPI package

**Symptom:** `pip install -r requirements.txt` would fail to resolve `torch==1.12.0+cu116` and `torchvision==0.13.0+cu116`, since the `+cu116` build tag only exists on PyTorch's own package index, not PyPI.

**Fix:** Install these two explicitly from PyTorch's index *before* running the main `pip install -r requirements.txt`, then remove them from `requirements.txt` so pip doesn't try (and fail) to resolve them again:

```bash
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116
```

Then edit `requirements.txt` and delete the lines:
```
torch==1.12.0+cu116
torchvision==0.13.0+cu116
pickle5==0.0.11
```
(`pickle5` removed here — see Section 7 for why, and how it was ultimately handled.)

```bash
pip install -r requirements.txt
```

This installed successfully, pulling in `numpy`, `Panda3D`, `Shapely`, `geopandas`, `tensorflow==2.11.0`, `waymo-open-dataset-tf-2-11-0`, and all other pinned dependencies without further build errors.

---

## 5. `ImportError: libtorch_cpu.so: cannot enable executable stack as shared object requires: Invalid argument`

**Symptom**, on `import torch`:
```
ImportError: libtorch_cpu.so: cannot enable executable stack as shared object requires: Invalid argument
```

**Cause:** Modern glibc (2.41+, shipped on newer Ubuntu releases) strictly enforces the ELF "non-executable stack" security marking. PyTorch 1.12's compiled `.so` files predate this convention and lack the marking, so the newer dynamic linker refuses to load them. This is a widely-reported issue affecting old PyTorch/TensorFlow builds on any sufficiently new Linux system, not specific to this GPU or Ubuntu version.

**Fix:** Use `patchelf` (a modern, actively-maintained ELF editor — `execstack` has been dropped from Ubuntu 26.04's repos) to clear the executable-stack flag on every `.so` file under torch's install directory:

```bash
sudo apt install patchelf -y   # already present on some systems

find ~/miniconda3/envs/cat -name "libtorch_cpu.so"
# -> /home/<user>/miniconda3/envs/cat/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so

find ~/miniconda3/envs/cat/lib/python3.9/site-packages/torch -name "*.so*" \
    -exec patchelf --clear-execstack {} \;
```

Verify:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# 1.12.0+cu116
# True
```

---

## 6. GPU too new for pinned PyTorch (`sm_120` / Blackwell not supported by CUDA 11.6 builds)

**Symptom**, as a warning first:
```
UserWarning:
NVIDIA RTX PRO 6000 Blackwell Workstation Edition with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86.
```

...followed by the process appearing to hang (no error, no progress) when it actually tried to run a CUDA kernel — likely while loading the DenseTNT model onto the GPU.

**Cause:** PyTorch 1.12+cu116 (from 2022) was compiled with kernels only for GPU architectures up through `sm_86`. The RTX PRO 6000 Blackwell is `sm_120` — a compute capability that didn't exist yet when this PyTorch build was compiled. There is no compiled kernel for it, so any real GPU operation fails (and can hang rather than error cleanly, depending on where in the CUDA init path it occurs).

**Decision:** Rather than upgrading PyTorch (which risks breaking CAT's code against newer/changed APIs — we'd already seen deprecation warnings for `pretrained=True` in torchvision), we chose to force CPU-only execution. This is slower but avoids introducing new compatibility risk into a codebase written for torch 1.12's APIs.

**Fix — disable CUDA visibility at the environment level:**
```bash
export CUDA_VISIBLE_DEVICES=""
python -c "import torch; print(torch.cuda.is_available())"
# False
```

This alone isn't sufficient, however — see Section 8, since CAT's code hardcodes GPU device references in several places rather than checking availability.

---

## 7. `ModuleNotFoundError: No module named 'pickle5'`

**Symptom**, once `requirements.txt` was installed without `pickle5` (deliberately excluded — see Section 4):
```python
File "/home/<user>/cat/advgen/structs.py", line 2, in <module>
    import pickle5 as pickle
ModuleNotFoundError: No module named 'pickle5'
```

**Cause:** `pickle5` is a backport of Python 3.8+'s pickle protocol 5 for *older* Python versions. On Python 3.9 the standard library's built-in `pickle` module already includes protocol 5 — the backport is functionally redundant. However, CAT's code explicitly does `import pickle5 as pickle`, so simply having native pickle support isn't enough; the package name itself must resolve.

**Fix:** Just install it directly — on this system it installed cleanly with no build issues (some older/newer Python combinations do see build failures for this package, in which case patching the import to plain `import pickle` is the fallback):
```bash
pip install pickle5
```

---

## 8. `RuntimeError: No CUDA GPUs are available` — hardcoded GPU device references in CAT's code

With `CUDA_VISIBLE_DEVICES=""` set, several places in CAT's own code crashed because they hardcode a CUDA device rather than checking `torch.cuda.is_available()` first. Three separate hardcoded references were found and patched:

### 8a. Model creation — `advgen/adv_generator.py`
```python
# before
self.model = VectorNet(args).to(0)

# after
self.model = VectorNet(args).to(0 if torch.cuda.is_available() else 'cpu')
```

### 8b. Checkpoint loading — `advgen/adv_generator.py`
```python
# before
self.model.load_state_dict(torch.load('./advgen/pretrained/densetnt.bin'))

# after
self.model.load_state_dict(torch.load(
    './advgen/pretrained/densetnt.bin',
    map_location=None if torch.cuda.is_available() else 'cpu'
))
```

### 8c. Inference call — `advgen/adv_generator.py` AND `advgen/adv_generator_rule.py`
Both files contained the identical pattern:
```python
# before
pred_trajectory, pred_score, _ = self.model(batch_data[0], 'cuda')

# after
pred_trajectory, pred_score, _ = self.model(
    batch_data[0],
    'cuda' if torch.cuda.is_available() else 'cpu'
)
```

**How these were found:** rather than fixing errors one at a time as they surfaced, a repo-wide search was run after the second occurrence to catch any remaining instances:
```bash
grep -rn "'cuda'" ~/cat/advgen/
```

> **Note for future scripts:** `cat_RLtrain.py` and any other entry points not yet run may contain the same pattern. Run the same `grep -rn "\.to(0)\|\.cuda()\|'cuda'"` check against any new script before running it.

---

## 9. `pygame.error: No available video device`

**Symptom**, once the model/CUDA issues were resolved and the scenario loop started running:
```
File ".../metadrive/obs/top_down_renderer.py", line 217, in __init__
    self._render_canvas = pygame.display.set_mode(self._render_size)
pygame.error: No available video device
```

**Cause:** `cat_advgen.py` calls `env.render(mode="top_down", ...)` to produce a top-down visualization, which needs a display — real or virtual — even though `use_render: False` is set in the environment config for the main simulation (that flag controls 3D rendering, not this 2D top-down debug view).

**Fix:** Run a virtual framebuffer (Xvfb) and point the session at it:
```bash
sudo apt install xvfb -y
Xvfb :99 -screen 0 1400x900x24 &
export DISPLAY=:99
```

**Gotcha — stale lock file:**
```
Fatal server error:
(EE) Server is already active for display 99
```
This can mean either (a) a genuinely stale lock from a previous crashed attempt, or (b) an Xvfb instance that's actually still running and healthy. Check before blindly removing the lock:
```bash
ps aux | grep -i xvfb
ls -la /tmp/.X11-unix/
```
In this case, a healthy Xvfb from several days earlier was still running on `:99` — no new instance was needed, just `export DISPLAY=:99` to use the existing one.

If you do need to start fresh and confirm no process is actually using it:
```bash
rm -f /tmp/.X99-lock
Xvfb :99 -screen 0 1400x900x24 &
```

---

## 10. `KeyError: '500'` — stale object reference during scenario reset (MetaDrive internal bug)

**Symptom:** After successfully running ~78 of 500 scenarios (visible progress bar, `avg_attack_success_rate` populating, occasional `INFO:root:Episode ended! Reason: arrive_dest.` lines), the run crashed:

```
File "/home/<user>/cat/cat_advgen.py", line 63, in <module>
    env.reset(force_seed=i)
File "/home/<user>/cat/metadrive/envs/base_env.py", line 401, in reset
    self.engine.reset()
File "/home/<user>/cat/metadrive/engine/base_engine.py", line 273, in reset
    manager.before_reset()
File "/home/<user>/cat/metadrive/manager/scenario_light_manager.py", line 21, in before_reset
    super(ScenarioLightManager, self).before_reset()
File "/home/<user>/cat/metadrive/manager/base_manager.py", line 41, in before_reset
    self.clear_objects([object_id for object_id in self.spawned_objects.keys()])
File "/home/<user>/cat/metadrive/manager/base_manager.py", line 78, in clear_objects
    exclude_objects = self.engine.clear_objects(*args, **kwargs)
File "/home/<user>/cat/metadrive/engine/base_engine.py", line 194, in clear_objects
    exclude_objects = {obj_id: self._spawned_objects[obj_id] for obj_id in filter}
KeyError: '500'
```

**Cause:** This is unrelated to any environment/dependency setup issue — it's an internal bug in MetaDrive's object lifecycle management. `BaseEngine.clear_objects()` (in `metadrive/engine/base_engine.py`) is handed a list of object IDs to destroy (here, traffic light objects tracked by `ScenarioLightManager`), and it assumes every ID in that list still exists in `self._spawned_objects`. In practice, at scenario-reset boundaries, an object can already have been removed by another manager or process before this cleanup runs — leaving a stale ID reference that triggers a hard `KeyError` instead of being silently skipped.

The exact scenario/frame this happens on is not deterministic in general — it manifested here at scenario index ~78 with the 500 pre-packaged scenarios, but the underlying condition (stale ID left in a `filter` list) can in principle occur on any scenario, depending on light/object spawn timing.

**Fix:** Patch `clear_objects()` in `metadrive/engine/base_engine.py` to skip IDs that no longer exist in `_spawned_objects`, rather than crash:

```python
# before (line ~194)
exclude_objects = {obj_id: self._spawned_objects[obj_id] for obj_id in filter}

# after
exclude_objects = {obj_id: self._spawned_objects[obj_id] for obj_id in filter if obj_id in self._spawned_objects}
```

This is a minimal, defensive change — it does not alter physics, scenario logic, or attack-generation behavior. It only prevents a crash when the cleanup step encounters an object ID that's already gone, which is a legitimate state to handle gracefully (the object is gone either way; the intent of the line was to destroy it, which is a no-op if it no longer exists).

Applied via a small Python patch script (safer than `sed` here, since the target line contains braces/colons/brackets that are awkward to escape correctly in shell one-liners):

```bash
cat > /tmp/patch_clear_objects.py << 'EOF'
path = "/home/<user>/cat/metadrive/engine/base_engine.py"

old = "exclude_objects = {obj_id: self._spawned_objects[obj_id] for obj_id in filter}"
new = "exclude_objects = {obj_id: self._spawned_objects[obj_id] for obj_id in filter if obj_id in self._spawned_objects}"

with open(path) as f:
    content = f.read()

if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("patched")
else:
    print("pattern not found")
EOF
python3 /tmp/patch_clear_objects.py
```

Verify the change landed:
```bash
sed -n '190,196p' ~/cat/metadrive/engine/base_engine.py
```

> **Note:** This is a fork-local patch to the *modified* MetaDrive copy bundled with CAT (`~/cat/metadrive/`), not to a pip-installed MetaDrive package. If you re-download or re-clone the modified MetaDrive fork from CAT's README link, you will need to reapply this patch.

---

## 11. `AttributeError: 'LVehicle' object has no attribute 'status'` — another stale object reference (further into the run)

**Symptom:** After the Section 10 patch, the run progressed much further (168 of 500 scenarios, up from crashing at ~78 before) but then hit a related but distinct crash:

```
File "/home/<user>/cat/cat_advgen.py", line 53, in <module>
    o, r, done, info = env.step([1.0, 0.]) # replace it with your own controller
File "/home/<user>/cat/metadrive/envs/scenario_env.py", line 148, in step
    ret = super(ScenarioEnv, self).step(actions)
File "/home/<user>/cat/metadrive/envs/base_env.py", line 296, in step
    engine_info = self._step_simulator(actions)
File "/home/<user>/cat/metadrive/envs/base_env.py", line 326, in _step_simulator
    scene_manager_after_step_infos = self.engine.after_step()
File "/home/<user>/cat/metadrive/engine/base_engine.py", line 383, in after_step
    new_step_info = manager.after_step(*args, **kwargs)
File "/home/<user>/cat/metadrive/manager/agent_manager.py", line 264, in after_step
    step_infos = self.for_each_active_agents(lambda v: v.after_step())
File "/home/<user>/cat/metadrive/manager/agent_manager.py", line 425, in for_each_active_agents
    ret[k] = func(v, *args, **kwargs)
File "/home/<user>/cat/metadrive/manager/agent_manager.py", line 264, in <lambda>
    step_infos = self.for_each_active_agents(lambda v: v.after_step())
File "/home/<user>/cat/metadrive/component/vehicle/base_vehicle.py", line 322, in after_step
    self._state_check()
File "/home/<user>/cat/metadrive/component/vehicle/base_vehicle.py", line 828, in _state_check
    if light.status == MetaDriveType.LIGHT_GREEN:
AttributeError: 'LVehicle' object has no attribute 'status'
```

**Cause:** Same family of bug as Section 10 — a stale/mismatched object reference — but in a different code path. `BaseVehicle._state_check()` (in `metadrive/component/vehicle/base_vehicle.py`) runs physics contact detection each step. When a contact's collision node is named/tagged as `MetaDriveType.TRAFFIC_LIGHT`, the code calls `get_object_from_node(node)` expecting back a traffic-light object, then immediately accesses `.status` on it. Due to object ID recycling (the same underlying mechanism flagged in Section 10 — MetaDrive's object pooling in `BaseEngine.spawn_object()` reuses object slots/IDs across resets for efficiency), the node can end up bound to a *vehicle* object (`LVehicle`) instead of a light object by the time this check runs, which has no `.status` attribute.

**Fix:** Add a defensive guard right after the lookup, following the same pattern already used elsewhere in this same function for other "didn't add, skip this contact" cases:

```python
# before
elif name == MetaDriveType.TRAFFIC_LIGHT:
    light = get_object_from_node(node)
    if light.status == MetaDriveType.LIGHT_GREEN:
        ...

# after
elif name == MetaDriveType.TRAFFIC_LIGHT:
    light = get_object_from_node(node)
    if not hasattr(light, "status"):
        # Stale/mismatched object reference: node is tagged as TRAFFIC_LIGHT
        # but the object currently bound to it is not a light (can happen
        # due to object ID recycling between resets). Skip safely.
        continue
    if light.status == MetaDriveType.LIGHT_GREEN:
        ...
```

This mirrors the existing `LIGHT_UNKNOWN` case a few lines below, which already uses `continue` to skip a contact that can't be meaningfully classified — so the fix is stylistically consistent with the surrounding code, not a new pattern being introduced.

Applied via a small Python patch script (same reasoning as Section 10 — the target block spans multiple lines with specific indentation, which is more reliable to match exactly in Python than to construct as a `sed` one-liner):

```bash
cat > /tmp/patch_light_status.py << 'PYEOF'
path = "/home/<user>/cat/metadrive/component/vehicle/base_vehicle.py"

old = """            elif name == MetaDriveType.TRAFFIC_LIGHT:
                light = get_object_from_node(node)
                if light.status == MetaDriveType.LIGHT_GREEN:"""

new = """            elif name == MetaDriveType.TRAFFIC_LIGHT:
                light = get_object_from_node(node)
                if not hasattr(light, "status"):
                    # Stale/mismatched object reference: node is tagged as TRAFFIC_LIGHT
                    # but the object currently bound to it is not a light (can happen
                    # due to object ID recycling between resets). Skip safely.
                    continue
                if light.status == MetaDriveType.LIGHT_GREEN:"""

with open(path) as f:
    content = f.read()

if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("patched")
else:
    print("pattern not found")
PYEOF
python3 /tmp/patch_light_status.py
```

Verify:
```bash
sed -n '822,835p' ~/cat/metadrive/component/vehicle/base_vehicle.py
```

> **Note:** Like Section 10, this patches the fork-local copy of MetaDrive bundled with CAT. Reapply if you re-clone/re-download.

> **Observation:** Both Sections 10 and 11 stem from the same root behavior (MetaDrive's object-recycling pool occasionally leaving stale ID-to-object bindings across scenario resets). If further runs surface additional `KeyError`/`AttributeError` crashes tied to object lookups (`get_object_from_node`, `_spawned_objects[...]`, etc.), the same "check before use, skip on mismatch" defensive pattern is the appropriate fix — search for other unguarded lookups of this shape rather than patching each occurrence reactively as it's hit.

---

## 12. `KeyError: '385'` — same unguarded-lookup bug, different call site (`get_objects()`)

**Symptom:** Progressed further still (168/500 again, same point as Section 11's original crash) but with a different traceback, this time originating one call earlier:

```
File "/home/<user>/cat/metadrive/component/vehicle/base_vehicle.py", line 827, in _state_check
    light = get_object_from_node(node)
File "/home/<user>/cat/metadrive/utils/utils.py", line 191, in get_object_from_node
    return get_object(ret)[ret]
File "/home/<user>/cat/metadrive/engine/engine_utils.py", line 23, in get_object
    return get_engine().get_objects([object_name])
File "/home/<user>/cat/metadrive/engine/base_engine.py", line 164, in get_objects
    return {id: self._spawned_objects[id] for id in filter}
KeyError: '385'
```

**Cause:** `BaseEngine.get_objects()` (a sibling method to `clear_objects()`, which was already patched in Section 10) has the exact same unguarded dict-comprehension pattern:

```python
elif isinstance(filter, (list, tuple)):
    return {id: self._spawned_objects[id] for id in filter}
```

This is called by `get_object()` (`engine_utils.py`), which is called by `get_object_from_node()` (`utils.py`), which is called by `BaseVehicle._state_check()` — the same call chain from Section 11, just failing one level deeper than the `hasattr` guard we'd already added there.

**Fix:** Apply the same defensive pattern as Section 10's `clear_objects()` fix, this time to `get_objects()`:

```python
# before
elif isinstance(filter, (list, tuple)):
    return {id: self._spawned_objects[id] for id in filter}

# after
elif isinstance(filter, (list, tuple)):
    return {id: self._spawned_objects[id] for id in filter if id in self._spawned_objects}
```

Applied the same way as previous patches (Python patch script for reliable multi-line matching):

```bash
cat > /tmp/patch_get_objects.py << 'PYEOF'
path = "/home/<user>/cat/metadrive/engine/base_engine.py"

old = """        elif isinstance(filter, (list, tuple)):
            return {id: self._spawned_objects[id] for id in filter}
        elif callable(filter):
            res = dict()
            for id, obj in self._spawned_objects.items():
                if filter(obj):
                    res[id] = obj
            return res"""

new = """        elif isinstance(filter, (list, tuple)):
            return {id: self._spawned_objects[id] for id in filter if id in self._spawned_objects}
        elif callable(filter):
            res = dict()
            for id, obj in self._spawned_objects.items():
                if filter(obj):
                    res[id] = obj
            return res"""

with open(path) as f:
    content = f.read()

if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("patched")
else:
    print("pattern not found")
PYEOF
python3 /tmp/patch_get_objects.py
```

Verify:
```bash
sed -n '160,170p' ~/cat/metadrive/engine/base_engine.py
```

**Also worth noting:** at this point, `get_object_from_node()`'s final line (`return get_object(ret)[ret]`) was additionally hardened, so that if the ID isn't present in whatever `get_objects()` returns, it returns `None` (consistent with the early-return-`None` path already present earlier in that same function) rather than crashing on the `[ret]` index lookup:

```python
# before
if is_road:
    return get_engine().current_map.road_network.get_lane(ret)
else:
    return get_object(ret)[ret]

# after
if is_road:
    return get_engine().current_map.road_network.get_lane(ret)
else:
    result = get_object(ret)
    return result[ret] if ret in result else None
```

This second change ended up not being strictly necessary once `get_objects()` itself was fixed (since a fixed `get_objects()` never omits a requested-but-missing key from a dict in a way that would crash `[ret]`... actually it does, if `ret` isn't in the returned dict at all — so this guard is still meaningful defense-in-depth). Both patches were applied and kept.

> **Note:** Fork-local patches — reapply if you re-clone/re-download the modified MetaDrive fork.

---

## 13. `AttributeError: 'NoneType' object has no attribute 'top_down_width'` — a fix regression, and why it happened

**Symptom:** After Sections 10–12 resolved the MetaDrive-internal crashes, a new failure appeared in CAT's own code (not MetaDrive's):

```
File "/home/<user>/cat/cat_advgen.py", line 45, in <module>
    adv_generator.before_episode(env)
File "/home/<user>/cat/advgen/adv_generator.py", line 158, in before_episode
    adv_info = dict(w=adv_obj.top_down_width,l=adv_obj.top_down_length),
AttributeError: 'NoneType' object has no attribute 'top_down_width'
```

**Cause — this one is a regression introduced by our own Section 12 fix, worth understanding clearly:**

The original code in `advgen/adv_generator.py` was:
```python
ego_obj = self.env.engine.get_objects(['default_agent']).get('default_agent')
try:
    adv_obj = self.env.engine.get_objects([adv_agent]).get(adv_agent)
except:
    adv_obj = ego_obj
```

The intent is clear: if the designated opponent vehicle (`adv_agent`) doesn't exist for this scenario, fall back to using the ego vehicle's info instead of crashing. **Before Section 12's fix**, `get_objects()` raised a `KeyError` when asked for a nonexistent ID — and that `KeyError` was exactly what this `try/except` was designed to catch, triggering the fallback.

**After Section 12's fix**, `get_objects()` was changed to silently omit missing IDs and return a dict without them, rather than raising. This was the correct fix for the crash it targeted — but it had a side effect here: `.get(adv_agent)` on the (now silently incomplete) returned dict just returns `None` — no exception is raised, so the `except:` block never triggers, and the intended fallback never happens. `adv_obj` stays `None`, causing this crash.

This is a good illustration of why defensive patches to shared utility functions can have non-local effects — fixing one call site's crash-on-missing-key behavior can silently change the behavior of every other caller that happened to rely on the old crash-then-catch pattern.

**Fix:** Make the fallback explicit, checking for `None` directly instead of depending on an exception that no longer occurs:

```python
# before
ego_obj = self.env.engine.get_objects(['default_agent']).get('default_agent')
try:
    adv_obj = self.env.engine.get_objects([adv_agent]).get(adv_agent)
except:
    adv_obj = ego_obj

# after
ego_obj = self.env.engine.get_objects(['default_agent']).get('default_agent')
adv_obj = self.env.engine.get_objects([adv_agent]).get(adv_agent)
if adv_obj is None:
    # Opponent vehicle not present for this scenario (or a stale/
    # missing object reference) -- fall back to the ego vehicle's
    # info, matching the original try/except's intended behavior.
    adv_obj = ego_obj
```

Applied via patch script:
```bash
cat > /tmp/patch_adv_obj_fallback.py << 'PYEOF'
path = "/home/<user>/cat/advgen/adv_generator.py"

old = """            ego_obj = self.env.engine.get_objects(['default_agent']).get('default_agent')
            try:
                adv_obj = self.env.engine.get_objects([adv_agent]).get(adv_agent)
            except:
                adv_obj = ego_obj"""

new = """            ego_obj = self.env.engine.get_objects(['default_agent']).get('default_agent')
            adv_obj = self.env.engine.get_objects([adv_agent]).get(adv_agent)
            if adv_obj is None:
                # Opponent vehicle not present for this scenario (or a stale/
                # missing object reference) -- fall back to the ego vehicle's
                # info, matching the original try/except's intended behavior.
                adv_obj = ego_obj"""

with open(path) as f:
    content = f.read()

if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("patched")
else:
    print("pattern not found")
PYEOF
python3 /tmp/patch_adv_obj_fallback.py
```

Verify:
```bash
sed -n '146,158p' ~/cat/advgen/adv_generator.py
```

> **Lesson for future patching:** when changing a shared utility's error-handling behavior (raise → silent skip, or vice versa), grep for other callers of that utility that use `try/except` around it — they may depend on the exception being raised, and will need the same "check explicitly instead of relying on the exception" treatment.

---

## 14. Recurring session-scoped environment loss (`CUDA_VISIBLE_DEVICES`, `DISPLAY`)

**Symptom:** After a fresh SSH reconnect, both of the following regressed even though they'd been fixed earlier in the same overall setup:
- The CUDA/Blackwell crash from Section 6 recurred (`RuntimeError: CUDA error: no kernel image is available for execution on the device`), because `export CUDA_VISIBLE_DEVICES=""` only applies to the shell session it was run in.
- The Xvfb/display crash from Section 9 recurred (`pygame.error: No available video device`), because the backgrounded `Xvfb :99 ...` process either wasn't still running, or `$DISPLAY` wasn't set in the new session.

**Fix — make `CUDA_VISIBLE_DEVICES` persistent to the conda environment itself** (rather than the shell), so it's automatically set every time `conda activate cat` runs, on any future session:

```bash
conda activate cat
conda env config vars set CUDA_VISIBLE_DEVICES=""
conda deactivate
conda activate cat
```

Verify:
```bash
echo $CUDA_VISIBLE_DEVICES   # should print an empty line
python -c "import torch; print(torch.cuda.is_available())"   # should print False
```

**For Xvfb**, there's no equivalent "attach to conda env" trick — a background process doesn't survive an SSH session ending unless it's detached properly (e.g. run inside `screen`/`tmux`, or with `nohup ... &` and `disown`). At minimum, check and restart per-session as needed:
```bash
ps aux | grep -i xvfb
# if nothing shown:
Xvfb :99 -screen 0 1400x900x24 &
export DISPLAY=:99
```

For a long training/generation run you plan to leave running unattended, it's worth starting both the Xvfb process and the actual `python cat_advgen.py` run inside a `tmux` or `screen` session, so neither depends on the SSH connection staying alive:
```bash
tmux new -s cat_run
# inside tmux:
conda activate cat
Xvfb :99 -screen 0 1400x900x24 &
export DISPLAY=:99
python cat_advgen.py
# detach with Ctrl+B then D; reattach later with: tmux attach -t cat_run
```

---

## 15. Considered but not used: upgrading to a Blackwell-native PyTorch (CUDA path)

For completeness, since GPU acceleration was considered as an alternative to the CPU-only approach used throughout this document:

- **PyTorch 2.7.0** was the first stable release with native `sm_120` (Blackwell) support, via pre-built CUDA 12.8 wheels (with updated cuDNN, NCCL, and Triton). PyTorch 2.11.0 is the current recommended stable version supporting Blackwell as of this writing.
- **Decision: not pursued.** CAT's code (`advgen/modeling/vectornet.py`, `adv_generator.py`, etc.) is written against torch 1.12.0's API, a jump of several major versions. Concrete known risks if upgrading:
  - `torchvision.models(pretrained=True)` — already seen as a deprecation warning on this setup — is removed entirely in torchvision 0.15+, which would ship alongside a modern torch 2.x.
  - `torch.load()`'s default `weights_only` behavior changed in newer torch versions, which can silently break loading older-format checkpoints like `densetnt.bin` unless handled explicitly.
  - Potential further NumPy 2.0-related incompatibilities, compounding the existing `gym`/NumPy 2.0 warning already present in this stack.
- Given the CPU-only path was already working end-to-end (Section 16 confirms a full clean run), the safer choice was made to stay on CPU rather than risk introducing new, harder-to-diagnose breakage for a speed gain. If pursuing GPU acceleration later, treat it as an isolated follow-up effort with its own testing pass — not a drop-in swap.

---

## 16. Data preparation — using the pre-packaged 500 scenarios

CAT provides 500 pre-processed Waymo Open Motion Dataset (WOMD) v1.1 scenarios so you don't need to run the full tfrecord conversion pipeline (`scripts/covert_WOMD_to_MD.py` + `scripts/select_cases.py`) to get started.

1. Download the folder from the [README's Google Drive link](https://drive.google.com/drive/folders/1xVQ84pF5clVtKw6d4NCC-0mYbo4cIZ_a) via your browser (Drive requires interactive auth, so this can't be scripted from a headless remote box).
2. Transfer it to the remote machine:
   ```bash
   scp -r <local_downloaded_folder> <user>@<host>:~/cat/
   ```
3. **Important:** the exact expected path is hardcoded in `cat_advgen.py`:
   ```python
   "data_directory": './raw_scenes_500',
   ```
   So the downloaded folder must end up at exactly `~/cat/raw_scenes_500/`. Rename/move or unzip accordingly:
   ```bash
   mv <downloaded_folder_name> ~/cat/raw_scenes_500
   # or
   unzip <file>.zip -d ~/cat/raw_scenes_500
   ```

---

## 17. Final working setup

Once all the above was applied, `python cat_advgen.py` completed a **full, clean 500/500 scenario run** (CPU-only, with a virtual display for the top-down renderer), finishing in ~31 minutes with a final `avg_attack_success_rate=0.9` and `avg_compute_time=0.358`s per scenario — no crashes, no hangs, no manual intervention mid-run.

### Summary of environment
- Ubuntu 26.04 LTS ("resolute")
- Miniconda3, installed to `~/miniconda3` (user-local, no sudo)
- Conda env `cat`, Python 3.9.25
- `torch==1.12.0+cu116` / `torchvision==0.13.0+cu116`, patched `.so` files via `patchelf --clear-execstack`
- CPU-only execution forced via `CUDA_VISIBLE_DEVICES=""` (GPU is `sm_120` Blackwell, unsupported by this PyTorch build)
- Three hardcoded CUDA device references patched in `advgen/adv_generator.py` and `advgen/adv_generator_rule.py`
- `pickle5` installed directly (worked fine on this Python 3.9 build)
- Xvfb virtual display on `:99` for the top-down pygame renderer
- 500 pre-packaged WOMD scenarios placed at `~/cat/raw_scenes_500/`
- Four defensive patches in the fork-local MetaDrive/CAT copy for object-lifecycle edge cases:
  - `metadrive/engine/base_engine.py`'s `clear_objects()` (Section 10)
  - `metadrive/component/vehicle/base_vehicle.py`'s `_state_check()` (Section 11)
  - `metadrive/engine/base_engine.py`'s `get_objects()` + `metadrive/utils/utils.py`'s `get_object_from_node()` (Section 12)
  - `advgen/adv_generator.py`'s `before_episode()` opponent-vehicle fallback (Section 13 — a regression fix for Section 12's own side effect)
- `CUDA_VISIBLE_DEVICES=""` made persistent to the `cat` conda environment (Section 14), rather than relying on a per-session `export`

### Known limitations of this setup
- **CPU-only inference/training, by choice.** DenseTNT inference and MetaDrive physics both run in software. A CUDA path exists (PyTorch 2.7+ supports Blackwell natively — see Section 15) but was deliberately not pursued, to avoid risking breakage in CAT's torch-1.12-era code for a speed gain. The CPU path is fully confirmed working end-to-end.
- **The fork-local patches (Sections 10–13) are local to this checkout.** If you re-download the modified MetaDrive fork fresh, or clone CAT again elsewhere, you'll need to reapply all four.
- **Sessions are not self-healing.** `CUDA_VISIBLE_DEVICES` is now persistent to the conda environment (Section 14), but Xvfb/`$DISPLAY` still needs to be checked/restarted per SSH session unless run inside `tmux`/`screen` (recommended for any long unattended run — see Section 14).
- The `deadsnakes`/`venv` route was explored but not used in the final setup — documented here in case conda is undesirable in your environment.
- **Confirmed working end-to-end:** a complete, uninterrupted 500/500 scenario run via `python cat_advgen.py` was achieved on this setup (~31 minutes, `avg_attack_success_rate=0.9`).

---

## Quick reference: all commands in order

```bash
# 1. Clone
git clone https://github.com/metadriverse/cat.git
cd cat

# 2. Install Miniconda (user-local, no sudo)
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
conda init bash   # then reopen shell / source ~/.bashrc

# 3. Accept conda ToS if prompted
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 4. Create env
cd ~/cat
conda create -n cat python=3.9 -y
conda activate cat

# 5. System deps
conda install -c conda-forge geos -y

# 6. Torch (from PyTorch's own index, not PyPI)
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# 7. Remove torch/torchvision/pickle5 lines from requirements.txt, then:
pip install -r requirements.txt
pip install pickle5

# 8. Fix executable-stack error on new glibc
sudo apt install patchelf -y
find ~/miniconda3/envs/cat/lib/python3.9/site-packages/torch -name "*.so*" \
    -exec patchelf --clear-execstack {} \;

# 9. Force CPU-only (GPU too new for this PyTorch build)
export CUDA_VISIBLE_DEVICES=""

# 10. Patch hardcoded CUDA references in advgen/adv_generator.py
#     and advgen/adv_generator_rule.py (see Section 8 for exact diffs)

# 11. Virtual display for top-down renderer
sudo apt install xvfb -y
Xvfb :99 -screen 0 1400x900x24 &
export DISPLAY=:99

# 12. Place pre-packaged scenario data
#     (download from README's Google Drive link, scp to remote, place at)
#     ~/cat/raw_scenes_500/

# 13. Patch metadrive/engine/base_engine.py's clear_objects() to skip
#     stale object IDs during scenario reset (see Section 10 for exact diff)

# 14. Patch metadrive/component/vehicle/base_vehicle.py's _state_check()
#     to skip stale/mismatched traffic-light object references
#     (see Section 11 for exact diff)

# 15. Patch metadrive/engine/base_engine.py's get_objects() and
#     metadrive/utils/utils.py's get_object_from_node() the same way
#     (see Section 12 for exact diffs)

# 16. Patch advgen/adv_generator.py's before_episode() opponent-vehicle
#     fallback, which broke as a side effect of step 15's fix
#     (see Section 13 for exact diff)

# 17. Make CUDA_VISIBLE_DEVICES persistent to the conda env (see Section 14):
conda env config vars set CUDA_VISIBLE_DEVICES=""
conda deactivate && conda activate cat

# 18. Run (confirmed: completes a full 500/500 scenario run, ~31 min)
python cat_advgen.py
```
