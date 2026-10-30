# Browser demo checks

The demo itself is `docs/website/demo/` (published with the project page): MuJoCo's WebAssembly build
(`@mujoco/mujoco`, the same version as the Python package) steps the task's scene model, onnxruntime-web runs the
policy's decoder, three.js draws it, all from a CDN. The encoder sees only the reference, so its latent is
precomputed for every frame. `tracker.mjs` there is the closed loop, shared with the Node checks here.

Export a checkpoint for the page (the scene model, the decoder with float16 weights, the reset states of the listed
clips and, one file per clip, their latents and reference poses, gzipped). The page offers the models listed in
`MODELS` of `demo/main.mjs`, MS-Human-700 first:

```bash
MS_HUMAN_700_ROOT=<MS-Human-700> uv run ms-export-web --task Mjlab-SuperMimic-Flat-MS-Human-700-Locked-Hands \
    --checkpoint <model.pt> --motion-file data/mshuman700_kit/demo_clips.txt --out docs/website/demo/data/mshuman700
uv run ms-export-web --task Mjlab-SuperMimic-Flat-MyoFullBody --checkpoint <model.pt> \
    --motion-file data/musclemimic/demo_clips.txt --out docs/website/demo/data/myofullbody
```

The two clip lists name the same held-out clips, as each body's retargeting of them.

Export the same clips with the native reference closed loop, and check the JavaScript observation, policy and closed
loop against it:

```bash
uv run ms-export-web --task Mjlab-SuperMimic-Flat-MyoFullBody --checkpoint <model.pt> \
    --motion-file data/musclemimic/demo_clips.txt --out logs/web/demo --reference
cd web && npm install && node test/align.mjs ../logs/web/demo
```

Serve the project page locally (`npm run serve`) and open http://localhost:8000/demo/.
