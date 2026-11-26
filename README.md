# MiniMax-H3 ncnn Vulkan

ncnn implementation of MiniMax-H3 for video and stereo audio generation from text, keyframes or multimodal references. [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)

h3-ncnn-vulkan uses ncnn for inference. [ncnn](https://github.com/Tencent/ncnn)

## Features

<table>
<tr>
<td>

- **BF16** text encoder and DiT, FP32 vision and VAEs
- **Vulkan** GPU acceleration
- **Portable** standalone executable
- **No** CUDA, PyTorch, Python or FFmpeg dependency for inference
- Automatic weight streaming and tiled video VAE inference

</td>
<td>

- Text, first and last frames, or multimodal references
- **Dynamic** resolution and frame count
- **Stereo audio** generation
- Intermediate keyframes for longer videos
- **LoRA** adapters, portable modulation banks and optional rough modes
- Lossy **WebM** and lossless **Matroska** output

</td>
</tr>
</table>

## Download

Download the Linux x86-64 (glibc 2.28 or newer), macOS universal (Apple silicon and Intel), or Windows x86-64 ZIP from [GitHub Releases](https://github.com/nihui/h3-ncnn-vulkan/releases). For a local build, see [Build from Source](#build-from-source).

The release workflow builds all three packages from source and collects them in a draft release. [Release workflow](.github/workflows/release.yml)

### prepare model files

Run the examples from the package root. Add the package’s `bin` directory to your `PATH` in the current shell:

- Linux and macOS: `export PATH="$PWD/bin:$PATH"`
- Windows (PowerShell): `$env:Path = "$PWD\bin;$env:Path"`

The default model path is `models/h3` in the package root. Set `-m` to use a different model directory.

```
bin/
    h3-ncnn-vulkan
models/
    h3/
        LICENSE
        NOTICE
        processor/
        text_encoder/
        vae/
        audio_vae/
        transformer/
```

Download the files for text-to-video-and-audio (T2VA) from the package root. The default 50-point schedule uses the prepared BF16 modulation bank. This set takes about 128 GB. [h3 model folder](https://huggingface.co/nihui-szyl/h3-ncnn/tree/main/h3)

```shell
hf download nihui-szyl/h3-ncnn --local-dir models \
    --include 'h3/LICENSE' \
    --include 'h3/NOTICE' \
    --include 'h3/processor/*' \
    --include 'h3/text_encoder/*' \
    --include 'h3/vae/decoder.ncnn.*' \
    --include 'h3/audio_vae/decoder.ncnn.*' \
    --include 'h3/transformer/fl2va/transformer-core.ncnn.*' \
    --include 'h3/transformer/fl2va/fixed-build/modulations.ncnn.*' \
    --include 'h3/transformer/fl2va/fixed-modulations-bf16-steps50-none.bin'
```

Keep the `fixed-build` files. The runtime uses them to validate the prepared bank and to compute modulations for other schedules. The [first generation example](#example-command) uses this download.

<details>
<summary>Additional model files for FL2VA, Ref2VA and LoRA</summary>

For FL2VA with one or two image keyframes, add the vision encoder, spatial VAE encoder and video bank. This adds about 2.9 GB to the T2VA download.

```shell
hf download nihui-szyl/h3-ncnn --local-dir models \
    --include 'h3/vision/*' \
    --include 'h3/vae/encoder_spatial.ncnn.*' \
    --include 'h3/transformer/fl2va/fixed-modulations-bf16-steps50-video.bin'
```

For the Ref2VA example with an image and audio, add the Ref2VA core, producer and matching bank alongside the encoders. This adds about 70 GB to the T2VA download.

```shell
hf download nihui-szyl/h3-ncnn --local-dir models \
    --include 'h3/vision/*' \
    --include 'h3/vae/encoder_spatial.ncnn.*' \
    --include 'h3/audio_vae/encoder.ncnn.*' \
    --include 'h3/transformer/ref2va/transformer-core.ncnn.*' \
    --include 'h3/transformer/ref2va/fixed-build/modulations.ncnn.*' \
    --include 'h3/transformer/ref2va/fixed-modulations-bf16-steps50-both.bin'
```

Use the bank that matches your references as described under [Modulation cache](#modulation-cache). An audio reference needs `audio_vae/encoder.ncnn.*`. A video reference needs `vae/encoder_temporal.ncnn.*`.

pi-H3 also needs the shared `transformer/pi-h3.safetensors`. The optional two- and four-forward banks in `transformer/fl2va/` cover text-only and keyframe inputs. The LoRA adapters themselves are available from their publishers. [LoRA](#lora)

The complete model folder takes about 206 GB.

```shell
hf download nihui-szyl/h3-ncnn --local-dir models --include 'h3/*'
```

</details>

## About MiniMax-H3

MiniMax-H3 generates video and audio together. This implementation supports text to video and audio (T2VA), first and last frames to video and audio (FL2VA), and image, video or audio references (Ref2VA).

T2VA and FL2VA share the DiT in `transformer/fl2va/`. Ref2VA uses `transformer/ref2va/`.

## Usages

### requirements

- A Vulkan-capable GPU and driver.
- Disk space for the model files used by the selected workflow.
- Additional disk space for streamed weights. Set `H3_WEIGHT_TMPDIR` to a disk-backed directory with tens of GiB free when paging.
- RAM and VRAM for the requested resolution, frame count, prompt and references.

The table gives estimated planning budgets with disk weight streaming, rounded up to the next multiple of 128 MiB.

| Output resolution | Frames | Duration | RAM budget (MiB) | VRAM budget (MiB) |
|---|---:|---:|---:|---:|
| 640 × 352 | 56 | 2.33 s | 1664 | 1792 |
| 640 × 352 | 124 | 5.17 s | 1664 | 3072 |
| 640 × 352 | 243 | 10.13 s | 1664 | 5376 |
| 832 × 480 | 56 | 2.33 s | 1664 | 2688 |
| 832 × 480 | 124 | 5.17 s | 1664 | 4992 |
| 832 × 480 | 243 | 10.13 s | 1664 | 8704 |
| 1024 × 1024 | 124 | 5.17 s | 1664 | 11392 |
| 1280 × 736 | 56 | 2.33 s | 1664 | 5120 |
| 1280 × 736 | 124 | 5.17 s | 1664 | 10240 |
| 1280 × 736 | 243 | 10.13 s | 1664 | 19328 |
| 1920 × 1088 | 56 | 2.33 s | 3072 | 10368 |
| 1920 × 1088 | 124 | 5.17 s | 3200 | 21760 |
| 1920 × 1088 | 243 | 10.13 s | 3200 | 41600 |
| 640 × 352 | 489 (joined) | 20.38 s | 1664 | 4352 |
| 1280 × 736 | 489 (joined) | 20.38 s | 1792 | 15488 |

The estimates use the prompt in the memory example below, 24 fps, the default 50 scheduler points, a portable BF16 modulation bank, rough mode 0 and WebM output. Single-segment rows use T2VA. Joined rows use FL2VA with four image keyframes and segments of 158, 158 and 175 frames.

The planner uses the ncnn graph, tensor sizes and allocation lifetimes on the selected Vulkan device, including video VAE tiles and the modulation producer when a bank needs baking. The OS and Vulkan driver need additional capacity. With the bundled ncnn revision, mapped weight pages can stay resident during disk paging, so process RAM can exceed the planning budget.

### Example Command

Text to video and audio

```shell
h3-ncnn-vulkan -p "A red ceramic teapot slowly rotates on a pale oak table in daylight." \
    -s 384,256 -f 56 -o teapot.webm
```

First and last frames to video and audio

```shell
h3-ncnn-vulkan -t fl2va -i first.png -i last.png \
    -p "The red ceramic teapot slowly turns cobalt blue." \
    -s 384,256 -f 124 -o keyframes.webm
```

Generate a longer video with intermediate keyframes

```shell
h3-ncnn-vulkan -t fl2va -i A.png -i B.png -i C.png -i D.png \
    -f 73 -f 90 -f 107 \
    -p "The teapot turns to the left." \
    -p "The teapot continues turning as it moves closer." \
    -p "The teapot completes the turn in close-up." -o joined.webm
```

Pass one frame count per pair of adjacent keyframes. Use one shared prompt or one prompt per segment. The writer counts each shared boundary once, so this example produces 268 frames. All segments share one canvas and optional audio reference. Each model is loaded once per stage, and adjacent segments reuse their shared keyframe encodings.

Multimodal references

```shell
h3-ncnn-vulkan -t ref2va -i teapot.png --reference-audio tone.wav \
    -p "Using <Picture 1> as the reference, show the teapot rotating slowly. Use <Audio 1> as the sound reference." \
    --reference-short-edge 512 -s 384,256 -f 124 -o reference.webm
```

Set size, frame count, scheduler steps, seed and GPU

```shell
h3-ncnn-vulkan -s 640,352 -f 124 -l 50 -r 2026 -g 0 \
    -p "A red ceramic teapot rotates on a table." -o teapot.webm
```

Estimate memory with explicit budgets, in MiB

```shell
h3-ncnn-vulkan --estimate-memory --ram-budget 16384 --vram-budget 32768 \
    -p "A red ceramic teapot rotates on a table." -s 1280,736 -f 124 -o teapot.webm
```

Generate with the same parameters

```shell
h3-ncnn-vulkan --ram-budget 16384 --vram-budget 32768 \
    -p "A red ceramic teapot rotates on a table." -s 1280,736 -f 124 -o teapot.webm
```

A budget of `0` uses the available capacity automatically. Larger budgets keep more weights resident and improve throughput. The estimate reports the selected stage peaks and the overall requirements with disk paging or resident weights. For joined FL2VA, it includes intermediate results held across segments.

Optional rough K/V reuse

```shell
h3-ncnn-vulkan -t fl2va -i first.png --rough-mode 2 \
    -p "The red ceramic teapot slowly turns cobalt blue." \
    -s 384,256 -f 124 -o rough.webm
```

Rough modes reuse K/V and can change movement and sound. Active rows attend to the full K/V sequence. `--kv-cache-mib` caps the cache for the clip at 6144 MiB by default.

| Mode | Cached K/V rows | Refresh forwards |
| --- | --- | --- |
| 0 | None | Every forward |
| 1 | Text | 1, 3, 5, … |
| 2 | Text and references | 1, 3, 5, … |
| 3 | Text and references | First forward only |

### LoRA

Load an adapter with `--lora`. Repeat the option to combine adapters and give `--lora-scale` values in the same order. Unspecified strengths default to 1. Merged adapters combine updates in FP32 during loading. Pika Flex Turbo evaluates its low-rank projections in Vulkan.

```shell
h3-ncnn-vulkan -t fl2va -i teapot.png -i teapot.png \
    -p "A smooth full orbit around the stationary teapot." \
    -s 640,352 -f 73 -l 29 \
    --lora minimax_h3_flf2v_lora_v1.safetensors -o orbit.webm
```

Combine Orbit with pi-H3 in the same run. pi-H3 sets the sampling schedule.

```shell
h3-ncnn-vulkan -t fl2va -i subject.png -i subject.png \
    -p "A smooth 360 degree camera orbit around the stationary subject." \
    -s 384,384 -f 124 -l 3 \
    --lora minimax_h3_flf2v_lora_v1.safetensors \
    --lora pi_h3_2step_adapter/diffusion_pytorch_model.safetensors -o orbit-pi.webm
```

<details>
<summary>Supported adapters and sampling presets</summary>

Use pi-H3 for a text-only generation

```shell
h3-ncnn-vulkan -p "A red ceramic teapot with steam rising in daylight." \
    -s 384,256 -f 73 \
    --lora pi_h3_4step_adapter/diffusion_pytorch_model.safetensors -o pi-h3.webm
```

| Adapter | Workflow | Sampling |
| --- | --- | --- |
| 360 Orbit | FL2VA | Set `-l 29` for 28 DiT forwards. Use the same reference for both keyframes. |
| Character Swap V1 / V1.5, Fight Flow V1 / V2 | Ref2VA | H3 schedule. For Character Swap, pass the source clip with `--reference-video` and the new character with `-i`. |
| Sentinel7 Extreme Realism | FL2VA | H3 schedule. Includes audio input and output projections. |
| Larry H3 Turbo | T2VA / FL2VA | Dual-clock Euler. V4 defaults to 6 DiT forwards. Earlier 4-step weights default to 4. |
| Pika H3 Flex Turbo | Match the FL2VA or Ref2VA filename | Simple Euler with the file's 2, 4 or 8 forward preset. FL2VA uses video shift 6. |
| pi-H3 2-step / 4-step VAE adapters | T2VA / FL2VA fixed build | pi-H3 Euler with 2 or 4 DiT forwards and strength 1. |

Keep the original Pika filename and pi-H3 folder name to select their presets automatically. The pi-H3 VAE adapters use the existing video VAE. The DWD checkpoints require a separate pixel refiner.

Adapter and sampler references are available from the publishers. [Orbit](https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA), [Character Swap](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA), [Fight Flow](https://huggingface.co/JOKER141/H3-Fight-Flow-Fight-Interaction-LoRA), [Larry Turbo](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora), [Pika Flex Turbo](https://huggingface.co/Pika-Labs/Pika-H3-Flex-Turbo-LoRA), [pi-H3](https://huggingface.co/Lakonik/pi-H3), [Extreme Realism](https://huggingface.co/Sentinel7/h3/blob/main/2834417/3446189/README.md)

</details>

### Modulation cache

The model download includes portable BF16 modulation banks for the default `-l 50`. Filenames end in `none`, `video`, `audio` or `both`, according to the input references. The optional pi-H3 banks in `transformer/fl2va/` cover text-only generation and FL2VA with one or both keyframes, at `-l 3` and `-l 5`.

Matching banks can be copied between machines. The loader checks model fingerprints, sampling preset and reference modalities. Requests without a matching bank use the modulation producer and cache the result under `fixed-build/banks/`.

<details>
<summary>Precompute a bank for another schedule</summary>

Use `--bake-modulations` to precompute a portable bank for a base H3 schedule or supported pi-H3 preset. Specify the workflow and reference type.

```shell
h3-ncnn-vulkan -t fl2va -l 17 --bake-modulations video
h3-ncnn-vulkan -t ref2va -l 17 --bake-modulations both
h3-ncnn-vulkan -t fl2va --lora pi_h3_2step_adapter/diffusion_pytorch_model.safetensors --bake-modulations video
```

</details>

### Full Usages

```console
Usage: h3-ncnn-vulkan [options]

  -h                 show this help
  -p TEXT            prompt (one for the video or one per FL2VA segment)
  -o PATH            lossy .webm or lossless .mkv (default: output.webm)
  -i PATH            FL2VA keyframe or Ref2VA image reference (repeat in order)
  -s WIDTH,HEIGHT    output dimensions, multiples of 32
  -f N               frames at 24 fps, rounded up (repeat per FL2VA segment)
  -l N               scheduler points, including zero (default: 50 or adapter preset)
  -r N               sampling seed (default: 42)
  -m DIR             ncnn model directory (default: models/h3)
  -g N               Vulkan device index (default: automatic)
  -t TYPE            t2va, fl2va, ref2va (default: t2va)

  --last-image PATH                 optional final FL2VA keyframe
  --reference-video PATH            WebM/MKV video reference, including its audio stream
  --reference-audio PATH            WAV or WebM/MKV audio reference for fl2va or ref2va
  --reference-short-edge N          ref2va image short edge (default: 2048)
  --reference-video-short-edge N    ref2va video short edge (default: 768)
  --lora PATH                       H3 LoRA .safetensors file (repeat to combine)
  --lora-scale VALUE                adapter strengths in --lora order (default: 1)
  --bake-modulations TYPE           build a bank only: none, video, audio, both
  --rough-mode 0|1|2|3              exact, alternate text, alternate full, freeze full (default: 0)
  --kv-cache-mib N                  Vulkan K/V cache ceiling (default: 6144)
  --ram-budget MiB                  process RAM budget (0=auto, default)
  --vram-budget MiB                 Vulkan memory budget (0=auto, default)
  --estimate-memory                 calculate the memory plan without running inference
```

- Use `-s WIDTH,HEIGHT` to set the output dimensions. Each side is a multiple of 32 from 32 to 4096 pixels. The default is 124 frames at 24 fps. The VAE rounds the frame count up to `17 * n + 5`, with supported output lengths of 22 to 345 frames per segment.
- `-l` counts scheduler points including the final zero. The default 50 points run 49 DiT forwards, and 17 points run 16.
- FL2VA accepts either keyframe or both. It also accepts one experimental `--reference-audio` with or without keyframes. Refer to it as `<Audio 1>` in the prompt. Speaker similarity can vary.
- Ref2VA accepts up to 9 images, 3 videos and 3 audio files. Pass references in prompt order. Number `<Picture i>`, `<Video i>` and `<Audio i>` independently from 1. Audio embedded in a reference video receives its own audio label.
- Images can be PNG, JPEG or WebP. Reference videos can be WebM or MKV with VP8/VP9 and optional Opus or PCM. Audio references can be mono/stereo PCM WAV or Opus/PCM tracks in WebM or MKV.
- The output filename extension selects the encoding.

| Extension | Video | Stereo audio |
| --- | --- | --- |
| `.webm` | VP9 4:2:0 | Opus, 48 kHz |
| `.mkv` | Lossless VP9 RGB 4:4:4 | FP32 PCM, 32 kHz |

## Build from Source

1. Initialize the submodules in the source checkout.

```shell
git submodule update --init --recursive --depth 1
```

2. Build and install with CMake.

```shell
cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
cmake --install build --prefix "$PWD/build/package"
```

The build requires a C/C++ compiler, CMake, Git, Ninja, Make, NASM, diffutils, Perl and Vulkan development headers. CMake builds ncnn and the static media libraries from the pinned submodules under `src/`. RTTI and C++ exceptions are disabled by default (`NCNN_DISABLE_RTTI=ON`, `NCNN_DISABLE_EXCEPTION=ON`).

On Windows, use MSYS2 UCRT64 with GCC, CMake, Ninja, Make, NASM, diffutils and Perl for the bundled build. On macOS, pass `-DVulkan_LIBRARY=/path/to/libMoltenVK.a` to link MoltenVK.

Use `-DUSE_SYSTEM_NCNN=ON` for an installed ncnn with Vulkan and batch support. Select system image libraries with `USE_SYSTEM_JPEG`, `USE_SYSTEM_PNG`, `USE_SYSTEM_WEBP` or `USE_SYSTEM_ZLIB`. System PNG uses system zlib. For prebuilt static codecs, set `H3_MEDIA_PREFIX` to the installation prefix.

## Original MiniMax-H3 Project

- [MiniMax-H3 model and documentation](https://huggingface.co/MiniMaxAI/MiniMax-H3)

## Other Open-Source Code Used

- Neural network inference and Vulkan shader compilation. [ncnn](https://github.com/Tencent/ncnn), [glslang](https://github.com/KhronosGroup/glslang)
- Image decoders, encoder layout and BPE tokenizer. [h3image-ncnn-vulkan](https://github.com/nihui/h3image-ncnn-vulkan), [qwenimage-ncnn-vulkan](https://github.com/nihui/qwenimage-ncnn-vulkan)
- Image codecs. [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo), [libpng](https://github.com/pnggroup/libpng), [libwebp](https://github.com/webmproject/libwebp), [zlib-ng](https://github.com/zlib-ng/zlib-ng)
- Video and audio codecs and containers. [libvpx](https://github.com/webmproject/libvpx), [libopus](https://github.com/xiph/opus), [libwebm](https://github.com/webmproject/libwebm)

The standalone package includes third-party license notices.

## License

Source code is licensed under Apache-2.0. [Project license](LICENSE)

The checkpoint and converted weights are subject to the terms for access, use and redistribution in the MiniMax H3 Community License. [Model license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
