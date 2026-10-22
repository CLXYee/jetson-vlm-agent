# Jetson VLM Agent

**Prompt-driven visual understanding on NVIDIA Jetson using Gemma and Qwen vision-language models.**

Jetson VLM Agent captures frames from cameras, video files, or streams and generates natural-language descriptions with locally loaded vision-language models (VLMs). It combines threaded inference, live caption overlays, and CSV logging in a modular Python application for edge AI experimentation.

An image-folder mode supports scene-captioning experiments, including geospatial and landscape imagery.

## Features

- **Local VLM inference:** Model wrappers for Gemma 3, Qwen2-VL, Qwen2.5-VL, and Qwen3-VL.
- **Flexible inputs:** Camera devices, video files, and stream URLs through `jetson-utils`, plus folders of images.
- **Custom prompts:** Configure the visual question or captioning instruction from the command line.
- **Threaded processing:** Run inference in the background while rendering frames with the latest available caption.
- **Caption overlays:** Pygame-based video display and a Matplotlib option for Jetson image mode.
- **Experiment outputs:** CSV descriptions, console inference timings, and optional annotated video recording in video mode.
- **Platform-aware setup:** Scripts detect Jetson hardware and select an environment setup path.

“Real-time” refers to processing a live source with asynchronous caption updates. Caption frequency depends on the model, input, and hardware; it does not imply inference on every frame or full-video temporal reasoning.

## How it works

1. Capture a frame from a video source or load an image from a folder.
2. Prepare the image and text prompt using the selected model's processor.
3. Generate a description with the locally loaded VLM.
4. Update the displayed caption and optionally save the result.

The main components—input handling, model inference, agent orchestration, and display—are separated so they can be adapted independently.

## Requirements

| Component | Requirement |
| --- | --- |
| Primary platform | NVIDIA Jetson running Linux with a compatible JetPack/CUDA stack |
| Python | Setup scripts target Python 3.10 on JetPack 5/6 and non-Jetson systems |
| Inference | CUDA-enabled PyTorch and a compatible `torchvision` build |
| Model loading | Hugging Face Transformers, Accelerate, and access to the selected model |
| Jetson input/display | `jetson-utils`, built for the active Python environment |
| Graphical output | A working desktop/display session for Pygame or Matplotlib |
| Video recording | The system FFmpeg executable |

Choose a model that fits the device's available memory. The current wrappers use CUDA and request BF16 model loading; CPU-only and Apple Silicon execution are not implemented. The non-Jetson image path also requires a compatible NVIDIA GPU.

JetPack 4 is detected by the setup scripts, but its Python 3.8 setup is unsuitable for the newer Transformers models used here. Detection alone does not establish compatibility across all Jetson devices or JetPack versions.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CLXYee/jetson-vlm-agent.git
cd jetson-vlm-agent
```

### 2. Inspect and build the environment

Preview the detected platform and proposed configuration:

```bash
bash build_env.sh --check
```

Review the selected script under `script/`, then run:

```bash
bash build_env.sh
```

The setup uses Conda when available and otherwise creates a Python virtual environment named `video_query`. The Jetson path installs system dependencies and builds `jetson-utils`; some steps require `sudo`.

The scripts contain platform-specific wheel URLs and dependency choices. Check that they match your installed JetPack, CUDA, Python, and hardware before installation, especially on JetPack 5.

### 3. Activate the environment

With Conda:

```bash
conda activate video_query
```

With Python venv:

```bash
source video_query/bin/activate
```

For Qwen models, install the vision preprocessing helper, which is imported by the model wrapper but is not explicitly installed by the setup scripts:

```bash
python -m pip install qwen-vl-utils
```

### 4. Verify dependencies

```bash
python -m torch.utils.collect_env
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

On Jetson, also verify:

```bash
python -c "import jetson_utils; print('jetson-utils import OK')"
```

If the selected model requires authentication or license acceptance, complete those steps on Hugging Face before loading it.

## Usage

Run commands from the repository root. Create output directories before enabling logging or recording:

```bash
mkdir -p output
python video_query.py --help
```

### Live camera with Gemma 3

Display a camera feed with generated captions and save descriptions to CSV:

```bash
python video_query.py \
  --mode video \
  --source /dev/video0 \
  --model_id google/gemma-3-4b-it \
  --prompt "Describe the scene in one sentence." \
  --max_tokens 32 \
  --on_video \
  --save_output \
  --output_file output/gemma_camera.csv
```

### Use a Qwen model

Select a checkpoint compatible with the installed Transformers version and device memory:

```bash
python video_query.py \
  --mode video \
  --source /dev/video0 \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --prompt "Describe the visible objects." \
  --max_tokens 32 \
  --on_video
```

The Qwen wrapper prepends an instruction requesting an answer within ten words. A larger token limit alone does not remove that instruction.

### Video file without a display

Run the Jetson video pipeline without opening a display window:

```bash
python video_query.py \
  --mode video \
  --source ./input/video.mp4 \
  --model_id google/gemma-3-4b-it \
  --prompt "Describe the scene precisely." \
  --headless \
  --save_output \
  --output_file output/video_descriptions.csv
```

Replace the source with your own file or an RTSP URL supported by `jetson-utils`. Headless mode still requires the Jetson video dependencies. Use **Ctrl+C** to exit; the main loop does not automatically terminate at the end of a video file.

### Process an image folder

```bash
python video_query.py \
  --mode image \
  --image_source ./selected/Global \
  --model_id google/gemma-3-4b-it \
  --prompt "Describe the landscape and visible geographic features." \
  --plot plt \
  --wait_time 5 \
  --save_output \
  --output_file output/image_descriptions.csv
```

Replace `./selected/Global` with a folder of your own images. Image mode selects a Jetson or PC agent based on hardware detection. On Jetson, `--plot plt` selects Matplotlib; `--plot pygame` selects the Pygame display path.

The current entry point sets the image-mode token limit to **256**, overriding `--max_tokens`. The `--headless` flag is handled only in video mode.

### Record annotated video

```bash
python video_query.py \
  --mode video \
  --source /dev/video0 \
  --model_id google/gemma-3-4b-it \
  --on_video \
  --save_video \
  --video_path output/annotated_video.mp4 \
  --save_output \
  --output_file output/recorded_descriptions.csv
```

Video recording uses the display loop and FFmpeg. Use it with a graphical session, without `--headless`, and with frame dimensions compatible with the configured output size. The writer currently uses a fixed 15 FPS; it does not preserve source timing.

## Command-line reference

| Argument | Purpose | Default |
| --- | --- | --- |
| `--mode` | Select `video` or `image` | `video` |
| `--model_id` | Hugging Face model identifier | `google/gemma-3-4b-it` |
| `--prompt` | Override the mode's default instruction | Mode-dependent |
| `--max_tokens` | Generation limit in video mode | `16` |
| `--source` | Camera device, video file, or stream URL | `/dev/video0` |
| `--frame_rate` | Requested capture frame rate; not inference FPS | `60` |
| `--on_video` | Enable video display and caption overlays | Off |
| `--headless` | Disable display in video mode | Off |
| `--width` / `--height` | Configure display dimensions | `1280` / `720` |
| `--save_output` | Enable CSV logging | Off |
| `--output_file` | CSV destination | `output/prompt_history.csv` |
| `--save_video` | Request recording; use video mode on Jetson | Off |
| `--video_path` | Recording destination | `output/output.mp4` |
| `--image_source` | Image-folder input | `./selected/Global` |
| `--wait_time` | Image-mode wait interval in seconds | `10` |
| `--plot` | Jetson image display: `plt` or `pygame` | `plt` |
| `--return_tensors` | Capture format: `cuda`, `np`, or `pt` | `cuda` |
| `--quantize` | Request 4-bit NF4 loading through bitsandbytes | Off |

Use the default `--return_tensors cuda` for the current threaded Jetson capture path, which copies frames with `cudaMemcpy`. Other formats are exposed by the parser but are not interchangeable throughout that path.

Quantization depends on a compatible bitsandbytes build and model-loading behavior on the target device. It is an experimental option rather than a guaranteed low-memory configuration.

## Outputs

| Output | Contents |
| --- | --- |
| Video-mode CSV | `timeframe`, `description` |
| Jetson image-mode CSV | `timeframe`, `description` |
| PC image-mode CSV | `timeframe`, `filename`, `description` |
| Console | Generated descriptions and inference durations |
| Annotated MP4 | Rendered frames with caption overlays in video mode |

Jetson timestamps are Unix timestamps; the PC image agent uses formatted date/time strings. Jetson agents write CSV entries in batches of five, so the final incomplete batch may be lost on interruption. Use separate CSV files when switching agent types because their schemas differ.

## Project structure

| Path | Responsibility |
| --- | --- |
| `video_query.py` | CLI, model selection, and agent initialization |
| `model.py` | Gemma and Qwen loading, preprocessing, and generation |
| `video_agent.py` | Threaded video inference, captions, logging, and recording |
| `image_agent.py` | Jetson and PC image-folder agents |
| `camera.py` | Jetson camera, video-file, and stream capture |
| `image_source.py` | Jetson image-folder input |
| `display.py` | Display adapters and caption overlays |
| `utils/` | Image conversion, display helpers, and supporting utilities |
| `build_env.sh` | Platform detection and setup dispatch |
| `script/` | Platform-specific environment and PyTorch setup scripts |
| `Dockerfile`, `run_docker.sh`, `docker_build_env.sh` | Container setup utilities |
| `selected/`, `output/` | Input samples and generated outputs |

## Current limitations

- **Frame-based understanding:** Each inference call processes an individual image. The agent does not maintain conversation history or perform temporal reasoning across a video clip.
- **Benchmark stopping condition:** The video and Jetson image agents request a stop after 100 inference calls. The CLI's video loop may remain active until interrupted.
- **Image recording on Jetson:** The Jetson image agent accepts recording arguments but does not implement video writing.
- **API and server options:** `--api` currently passes an unsupported constructor argument; its wrapper uses a local Transformers pipeline rather than a remote API. `--on_server` is a placeholder with no streaming implementation.
- **Container workflow:** Container helpers are included, but the launcher does not configure camera-device or graphical-display access. Review those mappings and the base-image compatibility before using it for live capture.

## Troubleshooting

| Symptom | Suggested check |
| --- | --- |
| CUDA is unavailable | Confirm the active environment contains a PyTorch build compatible with the device's CUDA/JetPack stack. |
| `jetson_utils` cannot be imported | Build it for the active Python environment and verify the installed extension is visible there. |
| A Qwen preprocessing import fails | Install `qwen-vl-utils` in the active environment. |
| Model class cannot be imported | Check that the installed Transformers version provides the selected model class. |
| Model download is denied | Check the model's access requirements and Hugging Face authentication. |
| GPU runs out of memory | Use a smaller checkpoint, reduce generation length, or evaluate compatible quantization. |
| Camera or stream cannot be opened | Check the source path, device permissions, stream availability, and codec support. |
| No graphical display is available | Use `--headless` in video mode and disable recording. |
| CSV or MP4 creation fails | Create the parent directory and check write permissions; for MP4, also check FFmpeg and frame dimensions. |
| Captions update slowly | Compare per-call inference times; capture FPS and caption-update rate are separate. |

## License

The previous README declares MIT licensing, but the repository currently has no standalone `LICENSE` file. Add the appropriate license file with the correct copyright holder before relying on that declaration for redistribution. Model weights and third-party dependencies remain subject to their respective licenses.
