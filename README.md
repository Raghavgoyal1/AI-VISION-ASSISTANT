# AI Vision Assistant

A real-time webcam assistant that **detects objects**, **describes the scene in natural language**, and fuses both into a single human-readable summary — all live, on-screen.

| Component | Model | Job |
|---|---|---|
| Object Detection | YOLO11 (Ultralytics) | Draws bounding boxes and labels for everything the camera sees |
| Image Captioning | BLIP (Salesforce) | Generates a natural-language caption of the current frame |
| Scene Analyzer | Rule-based fusion | Combines detections + caption into one description |

## Demo

Press **SPACE** in the video window to analyze the current frame. The console and on-screen overlay will show:

```
YOLO detections: person -> 92%, laptop -> 87%
BLIP caption: a person sitting at a desk with a laptop
Final: I can see person, laptop. A person sitting at a desk with a laptop.
```

## Project Structure

```
AI-Vision-Assistant/
├── main.py             # Entry point — camera loop, UI, keyboard controls
├── camera_engine.py     # Webcam capture wrapper (OpenCV)
├── vision_engine.py     # YOLO11 object detection
├── caption_engine.py    # BLIP image captioning
├── scene_analyzer.py    # Combines detections + caption into a description
├── config.py             # Central settings (camera index, thresholds, model paths)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Requirements

- Python 3.9+
- A webcam
- (Optional) CUDA-capable GPU or Apple Silicon for faster inference — CPU also works, just slower

## Installation

```bash
git clone https://github.com/<your-username>/AI-Vision-Assistant.git
cd AI-Vision-Assistant

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

The YOLO11 weights (`yolo11n.pt`) and the BLIP model are downloaded automatically on first run — no manual download needed.

## Usage

```bash
python main.py
```

Optional flags:

```bash
python main.py --camera 1 --confidence 0.6 --model yolo11n.pt
```

| Flag | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera device index |
| `--confidence` | `0.50` | Minimum YOLO detection confidence (0–1) |
| `--model` | `yolo11n.pt` | YOLO model path/name |

**Controls:**
- `SPACE` — analyze the current frame (runs BLIP + scene fusion)
- `Q` — quit

## How It Works

1. `camera_engine.py` continuously grabs frames from the webcam.
2. Every frame is run through `vision_engine.py` (YOLO11), which returns bounding boxes and labels drawn live on screen.
3. When you press **SPACE**, the current frame is also sent to `caption_engine.py` (BLIP), which generates a free-text caption.
4. `scene_analyzer.py` merges the YOLO object list and the BLIP caption into one readable sentence, shown both in the console and as an overlay.

## Configuration

All tunable values (camera index, resolution, confidence threshold, model names, UI colors) live in `config.py` so you don't need to touch the engine code to adjust behavior.

## Troubleshooting

- **`Could not open webcam`** — make sure no other app is using the camera, and try a different `--camera` index (0, 1, 2...).
- **Slow performance** — the app auto-selects CUDA → MPS → CPU, but a first-run download of model weights can take a minute depending on your connection.
- **Import errors** — double check you're in the virtual environment and ran `pip install -r requirements.txt`.

## License

Released under the [MIT License](LICENSE).
