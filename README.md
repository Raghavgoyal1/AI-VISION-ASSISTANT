<div align="center">

# 👁️ AI Vision Assistant

### Real-time object detection, image captioning, and scene understanding — all from your webcam.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![BLIP](https://img.shields.io/badge/BLIP-Salesforce-FF6F00?style=for-the-badge)](https://huggingface.co/Salesforce/blip-image-captioning-base)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-Live%20Video-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

**Detect → Caption → Understand — in real time, on your own machine.**

</div>

---

## ✨ What It Does

AI Vision Assistant watches your webcam feed and builds a live, human-readable understanding of what's in front of it — combining three AI models into one pipeline:

```
📷  Webcam Frame
      │
      ▼
🎯  YOLO11  ──────────────►  Bounding boxes + object labels (live, every frame)
      │
      ▼  (on SPACE key)
🖼️  BLIP  ────────────────►  Natural-language caption of the scene
      │
      ▼
🧠  Scene Analyzer  ──────►  "I can see person, laptop. A person sitting at a desk."
```

> Press **SPACE** anytime to get a full scene breakdown, printed to console *and* overlaid on the video feed.

---

## 🎬 Demo

```
🎯 YOLO Detections:
   person    → 92%
   laptop    → 87%
   coffee cup → 78%

🖼️ BLIP Caption:
   "a person sitting at a desk with a laptop"

🧠 Final Scene Description:
   "I can see person, laptop, coffee cup. A person sitting at a desk with a laptop."
```

<div align="center">
  <em>🎥 Add your own demo GIF/screenshot here — drop it in a /docs or /assets folder and link it!</em>
</div>

---

## 🧩 Tech Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| 🎯 Object Detection | **YOLO11** (Ultralytics) | Real-time bounding boxes & labels |
| 🖼️ Image Captioning | **BLIP** (Salesforce) | Natural-language scene captions |
| 🧠 Fusion Logic | Custom Python | Merges detections + caption into one description |
| 📷 Video I/O | **OpenCV** | Webcam capture & live overlay rendering |
| ⚡ Backend | **PyTorch** | Auto-selects CUDA → MPS → CPU |

---

## 📂 Project Structure

```
AI-Vision-Assistant/
├── 🎬 main.py              # Entry point — camera loop, UI, controls
├── 📷 camera_engine.py      # Webcam capture wrapper (OpenCV)
├── 🎯 vision_engine.py      # YOLO11 object detection
├── 🖼️ caption_engine.py     # BLIP image captioning
├── 🧠 scene_analyzer.py     # Fuses detections + caption
├── ⚙️ config.py             # Central settings
├── 📄 requirements.txt
├── 🚫 .gitignore
├── 📜 LICENSE
└── 📘 README.md
```

---

## 🚀 Quick Start

### 1️⃣ Clone the repo

```bash
git clone https://github.com/<your-username>/AI-Vision-Assistant.git
cd AI-Vision-Assistant
```

### 2️⃣ Set up a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

> 💡 YOLO11 weights and the BLIP model are downloaded **automatically** on first run — no manual setup needed.

### 4️⃣ Run it

```bash
python main.py
```

---

## 🎮 Controls

| Key | Action |
|:---:|:---|
| `SPACE` | 🧠 Analyze the current frame (caption + scene description) |
| `Q` | ❌ Quit the application |

---

## ⚙️ Configuration & CLI Flags

Fine-tune behavior without touching the code:

```bash
python main.py --camera 1 --confidence 0.6 --model yolo11n.pt
```

| Flag | Default | Description |
|:---|:---:|:---|
| `--camera` | `0` | Camera device index |
| `--confidence` | `0.50` | Minimum YOLO detection confidence |
| `--model` | `yolo11n.pt` | YOLO model path/name |

Prefer editing a file instead? All defaults live in **`config.py`**.

---

## 🛠️ How It Works

<details>
<summary><strong>Click to expand the full pipeline breakdown</strong></summary>

<br>

1. **`camera_engine.py`** continuously grabs frames from your webcam using OpenCV.
2. Every single frame is run through **`vision_engine.py`** (YOLO11), which detects objects and draws live bounding boxes.
3. When you press **SPACE**, that frame is also sent to **`caption_engine.py`** (BLIP), generating a free-text caption like *"a dog sitting on a couch."*
4. **`scene_analyzer.py`** merges the YOLO object list and BLIP caption into one natural sentence.
5. Everything is rendered live on the video window **and** logged to your console.

</details>

---

## 🩺 Troubleshooting

<details>
<summary><strong>Could not open webcam</strong></summary>

Make sure no other app is using your camera, and try a different index:
```bash
python main.py --camera 1
```
</details>

<details>
<summary><strong>Running slow</strong></summary>

The app auto-selects the fastest available device (CUDA → MPS → CPU). First-run model downloads can also take a minute depending on your internet speed.
</details>

<details>
<summary><strong>Import errors</strong></summary>

Confirm you're inside the virtual environment and ran:
```bash
pip install -r requirements.txt
```
</details>

---

## 🗺️ Roadmap

- [ ] Save analyzed frames + descriptions to a log file
- [ ] Add support for multiple camera streams
- [ ] Web dashboard for remote viewing
- [ ] Voice narration of scene descriptions

---

## 🤝 Contributing

Pull requests are welcome! Feel free to open an issue first to discuss what you'd like to change.

---

## 📜 License

Released under the **[MIT License](LICENSE)** — free to use, modify, and distribute.

---

<div align="center">

**If this project helped you, consider giving it a ⭐!**

</div>
