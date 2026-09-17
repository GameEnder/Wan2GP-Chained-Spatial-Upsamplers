# 🔗 Chained Spatial Upsamplers for Wan2GP

A plugin for [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) that allows multiple spatial upsamplers and visual refiners to be staged sequentially in series (e.g. AI restoration followed by face refinement or sharp resampling).

---

## ✨ Features

- **Multi-Stage Pipelines**: Chain multiple spatial upscalers and refiners in series (Stage 1 $\to$ Stage 2 $\to$ ... $\to$ Stage $N$).
- **Native Wan2GP Integration**: Configured chains appear directly in Wan2GP's **Spatial Upsampling** dropdowns across Media Generator and Post Processing tabs.
- **Interactive UI Manager**: Built-in **Chained Upsamplers** tab to create, configure, test, and save custom multi-stage presets.
- **Real-Time Synchronization**: Automatically updates Wan2GP form dropdowns as soon as presets are created or modified without requiring a browser reload.
- **Memory & VRAM Management**: Automatically unloads models and clears VRAM between stages for smooth multi-stage processing on consumer GPUs.
- **Detailed Progress Tracking**: Displays stage-by-stage progress (e.g. `Stage 1/2 (FlashVSR): ...`, `Stage 2/2 (H3 Face Refiner): ...`).

---

## 📦 Installation

### Method 1: Via Wan2GP UI (Recommended)
1. In Wan2GP, navigate to the **Plugins** tab.
2. Under **Install New Plugin**, paste the GitHub repository URL.
3. Click **Install**.
4. Enable the plugin and restart Wan2GP.

### Method 2: Manual Installation
Clone this repository into your Wan2GP `plugins/` directory:

```bash
cd Wan2GP/plugins
git clone https://github.com/<your-username>/WanGP-addon.git wan2gp-chained-upsamplers
```

Restart Wan2GP.

---

## 🚀 How to Use

### 1. Creating & Editing Presets
1. Open Wan2GP and select the **Chained Upsamplers** tab.
2. Select an existing preset or choose **➕ Create New Preset...**.
3. Set the **Preset Display Name** and **Description**.
4. Select upsamplers/refiners from the dropdown (e.g. `FlashVSR 2x`, `Lanczos 1.5x`, `H3 Face Refiner`) and click **➕ Add Stage to Chain**.
5. Click **💾 Save Preset**.

### 2. Running in Wan2GP
1. Go to the **Media Generator** or **Post Processing** tab.
2. In the **Spatial Upsampling** dropdown, select your saved chain (e.g. `Chain: FlashVSR 2x -> Face Refiner`).
3. Generate media or run post-processing. Each stage will execute sequentially with real-time stage progress reporting.

---

## 📄 License

Apache License.
