# 🛠️ Environment Troubleshooting Guide

### 1. TensorFlow and CUDA (GPU) Compatibility Issues
If you encounter errors getting TensorFlow to recognize or load your GPU libraries, install the bundled compatible packages by running the following commands in your terminal:

```bash
pip install --upgrade pip
pip install tensorflow[and-cuda]
```

---

### 2. Model Plotting Issues (`plot_model`)
If `plot_model` fails, it means you only have the Python wrapper but lack the core system tool. You must install the Python package **and** the system-level Graphviz software.

**Step 1: Install the Python library**
```bash
pip install graphviz
```

**Step 2: Install the system application based on your OS**

* **If you are using Arch Linux (or Manjaro):**
  ```bash
  sudo pacman -S graphviz
  ```

* **If you are using Ubuntu (or Debian):**
  ```bash
  sudo apt-get update
  sudo apt-get install graphviz
  ```

> 💡 **Note:** After installing the system application, make sure to restart your IDE or environment (like VS Code or Jupyter Notebook) so Python can detect the new system paths. Everything should work perfectly! 😎
