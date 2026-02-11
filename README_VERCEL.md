# Deploying MitoReach to Vercel

This folder contains a version of MitoReach optimized for **Vercel**. It uses a **FastAPI** backend (Python) and a **Modern Vanilla JS** frontend.

## 🚀 How to Deploy

1.  **Initialize a Git Repo**:
    ```bash
    cd MitoReach_Vercel
    git init
    git add .
    git commit -m "Vercel deployment"
    ```
2.  **Push to GitHub**:
    Create a new repository on GitHub and push this folder to it.
3.  **Deploy on Vercel**:
    *   Go to [Vercel.com](https://vercel.com).
    *   Click "Add New Project" and import your GitHub repo.
    *   **Frameowrk Preset**: Select "Other" (Vercel will automatically detect the `api/` folder and `requirements.txt`).
    *   Click **Deploy**.

## 🏗️ Architecture
- **Frontend**: `public/index.html` (Tailwind, Lucide, Glassmorphism).
- **Backend API**: `api/index.py` (FastAPI).
- **Routing**: `vercel.json` maps `/api` requests to the Python function and all others to the HTML frontend.

## 🧪 Local Testing
You can test the API locally if you have FastAPI installed:
```bash
pip install fastapi uvicorn
uvicorn api.index:app --reload
```
Then open `public/index.html` in your browser.
