# Deployment

TruthShield AI deploys as two separate services:

- Frontend: Cloudflare Pages from `frontend/`
- Backend: Hugging Face Spaces Docker Space from the repository root

## 1. Deploy The Backend To Hugging Face Spaces

Create a new Hugging Face Space:

- Owner: your Hugging Face username or org
- Space name: `truthshield-ai-backend`
- SDK: Docker
- Visibility: public or private

Deploy the **repository root**, not only `backend/`. The root `Dockerfile` copies both the backend and the measured `truthshield-image-detector-v2` model into the image. A backend-only deployment omits the trained model and silently falls back to much weaker generic or heuristic detection.

With Git:

```powershell
Set-Location "C:\Users\rishi\OneDrive\Desktop\Hackathon Project\truthshield-ai"
git lfs install
git remote add hf https://huggingface.co/spaces/<hf-user>/truthshield-ai-backend
git fetch hf main
git lfs push --all https://huggingface.co/spaces/<hf-user>/truthshield-ai-backend
git push hf HEAD:main --force-with-lease
```

The last command is intended for the newly created Space. Fetching first makes `--force-with-lease` refuse to overwrite Space changes you have not seen.

The 94 MB `model.safetensors` file is managed by Git LFS. Confirm that the Space contains the actual LFS object, not a small pointer file.

In the Space variables, keep `ENABLE_LOCAL_AI_MODELS=true`. Delete the old `AI_IMAGE_DETECTOR_MODELS=Organika/...` override or leave it blank. The code now keeps a packaged TruthShield model first even if an older override remains, but removing stale configuration makes the deployment easier to audit.

After the Space builds, the backend URL will be:

```text
https://<hf-user>-truthshield-ai-backend.hf.space
```

Check:

```text
https://<hf-user>-truthshield-ai-backend.hf.space/api/health
```

## 2. Deploy The Frontend To Cloudflare Pages

Set the frontend production environment variable:

```text
VITE_API_BASE_URL=https://<hf-user>-truthshield-ai-backend.hf.space
```

If you deploy from the Cloudflare dashboard:

- Connect the GitHub repo
- Root directory: `frontend`
- Framework preset: Vite
- Build command: `npm run build`
- Build output directory: `dist`
- Environment variable: `VITE_API_BASE_URL`

If you deploy from the CLI:

```powershell
Set-Location .\frontend
$env:VITE_API_BASE_URL = "https://<hf-user>-truthshield-ai-backend.hf.space"
npx wrangler whoami
npm run deploy
Remove-Item Env:VITE_API_BASE_URL
```

Direct-upload Pages projects do not deploy automatically when GitHub changes. Either run the command above after frontend changes or connect the Pages project to this repository with `frontend` as its root directory.

The Pages URL will look like:

```text
https://truthshield-ai-frontend.pages.dev
```

## 3. Lock Backend CORS To Your Frontend

After Cloudflare Pages gives you the final frontend URL, set this Space variable:

```text
FRONTEND_ORIGINS=https://truthshield-ai-frontend.pages.dev
```

If you add a custom domain later, append it with a comma:

```text
FRONTEND_ORIGINS=https://truthshield-ai-frontend.pages.dev,https://www.your-domain.com
```

## 4. Final Smoke Test

Open the deployed frontend and run:

- Image and video analysis with representative test files
- Image upload with a small `.jpg` or `.png`
- A known AI regression image; inspect the three-way verdict and the raw detector score separately
- A known real street/building photograph; a raw score inside the 0.15–0.95 abstention band must render as `Inconclusive`, not `Likely AI`
- Health check directly at `/api/health`

If the frontend loads but requests fail, first check:

- Cloudflare Pages environment variable `VITE_API_BASE_URL`
- Hugging Face Space build logs
- Hugging Face Space variable `FRONTEND_ORIGINS`
- Browser devtools console for CORS or network errors
- `Technical evidence -> Detector outputs`; if it says only `Local heuristic fallback`, the trained model was not packaged or could not load and the image verdict must be inconclusive
