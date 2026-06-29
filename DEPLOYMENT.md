# Deployment

TruthShield AI deploys as two separate services:

- Frontend: Cloudflare Pages from `frontend/`
- Backend: Hugging Face Spaces Docker Space from `backend/`

## 1. Deploy The Backend To Hugging Face Spaces

Create a new Hugging Face Space:

- Owner: your Hugging Face username or org
- Space name: `truthshield-ai-backend`
- SDK: Docker
- Visibility: public or private

The Space root should contain the contents of this repo's `backend/` folder, including `Dockerfile`, `README.md`, `requirements.txt`, `main.py`, `analyzers/`, `models/`, and `utils/`.

With Git:

```powershell
git clone https://huggingface.co/spaces/<hf-user>/truthshield-ai-backend hf-truthshield-backend
Copy-Item -Recurse -Force .\backend\* .\hf-truthshield-backend\
Set-Location .\hf-truthshield-backend
git add .
git commit -m "Deploy TruthShield backend"
git push
```

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
npm run build
npx wrangler login
npx wrangler pages deploy .\dist --project-name=truthshield-ai-frontend
Remove-Item Env:VITE_API_BASE_URL
```

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

- Text scan with one of the demo examples
- Image upload with a small `.jpg` or `.png`
- Health check directly at `/api/health`

If the frontend loads but requests fail, first check:

- Cloudflare Pages environment variable `VITE_API_BASE_URL`
- Hugging Face Space build logs
- Hugging Face Space variable `FRONTEND_ORIGINS`
- Browser devtools console for CORS or network errors
