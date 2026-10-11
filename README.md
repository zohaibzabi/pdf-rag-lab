# PDF RAG Lab (Python + Streamlit)

A beginner-friendly RAG demonstration for an AI class. Students upload a PDF, build an index, and ask questions grounded in retrieved passages.

## Free-resource design

- **PDF reading:** `pypdf` (free/open source).
- **Retrieval/indexing:** `scikit-learn` TF-IDF (free/open source, no paid vector database, no embedding API).
- **Frontend:** Streamlit Community Cloud (free hosting option; platform limits apply).
- **Answer generation:** `openai/gpt-oss-20b` served through Groq's API. Create a Groq account and API key. The provider has a free tier, but it has rate/token quotas; it is **not unlimited**. Check the current Groq limits before a class lab.

The model is hosted by Groq; this repository does not download model weights. You should not need to enter payment details for use within the available free tier, but availability and provider terms can change. Do not enable paid billing if you intend to use only free resources.

> No hosted GPT/LLM service can responsibly be promised as free and unlimited. For genuinely API-quota-free use, run an open-weight model locally with a local inference tool such as Ollama; that requires your own computer's RAM/CPU/GPU and is a different setup from Streamlit Community Cloud.

## Files

- `app.py` — Streamlit frontend, upload flow, chat, retrieved evidence display.
- `rag_engine.py` — PDF parsing, chunking, TF-IDF retrieval, and Groq generation.
- `requirements.txt` — packages installed by Streamlit Cloud.
- `.streamlit/config.toml` — basic app settings.
- `.gitignore` — excludes local secrets and Python cache files.

## 1. Create a free Groq API key

1. Open [Groq Console](https://console.groq.com/) and sign in/create an account.
2. Create an API key in the API keys area and copy it somewhere safe.
3. Never paste the key into Python code or commit it to GitHub.
4. The default model is `openai/gpt-oss-20b`. Model availability and free-tier quotas may change; see the [Groq models page](https://console.groq.com/docs/models) and [rate limits](https://console.groq.com/docs/rate-limits).

## 2. Run on your own computer (Windows)

Install Python 3.11 or 3.12, then open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:GROQ_API_KEY = "paste-your-key-here"
streamlit run app.py
```

Your browser should open the local app. Upload a text-based PDF, click **Build RAG index**, then ask a question in the chat box.

On macOS/Linux, activate with `source .venv/bin/activate` and set the key with `export GROQ_API_KEY="paste-your-key-here"` before running `streamlit run app.py`.

## 3. Upload the project to GitHub

1. Sign in to [GitHub](https://github.com/) and create a new repository, e.g. `pdf-rag-lab`.
2. Choose Public or Private according to your class needs.
3. Upload the files and folders from this project. Keep `app.py`, `rag_engine.py`, and `requirements.txt` at the repository root.
4. Do **not** upload your API key or `.streamlit/secrets.toml`. The included `.gitignore` excludes that secrets file.
5. Commit the files to the `main` branch.

If you use Git locally, run these commands from inside this project folder:

```bash
git init
git add .
git commit -m "Add PDF RAG lab app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/pdf-rag-lab.git
git push -u origin main
```

Replace `YOUR-USERNAME` and the repository name with your own values. If Git asks you to authenticate, follow GitHub's current authentication process.

## 4. Deploy from GitHub to Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
2. Select **Create app** / **New app**.
3. Choose your repository, branch `main`, and main file path `app.py`.
4. Open **Advanced settings** (or the app's **Secrets** settings).
5. Add this TOML secret, replacing the placeholder with your real key:

   ```toml
   GROQ_API_KEY = "gsk_your_real_key_here"
   ```

   Optional: to use another currently supported model, add `GROQ_MODEL = "model-id"` after checking the provider's current models page.
6. Click **Deploy**. Streamlit installs `requirements.txt` and builds the app.
7. After deployment, open the generated `*.streamlit.app` URL and test with a small text-based PDF.

Never share a screenshot that reveals your API key. The secret should be entered in Streamlit's settings, not committed to GitHub.

## 5. How the RAG pipeline works

1. **Ingestion:** `pypdf` extracts selectable text from each PDF page.
2. **Chunking:** each page is split into chunks of about 180 words with 35 words of overlap.
3. **Indexing:** `TfidfVectorizer` creates sparse term-weight vectors for the chunks.
4. **Retrieval:** the question is transformed into the same vector space; the app selects the top-scoring chunks.
5. **Augmentation:** the selected chunks and page numbers are attached to the prompt.
6. **Generation:** the hosted language model writes an answer using the provided context.

**Important terminology:** this app does not train or fine-tune the GPT model. In a standard RAG workflow, the document is ingested and indexed at runtime; model weights are unchanged.

## Troubleshooting

- **No text extracted:** the PDF may be a scan. OCR it first, then upload the searchable PDF.
- **Rate limit / HTTP 429:** the free API quota may have been reached. Check Groq's current rate limits and retry after reset.
- **Invalid model:** open Groq's supported models page and set `GROQ_MODEL` to a currently supported model.
- **Missing API key:** check Streamlit app settings → Secrets, or set `GROQ_API_KEY` in your local terminal.
- **Large PDFs:** begin with a smaller PDF for classroom demos. Streamlit upload limits and free-host resource limits apply.
- **Privacy:** uploaded PDF bytes are processed by the app, and relevant text chunks are sent to Groq for answer generation. Do not upload confidential student, client, medical, or other sensitive material.

## Official deployment references

- [Deploy an app on Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit Community Cloud secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
- [Groq supported models](https://console.groq.com/docs/models)
- [Groq rate limits](https://console.groq.com/docs/rate-limits)
