# Daily postdoc portfolio project — instructions for Claude

You are building ONE complete, public, portfolio-grade research software project in a single session,
for **Degala Chenchupradeep** (GitHub: pradeepdegalaroyal), a 2nd-year PhD scholar in Quantum Machine Learning at SRM University AP
who is targeting postdocs at MIT, Stanford, Oxford and Cambridge and research-scientist roles at top quantum labs.
The project must show the hiring PI / reviewer that he can contribute to *their* research agenda from day one.

Today: 2026-10-22. Inputs (read all of them first):
- `.brief/JOB.md` — the target position and its research theme
- `.brief/papers.md` and `.brief/papers.json` — recent Scopus literature (titles, dates, DOIs, citations, abstracts)
- `.brief/gaps.md` — limitation/open-problem sentences mined from those abstracts
- `.brief/PROFILE.md` — the candidate's background and existing repositories (do NOT duplicate them)

## Time box — this matters most
This is a **focused mini research study**, not a full postdoc project. Aim to finish everything in about
**60 minutes of wall-clock time**. The candidate is using this PC for other work at the same time, so:
- Keep experiments small: total experiment compute **<= 15 minutes**. Use small qubit counts / problem sizes,
  few seeds (3-5), and statevector or light noisy simulation. Never start a run you expect to take > 5 minutes.
- Limit resources: `torch.set_num_threads(6)`, at most ~5 GB of GPU memory, no background servers or watchers.
- Prefer one sharp, well-presented result over many half-finished ones. Do not over-engineer.

## Machine
Windows 11, Python 3.11.9, NVIDIA RTX 5050 Laptop GPU (8 GB, CUDA 12.8), 16 CPU threads.
Already installed: qiskit 2.4.2, qiskit-aer 0.17.2, qiskit-optimization 0.7.0, pennylane 0.45.1 (+lightning),
torch 2.11 (cu128), torch-geometric, numpy/scipy/pandas/matplotlib/networkx/scikit-learn.
Use these; do not pip install large packages. Use the shell's `python` (not `python3`).

## Plan — execute every step
1. **Research gap** → `docs/RESEARCH_GAP.md` (short, ~1 page)
   - Summarise the state of the art from the provided papers (cite as Author et al., Year, DOI).
   - Identify ONE concrete, testable gap that matches the job's theme and is feasible today on this laptop
     (simulation scale is fine — be explicit about it). State a falsifiable hypothesis and the success metric.
2. **Design** → `docs/METHOD.md` (short): method, one classical baseline AND the standard quantum baseline
   from the literature, metrics, one ablation, one noise/robustness sweep, seeds.
3. **Implement** as a clean package: `src/<package>/` (typed, docstrings), `experiments/run_all.py`
   (one command reproduces everything; `--quick` flag for a < 1 minute smoke run), `tests/` (a few fast pytest
   tests), `.github/workflows/ci.yml` (installs CPU deps and runs pytest + `run_all.py --quick` on ubuntu-latest),
   `requirements.txt` (pinned to the versions above), `pyproject.toml`.
4. **Run the experiments for real** (within the time box). Save raw results to `results/*.csv|json` and
   figures to `figures/*.png` (labelled axes, 200 dpi). Report mean ± std over seeds. Make the tests pass.
5. **README.md** (this is what PIs and recruiters read first):
   title, badges (license MIT, python, CI), 3-sentence abstract, motivation, research gap with citations,
   method diagram (mermaid) and summary, **results table + figures using only numbers from `results/`**,
   honest discussion (including negative results), limitations, future work (how it scales to real hardware /
   the PI's lab setting), reproducibility (exact commands, runtime, hardware), project structure, citation,
   license. Mention related work by the author: link 1-2 of his relevant repos from PROFILE.md.
6. `CITATION.cff` (author: Degala Chenchupradeep), keep the existing MIT `LICENSE` and `.gitignore` as they are.
7. Write `.brief/REPO.json`: {"name": "kebab-case-repo-name (<= 50 chars, descriptive, no dates)",
   "description": "<= 300 chars", "topics": ["up to 8 lowercase github topics"]}
8. Write `.brief/CV_ENTRY.json`: {"title": "short project title (<= 70 chars)",
   "summary": "one sentence (<= 200 chars) with the method and the key measured result from results/"}
9. Write `.brief/SUMMARY.md` (for the candidate only, not published): what was built, the key measured result,
   how it maps to the target job, 3 interview talking points, and 2 sentences he could use in a cover letter.

## What PIs, reviewers and industry research labs look for (meet all of these)
- A sharp research question tied to an open problem in recent literature — not a tutorial re-implementation.
- Fair, strong baselines; ablations; multiple seeds; error bars; scaling study (qubits / problem size / noise).
- Noise-aware, hardware-realistic thinking (NISQ constraints, shot noise, transpilation depth, loss budgets).
- Reproducible engineering: one-command run, pinned deps, tests, CI, small clean modules, type hints.
- Clear, honest communication: figures that make the point, limitations stated up front.

## Integrity rules (non-negotiable)
- Never invent results, citations or DOIs. Every number in README must come from a file in `results/`.
- Never claim quantum advantage unless your own measurements show it; frame simulation results as such.
- Do not copy text from papers or from the job advert; do not name the employer or say this is for an application.
- Never write API keys, tokens or personal contact details into any file.
- Write as the author's own project: no "generated by" notes, AI-assistant credits or co-author lines in any file.
- Do NOT run `git init`, `git commit` or `git push` and do not use `gh` — the runner publishes after you finish.
- Stay inside this folder.
