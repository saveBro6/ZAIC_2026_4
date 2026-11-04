# Problem Contract: DataMate (Zalo AI Challenge 2026)

---

## 1. Executive Summary & Objective

**DataMate** is a challenge in the Zalo AI Challenge 2026 competition that requires building an autonomous offline AI agent capable of answering natural-language Vietnamese business questions over multi-table structured tabular data (CSV, Excel, SQLite), anchored in the context of Vietnamese Small and Medium Businesses (SMBs / hộ kinh doanh).

The agent receives raw business data files, a business description (`README.md`), a natural-language Vietnamese question, a reference evaluation date (`as_of_date`), a solar–lunar calendar (`calendar.csv`), and a domain rulebook (`rulebook.md`). The agent must either compute the exact answer adhering strictly to data schemas and business rules, declare the question `unanswerable` if data is insufficient, or deliberately `abstain` to avoid penalties.

---

## 2. Formal Problem Formulation & Pipeline

### 2.1 Input Tuple
For every question $i \in \{1, \dots, N\}$, the agent is provided:
$$\mathcal{Q}_i = \langle \text{id}_i, \text{business\_id}_i, \text{question}_i, \text{as\_of\_date}_i \rangle$$
Contextualized by:
1. **Business Data Directory**: `<split>/data/<business_id>/` containing:
   - Structured tabular files: `*.csv`, `*.xlsx`, `*.sqlite`.
   - Documentation: `README.md` (data dictionary, notes, schema explanations; quality ranges from detailed to minimal).
2. **Split-wide Dynamic Rulebook**: `<split>/rulebook.md` detailing default business terminology, revenue/cost definitions, rankings, and filters for that specific split.
3. **Solar–Lunar Calendar**: `<split>/calendar.csv` mapping dates to Vietnamese lunar dates and holidays (especially Lunar New Year / Tết).

### 2.2 Output Tuple
For every question $i$, the agent must output a single JSON object in `submission.jsonl`:
$$\mathcal{A}_i = \langle \text{id}_i, \text{type}_i, \text{answer}_i [, \text{reason}_i] \rangle$$
Where $\text{type}_i \in \{\texttt{"number"}, \texttt{"percent"}, \texttt{"string"}, \texttt{"date"}, \texttt{"boolean"}, \texttt{"list"}, \texttt{"table"}, \texttt{"unanswerable"}, \texttt{"abstain"}\}$.

---

## 3. Data Specification & Directory Hierarchy

### 3.1 Directory Layout
```text
<split>/
├── data/
│   └── <business_id>/
│       ├── *.csv | *.xlsx | *.sqlite        # Tabular data files
│       └── README.md                        # Business description, data dictionary, or informal notes
├── rulebook.md                              # Split-specific business rules & vocabulary (DYNAMIC!)
├── calendar.csv                             # Solar–Lunar calendar mapping (Tết, lunar dates)
├── questions.jsonl                          # One test question per line
└── answers.jsonl                             # Ground truth reference (Training split only)
```

### 3.2 Dataset Splits & Volume

| Split | Businesses | Question Count | Ground Truth Provided | Purpose / Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Training** | 30 | ~3,000 | Yes (`answers.jsonl`) | Development, offline validation, fine-tuning, prompt tuning. |
| **Public Test** | 10–20 | ~1,000 | No | Round 1 public leaderboard feedback (max 5 submissions/day). |
| **Private Test** | 10–20 | ~500 | No | Final evaluation. Includes **unseen business types**. |

### 3.3 Business Profiles & Characteristics
- **Domains**: Grocery stores (tạp hóa), coffee/bubble tea shops (trà sữa, cà phê), restaurants/quán ăn, pharmacies (nhà thuốc), building materials (vật liệu xây dựng), motorbike repair shops (sửa xe máy), mother-and-baby stores (mẹ và bé), and novel domains in the private test.
- **Time Horizon**: 12–30 months of historical transactions per business.
- **Data Scale**: Ranges from a few hundred rows to ~300,000 rows per business.
- **Data Formats**: Mixed multi-file formats (`.csv`, `.xlsx`, `.sqlite`) within single businesses or across splits. Multiple sheets, varying encodings, Vietnamese column headers, accented and non-accented strings.
- **README Quality**: Inconsistent; some offer structured schemas, others are sparse informal notes.
- **Seasonality & Events**: Tết Nguyên Đán, Mid-Autumn (Trung Thu), public holidays, salary paydays, weekend surges.

### 3.4 Data Schema Details

#### `questions.jsonl`
```json
{"id": "q_01234", "business_id": "biz_017", "question": "thang nay hang nao ban cham nhat, ton nhieu ma ban ko dc", "as_of_date": "2026-08-27"}
```
- `id` (string): Unique identifier for the question.
- `business_id` (string): Identifies the folder under `<split>/data/<business_id>/`.
- `question` (string): Natural-language Vietnamese query (colloquial, abbreviation-heavy, with or without tone marks).
- `as_of_date` (string, `YYYY-MM-DD`): The virtual "today" against which all relative temporal references are grounded.

#### `answers.jsonl` (Training Only)
```json
{"id": "q_06de33fa21cc", "type": "string", "answer": "Xi măng Hà Tiên PCB40 bao 50kg", "alias": ["SP0001"]}
{"id": "q_01241", "type": "unanswerable", "answer": null, "reason": "Dữ liệu không có giá vốn nên không tính được lợi nhuận"}
```
- Includes accepted string aliases (`alias`), which include product codes, alternative naming variants from the database, etc.

#### `calendar.csv`
Contains solar-to-lunar conversion, lunar month/day, and holiday tags to resolve questions referencing lunar events (e.g., Tết, rằm, mùng một).

#### `rulebook.md` (Critical Requirement)
- Each split ships its **own** `rulebook.md`.
- **Dynamic Rulebook Policy**: The agent must inspect and apply the `rulebook.md` of the current active split at runtime. **Hardcoding rules from the training rulebook is strictly prohibited and prone to catastrophic domain shift.**

---

## 4. Answer Types & Strict Matching Rules

Every generated answer is compared against reference ground truth strictly according to its data type:

| Type | Target Content | Example | Evaluation & Matching Contract |
| :--- | :--- | :--- | :--- |
| **`number`** | Numeric values (VND amounts, counts, units, weights) | `48250000` | **Exact match**. Reference answers are integers (for monetary amounts, counts, quantities) or rounded to 2 decimal places. |
| **`percent`** | Percentage value (in percent, not ratio) | `-12.45` | **Exact match rounded to 2 decimal places**. The sign (`+` or `-`) must be strictly correct. E.g., `-12.45%` is written `-12.45`. |
| **`string`** | Names (product, customer, branch, day of week, etc.) | `"Sting dâu 330ml"` | **Normalized Exact Equality**. Both answer and reference/alias strings undergo: <br>1. Unicode NFC normalization.<br>2. Lowercase conversion.<br>3. Non-(letter/digit/whitespace) replaced with space.<br>4. Whitespace trimmed and runs collapsed to a single space.<br>*(Note: Vietnamese diacritics are preserved).*<br>Matches if normalized answer equals normalized reference or any item in `alias` (e.g. `sp0042`). |
| **`date`** | ISO format temporal answer | `"2026-08-15"` or `"2025-09"` | **Exact match**. Format `YYYY-MM-DD`. If question asks about a month: format `YYYY-MM`. |
| **`boolean`** | Logical true/false | `true` or `false` | **Exact match** (`true` or `false`). |
| **`list`** | Ordered sequence of names or numbers | `["Cà phê sữa", "Bạc xỉu", "Trà đào"]` | **Position-by-position exact match**. Must match the exact ordering requested by question or default rulebook order. Elements are scored via `string` or `number` matching rules. |
| **`table`** | Tabular summary (columns and rows) | `{"columns": ["khach_hang", "so_tien_no"], "rows": [["Chị Lan", 3200000], ["Anh Tú", 1500000]]}` | Columns are matched by normalized name (fallback to positional order if names do not match). Rows are evaluated **as an unordered set**. Cells matched via `string` or `number` rules. |
| **`unanswerable`** | Missing data required for resolution | `{"type": "unanswerable", "reason": "..."}` | **Binary correct/wrong**. Correct if and only if ground truth reference is `unanswerable`. Any answer provided for an unanswerable question is penalized as `wrong`. `reason` field is optional. |
| **`abstain`** | Deliberate refusal to answer | `{"type": "abstain"}` | **Neutral score (0 points)**. Always receives 0 regardless of ground truth. Any omitted question in `submission.jsonl` defaults to `abstain`. |

### 4.1 Ranking & Tie-Breaking Rules
- **Criterion**: As stated in the question; if unspecified, use the default criterion in `rulebook.md` (e.g., "bán chạy" defaults to quantity sold, descending).
- **Scope**: Computed over valid orders only (excluding cancelled/returned orders according to the rulebook).
- **Secondary Tie-breaker**: Elements with identical values are sorted by **item code ascending** (`code ASC`).
- **Boundary Guarantees**: Test sets guarantee that no ties exist at rank $k$ (or rank 1 for superlative questions like "... nhất").
- **Small Candidate Sets**: If fewer than $k$ matching candidates exist in the data, the reference answer is the complete list of all available candidates.

---

## 5. Evaluation Metric & Penalized Accuracy Mechanics

The official leaderboard metric across the $N$ questions in a test set is **Penalized Accuracy (PA)**:

$$\text{PA} = \frac{1}{N} \sum_{i=1}^{N} s_i$$

Where per-question score $s_i$ is defined as:
$$s_i = \begin{cases}
+1 & \text{if answer } i \text{ is \textbf{correct}} \\
0 & \text{if answer } i \text{ is \textbf{abstain}} \text{ (or missing)} \\
-1 & \text{if answer } i \text{ is \textbf{wrong}}
\end{cases}$$

### 5.1 Critical Scoring Dynamics & Risk Management
1. **Range**: $\text{PA} \in [-1.0, +1.0]$.
2. **Asymmetric Penalty for Guessing**:
   - Guessing randomly or hallucinating incurs $-1$.
   - Abstaining secures $0$.
   - **Breakeven Threshold**: Expected value of answering is $\mathbb{E}[s] = P(\text{correct}) \cdot (+1) + (1 - P(\text{correct})) \cdot (-1) = 2 \cdot P(\text{correct}) - 1$.
   - Answering is mathematically beneficial **if and only if** estimated confidence $P(\text{correct}) > 0.50$ (50%).
3. **Unanswerable Trap**: Providing any value to an `unanswerable` question yields $-1$. Correctly identifying `unanswerable` yields $+1$.
4. **Tie-Breaking on Leaderboard**:
   - Priority 1: Lower total inference wall-clock time in the organizers' re-run.
   - Priority 2: Earlier submission timestamp.

---

## 6. Business Logic & Default Rulebook Conventions

Unless overridden by the explicit phrasing of a question, the agent must adhere to the default conventions defined in `rulebook.md`:

1. **Temporal Grounding (`as_of_date`)**:
   - `"tháng này"`: From the 1st of the month of `as_of_date` up to and including `as_of_date` (not the full calendar month if `as_of_date` is mid-month).
   - `"tuần này"` / `"tuần rồi"`: Standard calendar week runs **Monday to Sunday**.
   - `"cùng kỳ năm ngoái"`: Equivalent historical window relative to `as_of_date` in the previous year.
2. **Revenue Calculation**:
   - Total of line-item amounts after line-level discounts.
   - **Do not deduct order-level vouchers**.
   - **Exclude VAT and delivery/shipping fees**.
   - **Exclude cancelled orders** and orders containing any returned items.
3. **Product Popularity ("bán chạy")**:
   - Default metric is **total sales quantity** (not revenue or transaction frequency).
4. **Inventory Valuation & Costing**:
   - Default inventory cost is calculated as the **lot-based weighted-average purchase cost** (giá vốn bình quân gia quyền theo lô).
5. **Lunar Calendar & Holidays**:
   - Tết window: From **23 tháng Chạp** (23rd day of 12th lunar month) to **mùng 10 tháng Giêng** (10th day of 1st lunar month).
   - Lunar queries must be resolved using `<split>/calendar.csv`.

---

## 7. Technical, Environmental & System Constraints

All submissions are executed inside a standalone Docker container verified by organizers under strict conditions:

### 7.1 Hardware & Compute Limits
- **GPU**: $2 \times$ NVIDIA RTX 5090 (32 GB VRAM each $\rightarrow$ 64 GB total VRAM). Blackwell architecture (requires CUDA $\ge 12.8$, PyTorch/vLLM compatible builds).
- **RAM**: 512 GB System RAM.
- **Wall-Clock Time Budget**: $\le 6$ hours total inference time for the entire private test set (~500 questions across 10–20 businesses), **including model initialization and weight loading**.
  - Average permissible time per question: $\sim 40$ seconds.

### 7.2 Model Parameter Constraints (Hard Limit)
- **Total Combined Parameter Ceiling**: $\le \mathbf{15\text{ Billion}}$ parameters ($15\text{B}$).
  - Encompasses **all** inference models loaded or used across the entire pipeline: LLMs, embedding models, rerankers, classifiers, etc.
  - Counted whether loaded simultaneously or sequentially.
  - For Mixture-of-Experts (MoE) architectures, **all parameters** (total sparse + routing parameters) are counted, not just active parameters per token.
- **Allowed Quantization**: Any quantization scheme is permitted (e.g., INT4, FP4, AWQ, GPTQ, GGUF).
- **Eligible Base Models**: Weights must have been publicly released before **October 1, 2026** under an open/permissive license permitting such usage.
- **Declaration**: All models used must be explicitly disclosed in the solution `README.md`.

### 7.3 Operational & Security Sandbox
- **Zero External Network Access (`--net=none`)**: Completely offline inference. No API calls, no online downloads, no telemetry.
- **Read-Only Data Mount**: The test data directory `/data` is mounted strictly **read-only**. Any temporary files, index caches, or intermediate artifacts must be written to `/tmp`, `/result`, or local working directories.
- **No Human-In-The-Loop & No Cheating**:
  - Fully automated execution without manual intervention.
  - Hard-coded answers or conditional logic keyed directly on `id` or `business_id` are strictly prohibited.
  - Public/private test sets must not be used for fine-tuning or training.
- **Code Execution**: The agent is allowed and encouraged to synthesize and execute dynamic code (Python, SQLite, etc.) within the container.

---

## 8. Docker Specification, Interface & Output Contracts

### 8.1 Execution Command
The container is invoked by the evaluation harness as follows:
```bash
docker run --gpus all --net none \
  -v /path/to/test_data:/data:ro \
  -v /path/to/result_dir:/result:rw \
  <image_name> /bin/bash /app/predict.sh
```

### 8.2 File System Contract
1. `/data`: Mounted read-only input folder containing:
   - `/data/data/<business_id>/*`
   - `/data/rulebook.md`
   - `/data/calendar.csv`
   - `/data/questions.jsonl`
2. `/result/submission.jsonl`: Output answer file, line-delimited JSON.
3. `/result/logs/`: Per-question trace log directory. For each question, must log:
   - Question ID.
   - Final predicted answer & type.
   - Elapsed wall-clock time.
   - Diagnostic trace (generated code, SQL queries, intermediate execution outputs, error traces).

### 8.3 Submission File Example (`/result/submission.jsonl`)
```jsonl
{"id": "q_01234", "type": "string", "answer": "Sting dâu 330ml"}
{"id": "q_01235", "type": "number", "answer": 48250000}
{"id": "q_01236", "type": "percent", "answer": -12.45}
{"id": "q_01237", "type": "date", "answer": "2026-08-15"}
{"id": "q_01238", "type": "boolean", "answer": true}
{"id": "q_01239", "type": "list", "answer": ["Cà phê sữa", "Bạc xỉu", "Trà đào"]}
{"id": "q_01240", "type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["Chị Lan", 3200000], ["Anh Tú", 1500000]]}}
{"id": "q_01241", "type": "unanswerable", "reason": "Dữ liệu không có giá vốn nên không tính được lợi nhuận"}
{"id": "q_01242", "type": "abstain"}
```

### 8.4 Reproducibility Tolerance & Verification
- Final official score is determined by the **organizers' re-run** of the submitted Docker image.
- **Reproducibility Criterion**:
  $$|\text{PA}_{\text{uploaded}} - \text{PA}_{\text{re-run}}| \le 0.03$$
- If difference exceeds $0.03$, organizers inspect `/result/logs/`. If discrepancy cannot be justified, entry is disqualified.
- If re-run exceeds 6 hours or crashes, only answers flushed to `/result/submission.jsonl` are evaluated; unfinished questions are treated as `abstain`.

---

## 9. Competition Schedule & Key Milestones

| Phase / Milestone | Timeline (GMT+7) | Key Action Items & Protocols |
| :--- | :--- | :--- |
| **Round 1: Public Test** | Oct 26 – Nov 16, 2026 | Build pipelines, test on public test set, submit `submission.jsonl` (max 5/day). |
| **Docker Solution Freeze** | **Nov 16, 2026 @ 23:59** | Submit complete Docker image (code, models, weights, `predict.sh`) via public Google Drive link + SHA256 checksum. Max 2 submissions (last one evaluated). |
| **Round 2: Private Test** | Nov 20 (09:00) – Nov 23 (23:59) | Download private test set, run frozen solution locally, upload `submission.jsonl`. |
| **Verification & Re-run** | Nov 24 – Dec 1, 2026 | Organizers run Docker containers on reference 2x RTX 5090 machines, verify code, logs, and reproducibility. |
| **Final Winners Announcement** | **Dec 2, 2026** | Announcement of Top 2 winning teams. |

---

## 10. Recommended System Architecture & Engineering Principles

To succeed under this contract, the agent solution must be designed around:
1. **Dynamic Rulebook & Schema Ingestion**: Parse `<split>/rulebook.md` and SQLite/CSV schemas dynamically per run; avoid rigid prompt hardcoding.
2. **Code Generation / Tool Use (Text-to-SQL / Text-to-Python)**: Generating and executing code against temporary in-memory SQLite databases or Pandas/Polars DataFrames is the most accurate approach for large tabular files (up to 300,000 rows).
3. **Safe Multi-Format Data Loading**: Robust connectors handling `.csv`, `.xlsx` (multi-sheet, headers, merged cells), and `.sqlite`.
4. **Calibrated Abstention Policy**: Confidence calibration mechanism to selectively output `abstain` whenever code execution fails, schemas are ambiguous, or model confidence $\le 0.50$.
5. **Strict Type Formatting & Post-Processing**: Dedicated validators ensuring exact type formatting (ISO dates, NFC normalized strings, two-decimal rounding, valid list order, table JSON structure).
6. **Graceful Timeout & State Flushing**: Write answers line-by-line into `/result/submission.jsonl` immediately upon solving each question, guaranteeing non-zero scores even in case of timeout.
