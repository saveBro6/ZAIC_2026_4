# Agent Directive

## Environment & Runtime (Python)
- **Tooling Mandate**: You must strictly use `uv` for all Python operations. Do not use standard `python3`, `python`, `pip`, `venv`, `conda`, or `poetry` unless explicitly instructed otherwise.
- **Execution Rules**:
  - Run scripts using: `uv run <script_name>.py` or `uv run python -c ...`
  - Request install new dependencies if not found using: `uv add <package>` or `uv pip install <package>`
- **Performance Priority**: Leverage `uv`'s caching mechanisms to ensure ultra-fast execution of pipelines and autonomous tasks.

### Agent Skill: Polars Data Operations (CSV & Excel)

**Trigger:** Reading, parsing, filtering, transforming, or aggregating `.csv` and `.xlsx` tabular data.

**Core Rules:**

1. **Mandatory Imports:**
   Always use Polars as the primary engine:
   ```python
   import polars as pl
   ```

2. **Lazy Evaluation for Large CSVs:**
   * Default to `pl.scan_csv()` instead of `pl.read_csv()` when dealing with large datasets or complex pipelines to allow predicate and projection pushdown.
   * Finalize lazy execution with `.collect()` (or `.collect(engine="streaming")` if required):
     ```python
     q = (
         pl.scan_csv("data.csv")
         .filter(pl.col("status") == "active")
         .select(["id", "created_at", "amount"])
     )
     df = q.collect()
     ```

3. **Multi-Sheet Excel Processing:**
   * Use `fastexcel` (Polars' default backend) to inspect sheet names in milliseconds without parsing the entire file:
     ```python
     import fastexcel
     excel_reader = fastexcel.read_excel("data.xlsx")
     sheet_names = excel_reader.sheet_names
     ```
   * Read the required sheet (or iterate over sheets) with Polars:
     ```python
     df = pl.read_excel("data.xlsx", sheet_name=sheet_names[0])
     ```

4. **Expression-First Computations:**
   * Never iterate over rows or use Python-level lambdas (`apply()`) unless strictly unavoidable.
   * Handle Vietnamese column names with quotes: `pl.col("Tên sản phẩm")`.
   * Write transforms inside native execution contexts:
     * Modifications: `df.with_columns(...)`
     * Subsetting: `df.select(...)` and `df.filter(...)`
     * Aggregations: `df.group_by(...).agg(...)`

---

## Strict Execution Lifecycle & Transition Gates

Every code modification task must strictly adhere to the sequential lifecycle below. Skipping or reordering steps is strictly prohibited:

1. **Phase 1: Implementation**
   - Apply source code edits according to the user's requirements and prompt constraints.

2. **Phase 2: Code Review Gate (MANDATORY)**
   - **Trigger:** Immediately upon completing code modifications, BEFORE writing any walkthrough/summary.
   - **Action:** Delegate execution to the `code-reviewer` agent (defined in `.agents/agents/code-reviewer/agent.md`).
   - **Inspection Scope:**
     * Run `git status --short` and `git diff` (including `--cached` if staged).
     * Verify 100% diff alignment against the initial prompt to detect omissions, unintended changes, or scope creep.
     * Perform deep static analysis for logic flaws, edge cases, regression risks, broken contracts, and stray debug code.
     * Execute and verify the relevant test suites within the reviewer context.
   - **Gate Rule:**
     * If verdict is **`[BLOCKED]`**: Halt forward progress immediately. Return to Phase 1 to patch the flagged issues, then re-trigger Phase 2.
     * Proceed to Phase 3 ONLY when verdict is explicitly marked **`[PASSED]`**.

3. **Phase 3: Fast Linter Check**
   - Run linter verification on modified files: `uv run ruff check <modified_files>`.
   - **Strict No-Re-run Rule**: Do NOT re-run `pytest` in this phase since `code-reviewer` has already executed and passed the full test suite.
   - If linter errors or formatting issues occur, fix them before moving forward.

4. **Phase 4: Summary & Walkthrough**
   - **Entry Condition:** Phase 2 is **`[PASSED]`** AND Phase 3 linter check passes cleanly.
   - Synthesize changes, explain implementation details, and produce the final user-facing summary walkthrough.

---

### Non-Negotiable Directives
- **Zero-Bypass Policy:** Never generate summary artifacts or declare task completion without an explicit `[PASSED]` verdict from the `code-reviewer` agent.
- **No Duplicate Test Runs:** Never re-run `pytest` after `code-reviewer` has already executed tests and returned `[PASSED]`.
- **Context Preservation:** When handing off to `code-reviewer`, pass the exact verbatim user prompt along with relevant issue/ticket context to serve as the ground truth.
- **No Data Leakage / Zero Hardcoding Policy:**
  - Strictly prohibit hardcoding rules, string literals, column mappings, or conditional logic (`if-else`) tailored to specific datasets, shops, or ground truth files (e.g. from `train` split).
  - All query generation, schema linking, and error correction must remain 100% data-agnostic and generalize to unseen private evaluation sets.
  - Rely exclusively on runtime database engine feedback (SQLite errors), dynamically extracted schema metadata, and in-context learning—never hardcoded heuristic patches or code suggestions in prompt templates.

---

## Coding & Output Style
- Code comment, docstring must be written in English.
- Do not use separator lines, banner dividers, or rows of equal signs (=====).
- Keep the code, terminal and chat outputs clean and concise.
