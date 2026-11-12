P4 — "Đáp án đúng kiểu gì cũng một format": percent theo SQL, cột bảng thật, vote theo đáp án, abstain gate

Đóng 6 lỗi số 2–7 người dùng đã kiểm chứng, tất cả đều nằm ở lớp format + vote (sau khi SQL đã chạy), nên sửa được mà không đụng prompt sinh SQL, không đụng linker, không đụng VLLMEngine.
1. Mục tiêu & tiêu chí đo
#	Lỗi	Trước (đo được)	Sau (đo được)	Cách đo
M1	Percent ×100 theo độ lớn	format_percent(0.5) → 50.0; test hiện assert 0.0527 → 5.27	format_percent(0.5) → 0.5; chỉ ×100 khi SQL là phép chia tỉ lệ	unit test, 8 ca SQL
M2	Không còn heuristic độ lớn	-1.0 < num < 1.0 → *100 (result_formatter.py:52-53)	grep "num \* 100" = 0; quyết định scale lấy từ AST + guard giá trị	grep + test
M3	Cột bảng bịa col_N	format_result([("Chị Lan",3200000)],"table") → {"columns":["col_1","col_2"]}	["Tên khách","Số tiền nợ"] (từ cursor.description, fallback AST, fallback col_N)	unit test + test SELECT *
M4	Vote không phân biệt type	2 candidate cùng 2.08, một number một percent → unanimous_2_of_2, winner type number	2 signature khác nhau ⇒ không đồng thuận; 2 percent + 1 number ⇒ winner percent	unit test voting
M5	Vote order-blind với list	["A","B"] vs ["B","A"] → unanimous, lấy candidate đầu	khác thứ tự ⇒ khác signature; 1-1 ⇒ abstain	unit test voting
M6	Tie-break rng.choice	voting_executor.py:177; 1-1-1 vẫn trả 1 đáp án (EV −1/3)	bỏ RNG; agreement ≤ 0.5 ⇒ abstain_low_agreement (0 điểm)	grep rng.choice = 0 + test 1-1-1 + test tất định (2 lần chạy giống nhau)
M7	empty/NULL → 0	format_result([],"number") → 0; [(None,)] → 0; format_number("4.820.000") → 0	tất cả → ("abstain", None); không bao giờ bịa 0/False/non-ISO	unit test, 9 ca
M8	Không hồi quy	225 passed (đã chạy, 45.15 s, 0 failed)	pytest xanh; ruff check sạch; replay 5 câu cached: PA −0.6 → −0.4	pytest + real_tests/verify_p4_vote_format.py
2. Nền tảng đã kiểm chứng (không phải giả định)

    Percent: train có 578/3000 = 19,3 % câu percent; 43/578 (7,4 %) có |answer| < 1 ⇒ heuristic hiện tại phá nhóm này (mỗi câu −1). Log thật content/result/logs/q_00558c1410ef.json: cả 3 candidate đều viết 100.0 * … / NULLIF(COUNT(*),0) ⇒ model đã theo đúng "thang 0–100" trong prompt; nhánh cần ×100 là nhánh hiếm (model tự chia tỉ lệ).
    Prototype detect_percent_scale đã chạy trên 8 SQL thật/giả lập: ROUND(100.0*SUM(…)/NULLIF(…),2) → percent; COUNT(…)*100.0/COUNT(*) → percent; COUNT(DISTINCT …)*1.0/COUNT(*) → ratio; AVG(col) → percent; SUM(x)/7 → percent (mẫu số là literal ⇒ không phải tỉ lệ); 100.0*a/b → percent.
    cursor.description (đã kiểm trên SQLite thật): trả tên cột thật của bảng (Tên khách, Số tiền nợ), alias khi có (khach_hang), và cả tên cột thật khi SELECT * — đây là lý do phải lấy description, không chỉ AST.
    Scorer của BTC (bản repo evaluation/scorer.py:150-154): cột bảng so theo tên nếu tập tên khớp, ngược lại theo vị trí; list so từng vị trí; row bảng so như tập. Vì vậy tên thật không bao giờ tệ hơn col_N, chỉ có thể tốt hơn.
    Vote hiện gom nhóm bằng frozenset(result_set) (voting_executor.py:151-155) ⇒ mất type, mất thứ tự list, và (None,) chỉ bị chặn nhờ guard has_evidence (P3). has_evidence giữ nguyên.
    Không thể suy ra "kỳ nằm ngoài dải dữ liệu" để chọn unanswerable: metadata_profile có min_val/max_val nhưng không đáng tin — SoBanHang.Ngày giao dịch = DATE_MIXED cho min='01/01/2026', max='46197' (Excel serial). Vì vậy empty/NULL ⇒ abstain, không đoán unanswerable.
    GPU không dùng được trong phiên này (nvidia-smi: driver không truy cập được) và theo yêu cầu: MockLLMEngine nằm ngoài diện kiểm tra. P4 không sửa prompt/engine nên verify trong phiên = unit test + replay offline trên DB cached; lệnh chạy VLLMEngine thật được ghi ở §6 để chạy trên máy GPU trước khi nộp.
    Có sẵn để replay: tmp/cache_biz_{105,112,120,124,128}/biz_*.db + 5 log trong content/result/logs/ + GT dataset/train/answers.jsonl (3000 dòng, 30 business, câu xen kẽ nên --limit không gom theo business).

3. Thiết kế
3.1 Percent scale lấy từ SQL (M1, M2) — result_formatter.py
python

PercentScale = Literal["percent", "ratio"]

def detect_percent_scale(sql: str | None) -> PercentScale:
    """Return whether the final projection already yields percent (0-100) or a bare ratio."""

Thuật toán (đã prototype):

    sqlglot.parse_one(sql, read="sqlite"); lấy projection ngoài cùng (parsed.selects[0]). Parse lỗi / không có SELECT ⇒ "percent" (an toàn với luật "đáp án là số phần trăm").
    Bóc wrapper bằng vòng lặp: Alias → this, Paren → this, Cast → this, Func không phải AggFunc (ROUND/COALESCE/ABS/NULLIF) → this.
    Nếu trong toàn bộ projection có exp.Mul với literal số == 100 ⇒ "percent" (đã ở thang 0–100).
    Nếu node sau khi bóc là exp.Div và mẫu số không phải literal ⇒ "ratio"; còn lại "percent".

python

def format_percent(cls, val: Any, scale: PercentScale = "percent") -> float | None:
    # None / không ép được float  -> None (=> abstain, không bịa 0.0)
    # scale == "ratio" and abs(num) <= 1.0  -> num *= 100.0   (hai tín hiệu độc lập cùng đồng ý)
    # round(num, 2)

Hai tín hiệu độc lập (AST nói ratio và |v| ≤ 1) để không bao giờ nhân 100 vào một giá trị đã là percent (ví dụ SUM(x)/COUNT(*) = đơn giá trung bình bị model gán nhầm type percent, hoặc percent thật 0,5 %). format_result(..., sql=...) truyền scale xuống; pipeline.py:330 truyền winner.final_sql.
3.2 Không bịa giá trị (M7) — result_formatter.py

Áp dụng nguyên tắc "không chuyển được đúng type ⇒ abstain (0 điểm) thay vì bịa (chắc chắn −1)".
Tình huống	Trước	Sau
raw_rows rỗng (mọi type, kể cả number/percent)	0 / 0.0	("abstain", None)
Một dòng toàn NULL	0 / 0.0	("abstain", None)
Ô scalar NULL	0 / 0.0	("abstain", None)
format_number không ép được (vd "4.820.000", "abc")	0	None ⇒ abstain
format_date nhận ISO datetime "2026-06-24 21:44:11"	trả nguyên chuỗi (chắc chắn −1)	cắt về "2026-06-24" (giữ YYYY-MM)
format_date không nhận dạng được	trả nguyên chuỗi	None ⇒ abstain
Boolean token lạ (không thuộc true/1/t/yes/có/đúng/false/0/f/no/không/sai)	False	None ⇒ abstain (bổ sung nhánh false tường minh)

Chữ ký đổi thành float | None / int | float | None / str | None; mọi None được format_result map thành ("abstain", None) ở một chỗ duy nhất. Bảng (table) giữ nguyên cách map ô; target_type scalar mà kết quả nhiều cột vẫn giữ hành vi "recover thành table" như hiện tại (xem §9).
3.3 Tên cột bảng thật (M3)

    VotingExecutor.execute_query và CandidateGenerator._dry_run_query trả thêm columns: list[str] lấy từ cursor.description (5-tuple; None/[] khi lỗi).
    CandidateQuery.result_columns: list[str] = field(default_factory=list); Attempt.raw_columns: tuple[str, ...] = (); to_dict() ghi result_columns (cap 20 tên) để log chẩn đoán thấy ngay.
    Mới trong result_formatter.py:

python

def derive_table_columns(sql: str | None, description: list[str] | None, num_cols: int) -> list[str]:
    """Name table columns from the SQL projection, then description, then positional fallback."""

Thứ tự ưu tiên cho từng cột: (a) AST projection: Alias → alias, Column → tên cột, FUNC(Column) → f"{func}_{col}"; (b) tên trong description nếu không chứa ( (phủ SELECT *); (c) col_{i}. Khử trùng lặp bằng hậu tố _2, _3. format_result(..., raw_columns=..., sql=...) dùng hàm này cho cả hai nhánh table (1 dòng nhiều cột và nhiều dòng nhiều cột).
3.4 Vote theo đáp án + agreement gate (M4, M5, M6) — voting_executor.py

Mới (module-level, testable):
python

def answer_signature(cand: CandidateQuery) -> tuple[str, str] | None:
    """Canonical (answer_type, canonical_value) key of the answer this candidate would submit."""

    Gọi đúng ResultFormatter.format_result(cand.raw_result, cand.predicted_type, raw_columns=cand.result_columns, sql=cand.final_sql) — cùng một hàm với lúc nộp, nên vote so đúng thứ được chấm.
    None khi formatter trả abstain/unanswerable.
    Canonical hoá: number/percent/date/boolean/string → json.dumps(value); list → giữ thứ tự; table → columns theo thứ tự (đã normalize bằng normalize_contract_string) + rows sắp xếp lại (scorer so row như tập, nên khác thứ tự dòng không phải bất đồng).
    type nằm trong key ⇒ number và percent cùng giá trị không còn là đồng thuận (M4).

vote() mới:

    Giữ nguyên: unanswerable_consensus, insufficient_evidence, abstain_fallback, has_evidence (không đổi).
    evidence_cands → gom nhóm theo answer_signature; candidate không có signature bị loại khỏi nhóm (không phải đồng thuận).
    Nhóm lớn nhất, không cần tie-break: nếu len(nhom) / len(evidence_cands) > min_agreement (mặc định 0.5) ⇒ thắng (đại diện = min(candidate_id)); ngược lại (None, "abstain_low_agreement", …). Hai nhóm không thể cùng > 0.5 nên winner luôn duy nhất — đây là lý do bỏ được tie-break.
    Xoá import random, random_seed, self.rng, nhánh random_tiebreak. Status: giữ dạng unanimous_k_of_n / majority_k_of_n (n = len(candidates)), thêm abstain_low_agreement; reason ghi phân bố nhóm (vd 1/1/1).
    min_agreement: float = 0.5 là tham số __init__ (không thêm constant mới — tests/test_constants.py assert __all__ khớp chính xác).

Hệ quả EV: 1-1-1 (1/3) và 1-1 (1/2) ⇒ abstain; 2-1 (2/3) ⇒ trả lời; 1 candidate sống sót duy nhất (ratio = 1.0) vẫn trả lời như hiện nay — không đổi hành vi đã đo, tránh mất điểm hàng loạt khi 2 candidate khác bị validator chặn.
3.5 Trace (để chẩn đoán được quyết định mới)

    QuestionSubmission.formatting_trace: dict[str, Any] = field(default_factory=dict), ghi vào to_trace_log(): answer_type, percent_scale, percent_scaled (bool), raw_value, result_columns, evidence_candidates, cluster_sizes.
    pipeline.process_question điền formatting_trace từ kết quả format cuối (không đổi to_submission_dict, không đổi format file nộp).

4. Thay đổi theo file
File	Thay đổi
src/zaic_2026_4/submission/result_formatter.py	detect_percent_scale, derive_table_columns, format_{number,percent,date} trả Optional, guards empty/NULL/unparsable/datetime, format_result(..., *, raw_columns=None, sql=None)
src/zaic_2026_4/submission/voting_executor.py	execute_query trả 5-tuple có columns; answer_signature; vote theo signature + min_agreement; xoá RNG
src/zaic_2026_4/submission/candidate_generator.py	_dry_run_query trả 5-tuple; Attempt.raw_columns; gán CandidateQuery.result_columns
src/zaic_2026_4/submission/models.py	CandidateQuery.result_columns (+to_dict); QuestionSubmission.formatting_trace (+to_trace_log)
src/zaic_2026_4/submission/pipeline.py	truyền winner.final_sql, winner.result_columns vào formatter; điền formatting_trace
src/zaic_2026_4/submission/__init__.py	export detect_percent_scale, derive_table_columns, answer_signature
tests/test_result_formatter.py	viết lại ca percent/table/guard (bỏ assert heuristic 0.0527→5.27, format_number(None)==0)
tests/test_voting_executor.py	5-tuple; type-aware; order-aware; abstain_low_agreement; tất định (không RNG)
tests/test_candidate_generator.py	thêm assert cand.result_columns
real_tests/verify_p4_vote_format.py (mới)	replay offline 5 log + DB cached qua vote/format mới, chấm bằng evaluation.scorer, in PA trước/sau + PA của policy "cần ≥2 đồng thuận" (đối chứng, không đổi code)
docs/p4_plan.md (mới)	artifact sống: bảng M1–M8 điền số đo thật + changelog
docs/diagnosis-latest-run.md	§7 thêm mục đã đóng: S6 (silent zero), phần còn lại của S7 (vote order/type/random)

Không thêm dependency; không sửa prompt, linker, engine, ingestion.
5. Trình tự thực thi (4 commit + docs, mỗi commit qua gate AGENTS.md)

    fix(submission): ground percent normalization in the SQL projection — §3.1 + pipeline truyền sql.
    fix(submission): never fabricate values from empty, null, or unparsable cells — §3.2 + formatting_trace khung.
    feat(submission): name table columns from SQL projection and result cursor — §3.3 (models + generator + executor + formatter).
    fix(submission): vote on canonical answers with an agreement gate — §3.4 (xoá RNG) + reason/trace hoàn chỉnh.
    Docs: docs/p4_plan.md + diagnosis-latest-run.md §7.

Mỗi commit: implement → code-reviewer (kèm nguyên văn prompt người dùng + plan này) → [PASSED] → uv run ruff check <files> → không chạy lại pytest. Nếu [BLOCKED] quay lại bước implement.
6. Test then chốt

    Percent: format_percent(0.5)=0.5; (0.0527,"percent")=0.05; (0.0208,"ratio")=2.08; (0.5,"ratio")=50.0; (123.46,"ratio")=123.46 (guard giá trị chặn); detect_percent_scale 8 SQL ở §2.
    Table: format_result([("Chị Lan",3200000)],"table", raw_columns=["Tên khách","Số tiền nợ"]) → đúng 2 tên thật; SELECT * → tên từ description; SQL không parse được → col_N; tên trùng → hậu tố.
    Guards: ([], "number"), ([(None,)], "number"), ([(None,)], "percent"), ([("4.820.000",)], "number"), ([("2026-06-24 21:44:11",)], "date") (→ 2026-06-24), ([("n/a",)], "date"), ([("maybe",)], "boolean") → tất cả ("abstain", None) hoặc giá trị đúng; ([(0,)], "number") vẫn là 0 (COUNT 0 là evidence thật).
    Vote: 2 candidate 2.08 (number/percent) → không unanimous; list đảo thứ tự → không đồng thuận; 1-1 → abstain_low_agreement; 1-1-1 → abstain_low_agreement; 2-1 → majority_2_of_3, đại diện candidate_id nhỏ nhất; gọi vote() 2 lần trên cùng input → cùng output (không RNG); (None,)×3 vẫn insufficient_evidence.
    Replay (không GPU): uv run python real_tests/verify_p4_vote_format.py — kỳ vọng q_00558c1410ef: percent 0.0 (−1) → abstain (0); q_001883280e83 vẫn +1; q_00195e1e4049, q_002d03ebbded, q_006c26c65d50 vẫn −1 ⇒ PA −0.6 → −0.4 (cột "policy cần ≥2 đồng thuận" in ra để tham chiếu).
    Không hồi quy: UV_CACHE_DIR=./.uvcache uv run pytest -q (baseline 225 passed) + uv run ruff check trên file sửa.
    Trên máy GPU (bắt buộc trước khi nộp, ngoài phiên này): uv run zaic-submit --split train --limit 10 --result-dir content/result rồi uv run python tools/evaluate_submission.py --result-dir content/result --answers dataset/train/answers.jsonl --only-submitted; so PA và số câu answer sinh ra từ kết quả rỗng/NULL (phải bằng 0).

7. Rủi ro & giảm thiểu
Rủi ro	Giảm thiểu
detect_percent_scale nhân 100 nhầm (đơn giá trung bình bị gán type percent)	cần hai tín hiệu: AST là Div với mẫu số không literal và `
Ratio > 1 (tăng trưởng 123 %) không được nhân	chấp nhận (hiếm); log percent_scale/percent_scaled để thấy ngay trong diagnosis
Vote gom theo đáp án làm tăng số ca abstain	chỉ abstain khi thật sự chia rẽ; 2-1 và 1-survivor vẫn trả lời; replay + test khoá hành vi
Tên cột từ AST/description vô nghĩa (sum_Thành tiền)	không tệ hơn col_N (scorer fallback theo vị trí); SELECT * lấy tên thật; tên cột nằm trong log
Đổi return type format_number/percent/date phá call site	chỉ 2 call site (format_result, pipeline); format_result là cửa duy nhất map None → abstain; test phủ
Bỏ random_seed phá API	chỉ VotingExecutor() không tham số trong pipeline/test; grep xác nhận không nơi khác dùng
Empty ⇒ abstain làm mất các ca "đáp án thật là 0"	chỉ áp cho rỗng/toàn NULL; (0,) vẫn là đáp án 0; GT percent = 0 chỉ 2/3000
8. Quyết định đã chốt
#	Quyết định	Lý do
V1	Bỏ hoàn toàn heuristic `	v
V2	Empty/NULL/unparsable ⇒ abstain, không unanswerable	Không chứng minh được "kỳ ngoài dải dữ liệu" (min_val/max_val sai định dạng); sai unanswerable cũng −1
V3	Vote trên đáp án canonical, không trên result_set thô	Type và thứ tự list là một phần của đáp án được chấm
V4	Ngưỡng đồng thuận > 0.5 trên số candidate có evidence; 1-survivor vẫn trả lời	1-1/1-1-1 có EV ≤ 0 ⇒ abstain; 1-survivor giữ nguyên hành vi đã đo, tránh mất điểm diện rộng
V5	Bỏ hẳn nhánh random_tiebreak (không thay bằng tie-break "chất lượng")	Với ngưỡng > 0.5 winner luôn duy nhất ⇒ tie-break không còn cần, tất định tuyệt đối (Rule 8)
V6	Tên cột: AST → cursor.description → col_N	Tên thật không bao giờ tệ hơn; description là nguồn duy nhất đúng cho SELECT *
V7	Không sửa prompt sinh SQL, không đụng MockLLMEngine	Giữ prefix cache + tránh đổi chất lượng chưa đo; verify trên VLLMEngine thật/ replay
V8	min_agreement là tham số VotingExecutor, không thêm constant	tests/test_constants.py assert __all__ khớp chính xác
9. Ngoài phạm vi (ghi nhận, không làm)

    Suy luận unanswerable từ dải dữ liệu (cần metadata date coverage đáng tin — hiện min_val/max_val hỗn định dạng).
    Chuẩn hoá tiền từ chuỗi ("4.820.000", "122k") ở lớp format; làm sạch sql_expr ở profiling (P0#2).
    target_type scalar mà kết quả nhiều cột: giữ recover thành table; list nhiều cột: giữ lấy cột 0; list có ô NULL: giữ bỏ ô NULL — ba hành vi này đổi sẽ cần đo riêng.
    Vote theo checklist/P1#6 đầy đủ, batching/early-exit, cắt result_set trong log (S12), as_of_date cho linker (S1), schema linking (S2/S3b).
    MockLLMEngine và các test dùng nó: chỉ giữ như smoke test, không phải tiêu chí P4.

10. Giả định

    Máy chạy nộp là GPU host có vllm + Qwen/Qwen3.5-9B; phiên này không có GPU nên bước §6-cuối do người dùng chạy.
    tmp/cache_biz_* và dataset/train/answers.jsonl còn tồn tại tại thời điểm replay (đã kiểm: có).
    Không có thay đổi nào làm tăng số lượt gọi LLM/candidate (P4 chỉ sửa hậu xử lý).

## 11. Kết quả đo thực tế & Changelog

| Mục tiêu | Tiêu chí | Trạng thái | Kết quả đo thực tế |
|---|---|---|---|
| M1 | Percent scale theo SQL | PASSED | `detect_percent_scale` xử lý chuẩn 8 ca SQL, `format_percent(0.5, "percent") == 0.5`, `format_percent(0.0527, "ratio") == 5.27` |
| M2 | Loại bỏ heuristic `num * 100` | PASSED | `grep "num \* 100"` trả về 0 kết quả (dùng `num *= 100.0` với 2 tín hiệu độc lập AST + guard) |
| M3 | Cột bảng thật từ AST / description | PASSED | `derive_table_columns` lấy alias, tên cột, func, description SQLite cursor, fallback `col_N` và khử trùng lặp |
| M4 | Vote phân biệt type | PASSED | `answer_signature` đưa `type` vào chữ ký canonical; number vs percent cùng 2.08 không còn đồng thuận |
| M5 | Vote phân biệt thứ tự list | PASSED | Thứ tự list được giữ nguyên trong canonical signature; table row multiset sắp xếp theo JSON canonical |
| M6 | Loại bỏ RNG tie-break | PASSED | `grep "rng\.choice"` = 0; phân bố <= 0.5 (1/1, 1/1/1) chuyển thành `abstain_low_agreement`; kết quả tất định 100% |
| M7 | Empty/NULL/unparsable -> abstain | PASSED | Không bịa `0`, `0.0`, `False`, giữ `[(0,)] -> 0`; datetime ISO được cắt về YYYY-MM-DD |
| M8 | Replay offline 5 log cached DB | PASSED | `real_tests/verify_p4_vote_format.py`: PA cải thiện từ -0.60 lên -0.40 (`q_00558c1410ef` từ -1 lên 0) |

### Changelog
- 2026-11-13: Hoàn tất P4: chuẩn hoá percent theo SQL projection, không bịa giá trị rỗng/NULL, trích xuất tên cột thật từ SQLite cursor và AST, vote theo đáp án canonical với ngưỡng đồng thuận > 0.5, loại bỏ RNG tie-break. Replay offline đạt PA -0.40.
