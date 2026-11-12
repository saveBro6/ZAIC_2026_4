Validator + P2 + P3 — từ "kiểm tra hình thức" sang "chẩn đoán có mã + tự sửa hội tụ"
1. Mục tiêu

Ba lỗ hổng đang nối tiếp nhau: validator bắt sai (3 check false-positive) và bắt thiếu (không kiểm tồn tại cột, không kiểm định dạng ngày, không kiểm shape theo type); chẩn đoán là prose rời rạc nên retry dao động; vòng lặp retry không có bộ nhớ, không giữ candidate tốt nhất, và vote coi rỗng/NULL là "đồng thuận".

Kết thúc đợt này: mỗi lỗi có mã ổn định + gợi ý hành động, retry hội tụ hoặc dừng sớm, candidate tốt nhất không bị mất, và (None,)/rỗng không còn thắng phiếu.
#	Tiêu chí	Đo bằng
M1	0 blocking-issue trên corpus 12 SQL hợp lệ (gồm correlated latest-per-group, `substr	
M2	SQL đã đúng ⇒ 0 lượt gọi LLM sửa lỗi (warning không kích hoạt repair)	test + log counter
M3	Model lặp lại y nguyên ⇒ dừng sau ≤2 lượt (hiện 3) và giữ được kết quả đã chạy	stub engine test
M4	(None,)×3 hoặc rỗng ⇒ voting_status="insufficient_evidence" (hiện unanimous_3_of_3 → percent 0.0 = −1)	test voting
M5	Bắt được 4 mẫu hỏng thật từ docs/diagnosis-latest-run.md: T2."Mã HH", H."Ngày đặt", calendar."Status", ISO literal trên cột dd/mm/yyyy	test với metadata fixture
M6	Không hồi quy: full suite xanh; số lượt gọi LLM/candidate trên mock suite không tăng; validator ≤ 5 ms/SQL	pytest + timing

Ngoài phạm vi (ghi nhận, không làm): linker truyền as_of_date, sinh sql_expr ở profiling, abstain-gate theo EV, bỏ silent-zero trong result_formatter, vote theo checklist, batching/early-exit, log truncation. Đây là diagnosis P0#1–5 / P1#6,8,9 ≥ lớp trên.
2. Nền tảng đã kiểm chứng

    P0/P1 đã ở commit 63611a5. RepairContext + prefix dùng chung sẵn sàng nhận chẩn đoán có cấu trúc.
    sqlglot 30.29.0: sqlglot.optimizer.scope.traverse_scope cho scope.sources (Table vs Scope cho CTE/derived) và scope.columns — đủ để kiểm tồn tại cột theo scope, không false-positive với CTE/derived.
    parse_one(sql, read="sqlite").sql(dialect="sqlite", normalize=True) chuẩn hoá hoa/thường identifier nhưng giữ literal → dùng để phát hiện "model lặp lại y nguyên".
    Metadata sẵn có để kiểm: data_format, date_subtype, currency_subtype, null_count, distinct_count, top_10_samples, raw_samples (khoá phẳng table.col).
    Phải dùng processed_metadata đầy đủ (không phải reduced linked schema) khi kiểm tồn tại cột, nếu không sẽ báo sai.

3. Thiết kế
3.1 rules.py — một nguồn sự thật cho mã lỗi (mới)
python

Severity = Literal["blocking", "warning"]

@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: Severity = "blocking"
    column: str | None = None
    hint: str | None = None

    def to_prompt_line(self) -> str:
        """Render the issue as a single actionable diagnostic line."""

RULE_REGISTRY: dict[str, str]   # code -> prompt hint (VN), single source of truth
def render_diagnostics(issues: list[ValidationIssue]) -> list[str]

Bộ mã: OUTPUT_EMPTY, OUTPUT_UNPARSEABLE, TYPE_INVALID, SQL_SYNTAX, SQL_NOT_READ_ONLY, SQL_NOT_SELECT, SQL_MULTIPLE_STATEMENTS, TABLE_NOT_FOUND, COLUMN_NOT_FOUND, UNKNOWN_QUALIFIER, DATE_FORMAT_MISMATCH, LITERAL_NOT_IN_ENUM, SHAPE_MISMATCH, SHAPE_SUSPECT_NO_AGGREGATE, RANK_TIEBREAK_MISSING, MONEY_SCALE_RISK, ORDER_ASC_NULLABLE, MIN_NULLABLE, RESULT_SHAPE_MISMATCH, UNRESOLVED_ZERO, STAGNATION, DELIBERATE_ABSTAIN, EXECUTION_ERROR.

Chỉ registry hoá chẩn đoán, không render lại toàn bộ prompt sinh SQL (giữ byte prompt hiện tại để không phá prefix cache). Test bắt buộc: mọi code validator có thể phát đều có entry trong registry.
3.2 Validator: bỏ 3 check sai, thêm 5 check đúng

Bỏ (demote xuống warning, không block)
Check hiện tại	Lý do	Thay bằng
ORDER BY f ASC bắt buộc f IS NOT NULL	false positive; SQL đúng vẫn bị loại khỏi dry-run	warning ORDER_ASC_NULLABLE, chỉ khi metadata báo null_count > 0
MIN(f) bắt buộc f IS NOT NULL	như trên	warning MIN_NULLABLE, cùng điều kiện
Cấm subquery lồng MIN/MAX	"latest per group" hợp lệ, không thể thay bằng ORDER BY LIMIT 1	xoá
Cấm ||/CONCAT trong SELECT	date type bắt buộc YYYY-MM phải dùng substr||'-'||	warning CONCAT_IN_SELECT, bỏ qua khi target_type == "date"

Thêm (blocking trừ khi ghi rõ)

    TABLE_NOT_FOUND / COLUMN_NOT_FOUND / UNKNOWN_QUALIFIER — duyệt traverse_scope; qualifier trỏ tới Table ⇒ kiểm cột theo metadata, gợi ý tối đa 5 cột gần nhất (difflib.get_close_matches trên tên đã normalize); qualifier trỏ tới Scope (CTE/derived) ⇒ bỏ qua; qualifier không tồn tại ⇒ UNKNOWN_QUALIFIER blocking. Cột không qualifier: chỉ kiểm khi scope có đúng 1 bảng gốc hoặc cột không tồn tại ở bất kỳ bảng nào trong scope.
    DATE_FORMAT_MISMATCH (blocking) — chỉ xét so sánh trực tiếp Column ↔ string Literal (=, !=, <, <=, >, >=, BETWEEN, IN) khi cột được metadata gắn data_format/date_subtype dạng ngày. Lệch kiểu (ISO ↔ DD/MM/YYYY, Excel serial) ⇒ block kèm hint chuyển đổi generic (SUBSTR/strftime), vì so lệch kiểu trả 0 dòng âm thầm (Q4) hoặc lexicographic sai (Q1).
    LITERAL_NOT_IN_ENUM (warning) — khi distinct_count <= 10 và tập giá trị biết đủ (top_10_samples ∪ raw_samples phủ hết distinct): literal không khớp sau normalize ⇒ warning + 3 giá trị gần nhất. Bắt trực tiếp ca 'Huỷ' vs 'hủy đơn' (Q3).
    SHAPE_MISMATCH (blocking) + SHAPE_SUSPECT_NO_AGGREGATE / RANK_TIEBREAK_MISSING (warning) theo target_type: number/percent/date/boolean ⇒ đúng 1 projection; list ⇒ 1 projection; table ⇒ ≥2. Aggregate/LIMIT thiếu ⇒ warning "7 dòng cho câu number". list thiếu ORDER BY hoặc thiếu tie-break theo khoá/mã ⇒ warning.
    MONEY_SCALE_RISK (warning) — CURRENCY có currency_subtype báo phần thập phân nhưng SQL REPLACE bỏ cả . và , ⇒ warning (S8/Q5 lần trước ×10⁴).

Chữ ký mới: validate(raw_sql, target_type: str | None = None, metadata: dict[str, Any] | None = None) -> ValidationResult — giữ mặc định để call site cũ vẫn chạy; metadata=None ⇒ bỏ các check phụ thuộc metadata (không block vì thiếu dữ kiện).

Chính sách severity (quyết định): is_valid = không có issue blocking. Warning không bao giờ tự kích hoạt repair — chỉ được đính vào prompt khi một repair đã xảy ra vì lý do khác. Đây là điều kiện để đạt M2.
3.3 P2 — chẩn đoán có mã trong prompt sửa lỗi

ValidationResult.issues: list[ValidationIssue] (+ __str__ trả message để tương thích "; ".join(...); property issue_texts, codes()). parse_structured_sql_output phát OUTPUT_EMPTY / OUTPUT_UNPARSEABLE / TYPE_INVALID.

RepairContext.diagnostics đổi sang list[ValidationIssue]; _format_repair_task render mỗi issue một dòng:
Code block

- [COLUMN_NOT_FOUND] Cột "Mã HH" không tồn tại trong bảng "chi_tiet". Cột hợp lệ: "Mã hàng", "Mã HĐ". | Gợi ý: dùng đúng tên cột trong <selected_schema>.

Không còn prose tự do ⇒ model sửa được đích danh, và test assert theo code.
3.4 P3 — state machine tự sửa hội tụ

Thay hai vòng while hiện tại bằng một vòng duy nhất trên trạng thái có rank:
python

@dataclass(frozen=True)
class Attempt:
    sql: str
    target_type: str
    issues: list[ValidationIssue]
    executed: bool = False
    shape_ok: bool = False
    rows: int | None = None
    raw_output: str = ""

def rank(a: Attempt) -> tuple:            # so sánh "tốt hơn"
    return (a.executed, a.shape_ok, not has_blocking(a.issues), a.rows or 0)

Thuật toán:

    Sinh SQL lần 1 (temperature giữ nguyên), đánh giá: static (blocking/warning) → dry-run (nếu không blocking) → shape check động (RESULT_SHAPE_MISMATCH).
    best = max(best, current, key=rank) sau mỗi lượt — kể cả lượt có bất thường; đây là fix cho mất candidate tốt.
    Dừng ngay khi: sạch (không blocking, executed, shape_ok, có evidence) | unanswerable/abstain có chủ đích | hết ngân sách.
    Dừng sớm khi đình trệ: frozenset(blocking_codes) lặp lại, hoặc canonical(new_sql) == canonical(best.sql) (model lặp lại y nguyên). Ghi STAGNATION.
    Zero-result: chỉ retry khi có bằng chứng (DATE_FORMAT_MISMATCH, LITERAL_NOT_IN_ENUM, UNKNOWN_QUALIFIER, hoặc literal LSH không khớp); không có bằng chứng ⇒ dừng, đánh dấu unresolved_zero=True (thay _is_suspicious_zero_result hiện tại đang coi 0/rỗng là đáng ngờ vô điều kiện).
    Prompt sửa lỗi nhận: accepted_sql = best.sql, last_output = raw_output của lượt gần nhất, diagnostics = blocking + warning của lượt đó, và 2 attempt gần nhất (SQL canonical + mã lỗi) để model không lặp.
    CandidateQuery thêm deliberate_abstain: bool, unresolved_zero: bool, repair_stop_reason: str | None (đưa vào to_dict).

3.5 Voting — hai guard tối thiểu
Guard	Trước	Sau
has_evidence(result_set)	không có	false nếu None/rỗng/mọi ô None; loại khỏi nhóm đồng thuận; toàn bộ không evidence ⇒ (None, "insufficient_evidence")
Abstain có chủ đích	execute_candidates chạy SQL rỗng ⇒ thành candidate rỗng tham gia vote	không thực thi; nếu chỉ có abstain/unanswerable ⇒ trả candidate đó, status unanswerable_consensus để pipeline giữ type unanswerable
Nhãn status	hardcode 2_of_3/3_of_3	f"majority_{len(g)}_of_{len(candidates)}"

Kết quả kỳ vọng: Q4 kiểu (None,)×3 → pipeline abstain (0 điểm) thay vì −1; Q5 number 0 từ SUM rỗng → abstain.
4. Thay đổi theo file
File	Thay đổi
src/zaic_2026_4/submission/rules.py (mới)	Severity, ValidationIssue, RULE_REGISTRY, render_diagnostics
src/zaic_2026_4/submission/models.py	ValidationResult.issues → list[ValidationIssue] + issue_texts/codes; CandidateQuery thêm 3 field
src/zaic_2026_4/submission/sql_validator.py	bỏ/xoá 4 check, thêm 5 check (schema/date/enum/shape/money), chữ ký validate(..., target_type, metadata), parse_structured_sql_output phát ValidationIssue
src/zaic_2026_4/submission/candidate_generator.py	truyền metadata+target_type; state machine thay 2 vòng lặp; evidence-gated zero; best retention; stagnation/canonical stop; attempt history
src/zaic_2026_4/submission/prompt_template.py	RepairContext.diagnostics: list[ValidationIssue]; render [CODE]; thêm <previous_attempts> (tối đa 2)
src/zaic_2026_4/submission/voting_executor.py	has_evidence, loại empty/all-NULL, deliberate abstain, nhãn status động
src/zaic_2026_4/submission/__init__.py	export ValidationIssue, Severity
tests/test_rules_registry.py (mới)	registry phủ đủ code; format dòng chẩn đoán
tests/test_repair_state_machine.py (mới)	6 kịch bản dừng/hội tụ/best-retention
tests/test_sql_validator.py	viết lại: bỏ test 4 check cũ; thêm schema/date/enum/shape + corpus 12 SQL hợp lệ (M1)
tests/test_candidate_generator.py	parse_issues theo .code; kỳ vọng retry mới (M2/M3)
tests/test_repair_prompt.py	chẩn đoán [CODE], <previous_attempts>
tests/test_voting_executor.py	M4 + deliberate abstain + nhãn status
docs/diagnosis-latest-run.md	thêm mục "đã đóng" cho S5/S7(phần rỗng)/S9 khi xong
5. Trình tự thực thi (4 commit, mỗi commit qua gate)

    feat(submission): add structured validation issues and rule registry — rules.py, models, parse output, RepairContext/renderer, export. Không đổi hành vi check.
    fix(submission): replace false-positive SQL checks with schema-grounded validation — demote/xoá 4 check, thêm 5 check, candidate_generator truyền metadata/target_type.
    refactor(submission): drive self-correction with a ranked repair state machine — Attempt/rank/best/stagnation/canonical/evidence-gated zero/attempt history.
    fix(submission): treat empty results as insufficient evidence and preserve deliberate abstain — voting guards + nhãn động.

Mỗi commit: implementation → code-reviewer (nguyên văn prompt người dùng + plan này) → ruff check → không chạy lại pytest sau khi [PASSED]. Cuối cùng: cập nhật metrics M1–M6 + changelog vào docs/p0_p1_repair_prompt_plan.md (hoặc tách docs/validator_p2_p3_plan.md nếu anh/chị muốn giữ song song).
6. Test then chốt

M1 corpus 12 SQL phải pass (0 blocking): SELECT SUM(x) FROM t; · ORDER BY x ASC không NOT NULL + metadata null_count>0 (chỉ warning) · correlated WHERE x = (SELECT MAX(x) FROM t2 WHERE t2.g = t.g) · substr(d,7,4)||'-'||substr(d,4,2) với target_type="date" · CTE WITH a AS (...) SELECT ... · LEFT JOIN 3 bảng · top-k có tie-break mã · COUNT(DISTINCT x) · ROUND(AVG(x),0) · NULLIF(...,0) · bảng có cột trùng tên ở 2 bảng nhưng đã qualify · IN (...) trên enum khớp metadata.

M2/M3 kịch bản state machine: (a) SQL đúng ngay ⇒ đúng 1 lượt gọi; (b) model lặp y nguyên ⇒ ≤2 lượt + STAGNATION; (c) sửa được ở lượt 2 ⇒ success, sql_retry_count=1; (d) lượt cuối fail ⇒ giữ best với raw_result đã chạy (hiện bị xoá); (e) unanswerable có chủ đích ⇒ dừng ngay, không repair; (f) zero không bằng chứng ⇒ 1 lượt, unresolved_zero=True.

M5 mẫu thật: metadata fixture mô phỏng chi_tiet/chi_tiet_hoa_don/calendar; assert đúng 4 ca từ diagnosis, mỗi ca đúng code và có ≥1 gợi ý cột.
7. Rủi ro & giảm thiểu
Rủi ro	Giảm thiểu
False positive khi kiểm cột (CTE, derived, *, rowid, JSON path)	dùng traverse_scope; bỏ qua nguồn Scope; corpus M1; UNKNOWN_QUALIFIER chỉ khi không có trong sources lẫn CTE
Metadata thiếu cột do dùng reduced schema	bắt buộc truyền processed_metadata đầy đủ; metadata=None ⇒ tắt check phụ thuộc metadata
DATE_FORMAT_MISMATCH block oan khi cột đã được bọc strftime	chỉ xét so sánh trực tiếp Column ↔ literal, không xét expression có hàm
Enum check false positive do top_10_samples bị cắt	chỉ chạy khi distinct_count <= 10 và tập giá trị phủ hết distinct; chỉ warning
Đổi ValidationResult.issues phá call site/test	__str__ + issue_texts; cập nhật 2 call site + 4 test đã liệt kê
Luật abstain làm mất điểm các câu trả lời đúng hiện tại	chỉ loại tập rỗng/toàn NULL; (0,) từ COUNT vẫn là evidence; kiểm bằng test riêng
Validator thêm chi phí	map normalize dựng 1 lần/candidate; mục tiêu ≤5 ms/SQL, đo trong test
8. Quyết định đã chốt (ghi vào Decisions log)
#	Quyết định	Lý do
V1	Warning không kích hoạt repair, chỉ đính kèm khi repair đã xảy ra	Điều kiện để chấm dứt retry vô ích (M2)
V2	DATE_FORMAT_MISMATCH là blocking (phạm vi hẹp: so sánh trực tiếp)	Lệch định dạng ngày thất bại im lặng — loại lỗi đắt nhất (Q1, Q4)
V3	(None,)/rỗng bị loại khỏi đồng thuận ⇒ abstain	Đổi −1 thành 0 cho đúng nhóm lỗi này (M4)
V4	Xoá hẳn luật cấm subquery MIN/MAX, không giữ dưới dạng warning	Luật sai bản chất, warning vẫn gây nhiễu prompt
V5	Giữ nguyên byte prompt sinh SQL	Bảo toàn prefix cache + kết quả P1
V6	Không làm abstain-gate/checklist-vote ở đợt này	Thuộc lớp scoring, cần đo PA riêng để tránh hồi quy

Nếu anh/chị muốn đảo V3 (giữ hành vi trả 0 như hiện tại), nói rõ trước khi tôi bắt đầu commit 4; ba quyết định còn lại tôi đề nghị giữ nguyên.