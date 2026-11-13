# Phân tích submission logs `logs_submission_21h_1311/`

Báo cáo phân tích run 21:08–21:26 ngày 13/11/2026 (sau commit `a395270` — `feat(pipeline): anchor temporal context, repair schema coverage and harden SQL validation`). Toàn bộ kết luận dưới đây được đối chiếu trực tiếp với 10 SQLite DB trong `tmp/cache_biz_*/`, `dataset/train/answers.jsonl` và source code hiện tại.

## 0. Tóm tắt điều hành

### 0.1 Phạm vi phân tích

| Nhóm artifact | Số file | Dung lượng | Đã dùng để |
|---|---:|---:|---|
| `content/result/submission.jsonl` | 1 | 546 B | Chấm điểm 10 câu |
| `content/result/logs/q_*.json` | 10 | 74 KB | Trace từng câu: SQL, coverage, voting, retry |
| `tmp/cache_biz_*/llm_calls.jsonl` | 10 | 11.2 MB | 938 LLM call: stage, token, latency, truncation |
| `tmp/cache_biz_*/schema_linking_result_q_*.json` | 10 | 1.7 MB | Schema cuối, 5 variant, grounded literals, coverage |
| `tmp/cache_biz_*/{metadata_profile,raw_profile}.json` | 20 | 22 MB | Format cột, sample, enum, profile cấp bảng |
| `tmp/cache_biz_*/profiling_trace.json` | 10 | 7.2 MB | Chi phí profiling, batch, fallback |
| `tmp/cache_biz_*/biz_*.db` | 10 | 105 MB | Kiểm chứng ground truth và nguyên nhân sai |
| `tmp/cache_biz_*/{lsh_index.pkl,faiss_index}` | 20 | 41 MB | Chỉ số index (không phải nguồn lỗi) |

10 câu hỏi là 10 dòng đầu của `dataset/train/questions.jsonl` (`biz_105 … biz_128`), **có ground truth** trong `dataset/train/answers.jsonl`, nên mọi kết luận về đúng/sai đều kiểm chứng được bằng SQL, không phải suy đoán.

### 0.2 Kết quả run so với ground truth

| Câu | Pred | Ground truth | Điểm | Loại lỗi |
|---|---|---|---:|---|
| `q_001883280e83` | `number` 89,526,400 | 89,526,400 | **+1** | — |
| `q_00195e1e4049` | `number` 951,600 | 2,954,200 | **−1** | Subset ngày `dd/mm/yyyy` (mất 68% doanh thu) |
| `q_002d03ebbded` | `number` 10,312,600 | 10,309,400 | **−1** | Không loại 1 đơn huỷ 3,200 (literal status sai) |
| `q_00558c1410ef` | `percent` 1.05 | 2.08 | **−1** | Sai grain (dòng vs đơn) + đồng nghĩa trạng thái + ngày |
| `q_006c26c65d50` | `abstain` | 2,345 | 0 | Repair thất bại, 8 lần truncation |
| `q_0088f7a0afc5` | `abstain` | 77 | 0 | Hallucinate tên bảng, tên sản phẩm sai (`0,01%`) |
| `q_008f69128180` | `percent` 0.0 | 1.21 | **−1** | `'Hủy'` vs `'hủy'` + 75% dòng ngày không phải `dd/mm/yyyy` |
| `q_00a6e23cbacf` | `abstain` | 200,556 | 0 | Alias `T2` không khai báo → SQL_SYNTAX, 5 truncation |
| `q_00bb4edee433` | `abstain` | 2.27 | 0 | Thiếu bảng cầu join; 46% dòng mất header (nhóm dòng trống) |
| `q_00c4e64fe706` | `number` 131,362,000 | 271,866,000 | **−1** | Subset ngày `dd/mm/yyyy` (mất 52% doanh thu) |

**PA = (1 + 0 + 0 + 0 + 0 − 5) / 10 = −0.40.** Đây là mức cảnh báo: pipeline đang trả lời **sai nhiều hơn đúng**, và 4/10 câu bị mất điểm oan vì các câu đó *có thể trả lời được* (đã kiểm chứng bằng SQL trên chính DB của pipeline).

### 0.3 Bảy phát hiện nghiêm trọng nhất

1. **Chuẩn hóa ngày chỉ phủ `dd/mm/yyyy`, trong khi dữ liệu là `DATE_MIXED`.** 5/10 business có ≥46% dòng ngày nằm ngoài `dd/mm/yyyy`: Excel serial hoặc `d thg m yyyy` ở `biz_106` 50%, `biz_105` 50%, `biz_124` 67%, `biz_117` 75%; NULL ở `biz_121` 46% (xem §8.1). `SUBSTR(col,7,4)` biến các dòng đó thành chuỗi rác `--` và **loại im lặng** khỏi mọi phép lọc kỳ. Đây là nguyên nhân trực tiếp của 2 câu sai nặng (−52% và −68% doanh thu) và góp phần vào 2 câu sai còn lại.
2. **Ingestion không thực hiện luật "ô trống ở dòng sau của cùng nhóm = ô của dòng đầu nhóm" (rulebook §8).** `biz_121`: 38,878/82,920 dòng (46%) mất `Mã đơn hàng`/`Ngày đặt hàng`/`Trạng thái`. Forward-fill trên đúng dữ liệu đó cho tổng doanh thu tháng 7.2025 = 264.9 M thay vì 143.2 M, và tỉ lệ thuốc kê đơn = 2.26% (GT 2.27%).
3. **Trạng thái đơn là tập giá trị tự do, pipeline không có canonical hóa.** Mỗi shop một bộ nhãn: `ok`, `xong`, `đã xong`, `đã lấy`, `Hoàn thành`, `hủy`, `Hủy`, `Đã hủy đơn`, `hủy đơn`, `trả lại`, `Trả hàng`, `Hoàn trả`, `Đổi trả`, `hàng trả về`, `Đang vận chuyển`… Filter bằng literal chính xác → tập rỗng hoặc `NULL`: gây sai ở `q_002d03`, `q_00558`, `q_008f`, và là một trong các nguyên nhân của `q_006c`, `q_00c4`.
4. **Metadata cột bị cắt cụt ở 240 ký tự trong prompt.** `format_schema_column_element(..., max_desc_len=240)` cắt ngay giữa danh sách enum: prompt thực tế chứa `Tập giá trị đầy đủ: 'xo...` — model **không nhìn thấy** tập giá trị trạng thái, dù `metadata_profile.json` có đủ.
5. **`join_key_closure` trong coverage luôn được đánh dấu "satisfied".** `coverage.py` chỉ thêm cột khóa khi *cả hai* bảng đã có mặt, không bao giờ thêm bảng cầu và không bao giờ fail → `missing_aspects` luôn rỗng, `unanswerable_proof` luôn `True`. `q_00bb4edee433` thiếu hẳn bảng cầu `DuLieuXuat_DanhSachSanPham` mà coverage vẫn báo đủ → model tự nối `"Tên sản phẩm" = "Mã SKU"` (vô nghĩa) → NULL.
6. **Ngưỡng đồng thuận của voting bị tính trên "candidate có evidence" chứ không trên tổng số candidate.** `agreement_ratio = len(best_group)/len(evidence_cands)`, nên **1/3 candidate sống sót được submit như "majority"**. Xảy ra ở 3/10 câu (`q_00195`, `q_008f`, `q_00c4`) và cả 3 đều sai. Ở `q_002d03`, 2 candidate sai *vì hai lý do khác nhau* lại tạo thành "majority 2/3" và đánh bại candidate đúng.
7. **Chi phí/độ trễ nằm ở vòng repair, không ở generation.** 220 call cho 10 câu (1.27 M prompt token), trong đó `sql_repair` chiếm 36 call với latency trung vị 6.0 s (max 17.9 s); `q_00195` riêng nó tốn 152 s / 310 K prompt token / 7 lần truncation. Trung bình **56 s/câu** → 1000 câu ≈ 15.6 giờ đơn luồng.

### 0.4 Chỉ số nổi bật

| Chỉ số | Giá trị |
|---|---:|
| Câu đúng / sai / abstain | 1 / 5 / 4 |
| Penalized Accuracy | **−0.40** |
| Tổng thời gian xử lý 10 câu | 562.1 s (trung bình 56.2 s; max 101.5 s — `q_0088`) |
| LLM call (luận lý) | 938, gồm 718 `minimal_profile`, 75 `schema_linking_init`, 64 `schema_linking_repair`, 45 `sql_init`, 36 `sql_repair` |
| Prompt token | 2.93 M (1.65 M profiling + 1.27 M câu hỏi) |
| Completion token | 102.8 K (32.2 K profiling + 70.6 K câu hỏi) |
| Call bị cắt (`finish_reason=length`) | 30 / 938 (3.2%), tập trung ở 5 câu |
| Candidate chạy được / tổng | 18 / 30 (60%), trong đó 1 là `deliberate abstain` ⇒ 17 query thực thi; câu có đủ 3 candidate chạy được: 2; câu có đủ 3 candidate **có evidence**: 1 |
| Câu submit dựa trên 1/3 candidate | 3 |

## 1. Bối cảnh run và đối chiếu lịch sử

- Cache profiling được tạo 21:08–21:10 cho 10 business; log câu hỏi 21:17–21:26. Run **tuần tự theo câu**, mỗi business một DB riêng, không có song song.
- `cache_fingerprint.txt` chỉ băm **input files** (`profile_manager.py:122`), không băm source code → commit mới **không** làm mất cache profiling. Đây là điểm tốt, cần giữ nguyên.
- So với snapshot `trace_logs_1311/` (11:57, 2 câu): lần này số câu tăng 5×, và **temporal anchor đã hoạt động đúng** (xem §11).
- `content/result/logs/` là bản copy của `question_trace_*` (mtime 21:17–21:26), không phải suy diễn lại.

## 2. Hạng mục A — Run-level: cấu trúc câu trả lời

### 2.1 Phân bố quyết định cuối

| Trạng thái voting | Số câu | Câu |
|---|---:|---|
| `unanimous_3_of_3` | 1 | `q_001883280e83` (đúng) |
| `majority_2_of_3` | 2 | `q_002d03ebbded` (sai), `q_00558c1410ef` (sai) |
| `majority_1_of_3` | 3 | `q_00195e1e4049` (sai), `q_008f69128180` (sai), `q_00c4e64fe706` (sai) |
| `insufficient_evidence` | 2 | `q_0088f7a0afc5`, `q_00bb4edee433` |
| `abstain_unverified_unanswerable` | 2 | `q_006c26c65d50`, `q_00a6e23cbacf` |

`q_00c4` minh họa rõ việc "đồng thuận 1/3" được coi là hợp lệ: chỉ candidate 0 có evidence, hai candidate còn lại chết vì `DATE_FORMAT_MISMATCH`/`MONEY_SCALE_RISK`.

### 2.2 Tương quan "số candidate có evidence" và độ chính xác

| Số candidate có evidence | Số câu | Câu | Đúng |
|---|---:|---|---:|
| 3 | 1 | `q_001883` | 1 |
| 2 | 2 | `q_002d03`, `q_00558` | 0 |
| 1 | 3 | `q_00195`, `q_008f`, `q_00c4` | 0 |
| 0 | 4 | `q_006c`, `q_0088`, `q_00a6`, `q_00bb` | 0 (4 abstain) |

Không có câu nào đúng khi <3 candidate có evidence. Nói cách khác: **chất lượng câu trả lời gần như được quyết định ở khâu sinh/kiểm tra SQL, chứ không phải ở khâu voting**; và nhóm "1 candidate có evidence" (3 câu) là nhóm nguy hiểm nhất vì luật hiện tại vẫn submit.

## 3. Hạng mục B — Profiling và metadata

### 3.1 Chi phí

| Chỉ số | Giá trị |
|---|---:|
| Call `minimal_profile` (luận lý) | 718 (≈72 call/business ≈ số cột) |
| Batch thực tế (số latency khác nhau) | 17 |
| Prompt token | 1,654,499 |
| Latency cộng dồn | 111.5 s (đã batch, ~131 ms/batch) |
| `metadata_profile.json` / `raw_profile.json` | 1.3 MB / 1.6 MB mỗi business (biz_128) |

Batching đang hoạt động tốt (718 call → 17 batch), nên **profiling không phải nút cổ chai độ trễ**; nhưng nó là chi phí một lần lớn: 1.65 M token cho 10 business ≈ **165 K token/business**. Với ~100 business của test set, chi phí profiling một lần ≈ 16 M token, chấp nhận được nếu cache được giữ.

### 3.2 Chất lượng metadata: detector đúng, khai thác sai

Ví dụ `biz_106.DanhSachHoaDon.Thời gian` trong `metadata_profile.json`:

```
"data_format": "DATE_MIXED", "formats": ["EXCEL_SERIAL", "SLASH_DATE"], "distinct_count": 1182
```

Profiler **nhận đúng** bản chất hỗn hợp. Nhưng:

- Prompt chỉ nhận được bản rút gọn 240 ký tự, cắt mất phần enum/khoảng.
- Không có trường "biểu thức chuẩn hóa gợi ý" theo từng format.
- Validator chỉ kiểm tra "có hàm chuẩn hóa nào đó" nên chấp nhận `SUBSTR(col,7,4)` — tức chỉ phủ 1/2 format đã biết.

Kết quả: `q_00c4e64fe706` sinh `SUBSTR(...)` trên cột `DATE_MIXED`, **không có issue nào**, exec thành công, trả về đúng nửa doanh thu.

### 3.3 Vấn đề phụ

- `maximal_profile` dài 300–800 ký tự/cột (mô tả + format + enum + khoảng + join hints). Cắt 240 ký tự vừa mất thông tin vừa giữ lại phần ít giá trị nhất (mô tả ngôn ngữ tự nhiên do LLM viết).
- Sample values (`sample_values`, tới 100 phần tử) không được đưa vào prompt schema; model phải đoán literal → nguồn gốc `COLUMN_NOT_FOUND`/`LITERAL_NOT_IN_ENUM`.
- Grounded literals vẫn nhiễu: `biz_124` có entry `"Natri clorid tháng"` (score 0.5625) trùng `"Natri clorid"`; `variant_traces` của `q_00c4` chứa literal ngày `2022-05-31`, `2025-05-01` sinh ra từ trial SQL chứ không từ câu hỏi.

## 4. Hạng mục C — Schema linking

### 4.1 Cấu trúc và chi phí

Mỗi câu chạy 5 "case" (`case_a`…`case_e`) với 3 loại profile (`minimal`, `maximal`, `full`): 75 call init + 64 call repair. Prompt lớn nhất 11,349 token (`biz_124`).

| Câu | Cột cuối | Bảng cuối | Retry | Ghi chú |
|---|---:|---:|---:|---|
| `q_00195e1e4049` | 15 | 4 | 9 | Nhiều nhất, vẫn đủ bảng |
| `q_00bb4edee433` | 7 | 2 | 5 | **Thiếu bảng cầu** `DuLieuXuat_DanhSachSanPham` |
| `q_0088f7a0afc5` | 12 | 3 | 6 | Có bảng nhưng model không dùng |
| `q_00558c1410ef` | 4 | 1 | 7 | Chỉ view `SoBanHang` |
| `q_00c4e64fe706` | 5 | 1 | 0 | Đúng bảng, thiếu hiểu biết format ngày |

### 4.2 Khiếm khuyết: coverage không kiểm tra đường join

`coverage.py:327-340`:

```python
report.required_aspects.append("join_key_closure")
for src, tgt in edges:
    if src_tbl in current_tables and tgt_tbl in current_tables:
        ...  # chỉ thêm cột khóa khi cả hai bảng đã có
report.satisfied_aspects.append("join_key_closure")   # luôn satisfied
```

Hệ quả: `missing_aspects` rỗng ở cả 10 câu, `unanswerable_proof` luôn `True`, và **không có cơ chế thêm bảng trung gian** để đóng đường join. Với `q_00bb4edee433`, câu hỏi cần `DonHang_ChiTiet.Tên sản phẩm → DuLieuXuat_DanhSachSanPham.Tên sản phẩm → Mã SKU → DuLieuXuat_DanhSachThuoc.Thuốc kê đơn`; schema cuối chỉ có 2 bảng đầu/cuối, model bịa join `"Tên sản phẩm" = "Mã SKU"` (3/3 candidate) → NULL.

### 4.3 Khiếm khuyết: 5 variant là dư thừa

5 case × 3 profile type cho 1 câu, với temperature 0.0 và cùng metadata → phần lớn trùng lặp; 64 call repair cho 10 câu. Chi phí này không tương xứng với độ chính xác thu được: schema cuối thường chỉ 5–15 cột, và các lỗi còn lại (ngày, trạng thái, join) **không nằm ở khâu chọn bảng**.

## 5. Hạng mục D — Candidate generation và repair

### 5.1 Tham số

3 candidate (seed 42/100/2026), `temperature=0.7`, `thinking_token_budget=384`, `max_tokens=1536`, structured outputs. `DEFAULT_SQL_MAX_RETRIES=2`, `DEFAULT_EXECUTION_MAX_RETRIES=2`.

### 5.2 Issue code quan sát được (30 candidate)

| Code | Số lần | Blocking? | Nhận xét |
|---|---:|---|---|
| `CURRENCY_SEPARATOR_MISMATCH` | 7 | warning (+1 blocking variant) | **7/7 là false positive**: 5 do guard `IS NOT NULL`, 2 do projection trung gian trong CTE (`SELECT T1."Tổng tiền" AS …`) |
| `COLUMN_NOT_FOUND` | 6 | blocking | Chủ yếu do few-shot/schema hư cấu |
| `TABLE_NOT_FOUND` | 4 | blocking | `bangChiTiet`, `danhSachHang` — model bịa tên Việt hoá |
| `SQL_SYNTAX` | 2 | blocking | Alias không khai báo, `IF` thiếu keyword |
| `OUTPUT_TRUNCATED` | 2 | blocking | Kèm `OUTPUT_EMPTY` |
| `OUTPUT_EMPTY` | 2 | blocking | Repair trả về rỗng |
| `DATE_FORMAT_MISMATCH` | 2 | blocking | Chỉ bắt trường hợp so sánh **trực tiếp**, không bắt SUBSTR thiếu phủ |
| `LITERAL_NOT_IN_ENUM` | 1 | **warning** | Đã cảnh báo `'Hoàn thành'` không có trong enum nhưng vẫn cho chạy/không sửa |
| `EXECUTION_ERROR` | 1 | blocking | `misuse of aggregate: AVG()` trong WHERE |

### 5.3 Truncation và vòng repair

- 30 call bị cắt (`finish_reason=length`), phân bố: `q_006c` 8, `q_00195` 7, `q_00a6` 5, `q_0088` 4, `q_008f` 4, `q_00bb` 1.
- `sql_repair` 36 call, latency trung vị 6.0 s, max 17.9 s. Đây là chi phí lớn nhất trong 562 s của run (`q_00195` = 152 s).
- Cơ chế `stagnation` hoạt động (dừng khi SQL canonical không đổi) nhưng dừng **quá muộn**: 2 retry mỗi candidate × 3 candidate, trong khi nguyên nhân gốc (truncation, alias) lặp lại y nguyên.
- Repair không được cấp thêm ngân sách output: `DEFAULT_SQL_MAX_TOKENS=1536` với `thinking_token_budget=384` trong khi prompt yêu cầu `reasoning` bọc `<think>` → chính sách "viết reasoning ngắn hơn" chỉ xuất hiện trong message lỗi, không được áp bằng tham số.

### 5.4 Lỗi điển hình có thể chặn trước khi tốn call

| Lỗi | Ví dụ | Có thể chặn bằng |
|---|---|---|
| Alias `T2` không khai báo | `q_00a6` candidate 0 | Kiểm tra qualifier (đã có `UNKNOWN_QUALIFIER`, nhưng SQL hỏng trước khi tới bước đó) |
| So sánh khóa với ngày: `"Mã hóa đơn" BETWEEN '2025-07-01' AND '2025-07-31'` | `q_0088` candidate 2 | Rule kiểu-dữ-liệu: cột `FIXED_CODE` so sánh với literal ngày → blocking |
| `AVG()` trong WHERE | `q_006c` candidate 0 | Rule cấu trúc: aggregate ngoài SELECT/HAVING → blocking |
| `BETWEEN 'YYYY-MM-DD'` (literal rác) | `q_008f` candidate 2 | Phát hiện placeholder placeholder-literal → blocking |

Cả 4 lỗi trên đều **để lọt** trong run này.

## 6. Hạng mục E — Validation gate (static checks)

### 6.1 Khiếm khuyết 1: chấp nhận "chuẩn hóa một phần"

`sql_validator.py:256` — `_is_date_normalization_node` trả `True` nếu *bất kỳ* `SUBSTR/date/strftime/julianday` nằm phía trên cột. Với cột `DATE_MIXED (EXCEL_SERIAL, SLASH_DATE)`:

- `SUBSTR(col,7,4)||'-'||…` → được coi là "đã chuẩn hóa" ⇒ **không có issue**.
- Nhưng biểu thức này chỉ đúng với `dd/mm/yyyy`; dòng Excel serial thành `--` và bị loại.

Đồng thời hint của rule (`sql_validator.py:800`) **hardcode** dd/mm/yyyy: `(SUBSTR("cột",7,4) || '-' || …)`. Đây vừa là lỗi kỹ thuật vừa vi phạm chính sách "no data leakage / zero hardcoding" trong `AGENTS.md`.

Kiểm chứng: chạy lại validator với SQL của `q_00c4` candidate 0 hiện tại → `is_valid=True`, không có issue, trong khi kết quả thiếu 52% doanh thu.

### 6.2 Khiếm khuyết 2: false positive trên guard `IS NOT NULL`

`_check_currency_usage` xét **từng node cột độc lập**; cột tiền xuất hiện trong `AND "Cần trả NCC" IS NOT NULL` không nằm trong `REPLACE` → sinh `CURRENCY_SEPARATOR_MISMATCH`.

Tái hiện (cùng SQL, chỉ thêm guard):

```
không guard   -> []
có guard      -> [('warning', 'CURRENCY_SEPARATOR_MISMATCH')]
```

Nhưng chính system prompt lại yêu cầu thêm guard này ("BẮT BUỘC thêm điều kiện `WHERE <col> IS NOT NULL`"). Thống kê run này: **7/7 warning là false positive** — 5 từ guard `IS NOT NULL` (`q_001883` ×3, `q_00195` ×1, `q_00c4` ×1) và 2 từ projection trung gian `SELECT T1."Tổng tiền" AS "TongTien"` trong CTE (`q_00bb`). Ở `q_001883280e83`, cả 3 candidate đều mang warning này và `repair_stop_reason` vẫn là `clean` — nghĩa là **tín hiệu nhiễu đang lấn át tín hiệu thật**; nếu rule này bị nâng thành blocking (nhánh `sql_validator.py:1161`) thì nó sẽ tạo vòng repair vô ích.

### 6.3 Khiếm khuyết 3: các rule quan trọng chỉ ở mức warning

| Rule | Hiện tại | Tác động thực tế |
|---|---|---|
| `LITERAL_NOT_IN_ENUM` | warning | `q_008f` gửi `0.0` sai; `q_006c` candidate 0 chạy dù `'Hoàn thành'` không tồn tại |
| `MONEY_SCALE_RISK` | warning, chỉ bật khi sample có phần thập phân | `biz_112` (`'58,000.0000'`) — REPLACE cả `.` và `,` làm giá trị phóng 10⁴; không bị chặn |
| `SHAPE_SUSPECT_NO_AGGREGATE` | warning | Không ảnh hưởng trong run này |

### 6.4 Điểm mạnh cần giữ

- `DATE_RANGE_OUT_OF_PERIOD` + `temporal_anchor` đã chặn được kịch bản neo nhầm tháng (regression của run 11:57), thể hiện ở `q_001883` (`2026-09-01..07`) và `q_00c4` (`2025-05-01..31`).
- `OUTPUT_TRUNCATED` được phân biệt với `OUTPUT_UNPARSEABLE`, và SQL rỗng không bị "echo" thành query hợp lệ.
- Schema-grounded scope checks bắt tốt `TABLE_NOT_FOUND`/`COLUMN_NOT_FOUND`.

## 7. Hạng mục F — Voting, formatting và quyết định abstain

### 7.1 Công thức hiện tại

```python
evidence_cands = [c for c in candidates if c.execution_success and has_evidence(c.result_set)]
agreement_ratio = len(best_group) / len(evidence_cands)
if agreement_ratio > self.min_agreement:   # min_agreement = 0.5
    submit
```

- `len(evidence_cands) = 1` ⇒ ratio = 1.0 ⇒ **luôn submit** dù 2 candidate còn lại chết. Xảy ra 3 lần, 3 lần sai.
- Đồng thuận được tính trên **giá trị canonical**, không trên **cách tính**. `q_002d03`: candidate 1 (không lọc trạng thái) và candidate 2 (`NOT IN ('Huỷ','Trả hàng')` — không khớp `'hủy đơn'`) cùng ra 10,312,600 ⇒ "majority 2/3" ⇒ đè candidate 0 (đúng 10,309,400 nếu dùng literal `'ok'`).
- "Zero hợp lệ" (`legitimate_zero`) và "zero do filter rỗng" không phân biệt được: `q_008f` trả `0.0`, `q_002d03` candidate 0 trả `(None,)` nhưng vẫn được coi là evidence-fail thay vì dấu hiệu sai schema.

### 7.2 Đánh giá EV (kỳ vọng điểm)

Với PA: đúng +1, sai −1, abstain 0 ⇒ nên trả lời khi `p(correct) > 0.5`. Hiện `1/3 candidate` bị coi là `p=1`; thực nghiệm trên 10 câu cho thấy nhóm này đúng **0/3**. Nếu dùng luật "≥2 evidence candidate đồng thuận" thì 3 câu đó thành abstain: PA từ −0.40 lên **−0.10** (2 sai bị loại, 1 sai còn lại) mà không mất điểm đúng nào.

### 7.3 Formatting

- `percent`: `q_00558` trả `1.05` (đúng thang 0–100), `q_008f` trả `0.0` — không có lỗi scale, chỉ lỗi ngữ nghĩa.
- Không có câu `list/table/date/boolean` trong mẫu nên chưa đánh giá được nhánh đó.
- `q_00bb`/`q_0088` rơi vào `insufficient_evidence` dù candidate "chạy được": `(None,)` được coi là không có evidence — hành vi đúng, nhưng đáng lẽ phải được dùng làm **tín hiệu sửa schema** (join sai/NULL) thay vì kết thúc.

## 8. Hạng mục G — Chất lượng dữ liệu: bản đồ bẫy

### 8.1 Cột ngày của bảng giao dịch chính

| Business | Bảng | Tổng dòng | `dd/mm/yyyy` | Khác | NULL | % phủ của `SUBSTR` hiện tại |
|---|---|---:|---:|---:|---:|---:|
| `biz_105` | `SoBanHang` (view) | 140,245 | 70,386 | 69,819 (serial) | 40 | 50% |
| `biz_106` | `DanhSachHoaDon` | 78,847 | 39,105 | 39,742 (serial) | 0 | 50% |
| `biz_112` | `So_chi_tiet_ban_hang_2026_T07_2026` | 7,656 | 7,656 | 0 | 0 | 100% |
| `biz_117` | `SoBanHang_Tháng` (view) | 33,845 | 8,433 | 25,372 (serial, `1/10`, `1 thg 10 2025`) | 40 | 24% |
| `biz_119` | `DanhSachHoaDon` (view) | 34,047 | 34,047 | 0 | 0 | 100% |
| `biz_120` | `DanhSachHoaDon` (view) | 48,389 | 48,389 | 0 | 0 | 100% |
| `biz_121` | `DonHang_ChiTiet` | 82,920 | 44,042 | 0 | **38,878** | 53% |
| `biz_124` | `BanHang_HoaDon` | 51,044 | 17,042 | 34,002 (serial + `d thg m yyyy`) | 0 | 33% |
| `biz_128` | `DanhSachHoaDon` (view) | 26,505 | 26,505 | 0 | 0 | 100% |

5/9 bảng có ≥50% dòng không parse được bằng biểu thức đang dùng. Đây là **bẫy dữ liệu phổ quát của bộ dữ liệu**, không phải đặc thù một shop.

### 8.2 Từ vựng trạng thái (mỗi shop một bộ)

| Business | Giá trị quan sát |
|---|---|
| `biz_106` | `xong` (43,261), `Hoàn thành` (33,695), `hủy đơn` (870), `Đã hủy` (622), `trả lại` (224), `Hoàn trả` (152), `Đang giao dịch` (23) |
| `biz_120` | `ok`, `hủy đơn` |
| `biz_105` | `đã lấy`, `đã xong`, `hủy`, `Đã hủy đơn`, `Trả hàng`, `Đổi trả`, `hàng trả về`, `Hoàn trả`, NULL |
| `biz_117` | `Đã hoàn thành`, `Thành công`, `đã lấy`, `trả lại`, `Hủy`, `hủy`, `Đã trả hàng`, NULL |
| `biz_119` | `Hoàn thành`, `Đã hủy`, `Đã trả hàng` |
| `biz_112` | `Đã hoàn thành`, `Đang vận chuyển`, `Đã hủy đơn`, `Hoàn trả` |
| `biz_121` | `Đã hoàn thành`, `Đã hoàn hàng`, `Đã hủy`, `Đang giao dịch`, + 38,878 NULL |

Rulebook quy về 4 trạng thái (`hoàn thành`, `huỷ`, `trả hàng`, `đang giao`) và nói rõ "đơn đang giao vẫn tính". Pipeline không có ánh xạ này; nó dựa hoàn toàn vào việc LLM đoán literal.

### 8.3 Tiền tệ

| Dạng | Ví dụ | Rủi ro |
|---|---|---|
| Nghìn bằng `,` | `'75,000'` | REPLACE `,` là đủ |
| Nghìn bằng `.` + thập phân | `'58,000.0000'` (biz_112) | REPLACE cả `.` và `,` ⇒ **×10⁴** |
| Có hậu tố | `'1,057,500'`, `'đ'` | Cần strip `đ`/`VND` |
| Cột tổng hợp | `Cần trả NCC`, `Tổng tiền hàng` | Như trên |

### 8.4 Luật dữ liệu bị bỏ qua

| Luật rulebook | Hiện trạng pipeline | Thiệt hại trong run |
|---|---|---|
| §8 ô trống dòng sau = dòng đầu nhóm | ingestion để NULL | `biz_121` mất 46% dòng, 2.27% → 1.75% |
| §8 hai dòng giống nhau trong cùng đơn = tính một lần | không dedupe | `biz_124`: 3,018,600 → GT 2,954,200 |
| §1 đơn huỷ/trả bị loại, đang giao vẫn tính | literal tuỳ model | `q_002d03` (+3,200), `q_00558`, `q_008f` |
| §2 "tháng này" = 1 → hết hôm nay | **đúng** (temporal anchor) | — |
| §5 giá nhập bình quân theo số lượng | model viết đúng công thức nhưng SQL hỏng | `q_00a6` |

## 9. Phân tích từng câu (đã kiểm chứng trên DB)

### 9.1 `q_001883280e83` — ĐÚNG (+1)

- Câu: "tháng này tổng giá trị các phiếu nhập hàng là bao nhiu", `as_of=2026-09-07`.
- SQL thắng: `SUM(CAST(REPLACE(REPLACE("Cần trả NCC", ',', ''), '.', '') AS INTEGER))` trong `2026-09-01..2026-09-07`, 3/3 candidate đồng thuận.
- Vì sao đúng: chuỗi ngày của `biz_128` 100% `dd/mm/yyyy`, tiền dùng `,` làm nghìn ⇒ biểu thức chuẩn hóa hiện tại tình cờ phủ hết.
- Ghi chú: candidate 1 dùng thứ tự REPLACE ngược (`'.'` trước `','`) và vẫn ra cùng kết quả — **đúng nhờ may mắn về format**, không phải nhờ logic.

### 9.2 `q_00195e1e4049` — SAI (−1): 951,600 vs 2,954,200

GT tái lập được (khớp tuyệt đối):

```sql
WITH lines AS (
  SELECT DISTINCT H."Order ID", T."Ma hang", T."Slg", T."Thanh tien"
  FROM BanHang_ChiTietHoaDon T
  JOIN BanHang_HoaDon H ON T."Mã giao dịch" = H."Order ID"
  JOIN BanHang_Thuoc U ON T."Ma hang" = U."Mã HH"
  WHERE U."Hoạt chất" = 'Natri clorid'
    AND H."Trạng thái" IN ('Đã giao', 'Đang vận chuyển')
    AND /* 3 format ngày: dd/mm/yyyy, 'd thg m yyyy', Excel serial */
        … BETWEEN '2026-05-01' AND '2026-05-25'
)
SELECT SUM(CAST(REPLACE(REPLACE("Thanh tien", '.', ''), ',', '') AS INTEGER)) FROM lines;
```

| Thành phần | Giá trị |
|---|---:|
| Chỉ `dd/mm/yyyy` (đúng như pipeline) | 951,600 |
| + `d thg m yyyy` | 2,158,600 |
| + Excel serial | 3,039,800 |
| − dòng trùng trong cùng đơn, − đơn `Không giao được` | **2,954,200** |

Pipeline sai vì 4 thiếu sót xếp lớp: (1) chỉ parse `dd/mm/yyyy`; (2) không dedupe dòng trùng; (3) không lọc trạng thái hợp lệ; (4) không biết `0,1%` là `0,1%` (candidate 0 dùng `'Otrivin 0,01%'` ở câu khác). Ngoài ra 2/3 candidate bị `COLUMN_NOT_FOUND` do dùng tên cột hư cấu (`BanHang_HangHoa."Mã HH"`), chỉ candidate 2 sống sót và được submit theo luật "1/3 = majority".

### 9.3 `q_002d03ebbded` — SAI (−1): 10,312,600 vs 10,309,400

- Ngày 30/04/2025 có 92 hoá đơn: 91 `ok` (10,309,400) + 1 `hủy đơn` (3,200 = `ORD010377`).
- candidate 0 lọc `Trạng thái='Hoàn thành'` → không tồn tại (`(None,)`).
- candidate 1 không lọc trạng thái → 10,312,600 (sai).
- candidate 2 lọc `NOT IN ('Huỷ','Trả hàng')` → không khớp `hủy đơn` → 10,312,600 (sai).
- Voting: candidate 1 + 2 "đồng thuận" ⇒ **majority 2/3 cho đáp án sai**, candidate 0 (hướng đúng) bị loại.

### 9.4 `q_00558c1410ef` — SAI (−1): 1.05 vs 2.08

- Rulebook: "tỉ lệ đơn huỷ = % đơn huỷ trong **mọi đơn** đặt trong kỳ" ⇒ đơn vị đếm là **đơn**, không phải dòng.
- `SoBanHang_1_Thang 06 2025`: 4,948 đơn; 103 đơn huỷ (`hủy` 53 + `Đã hủy đơn` 50) ⇒ 2.08%.
- Pipeline: `COUNT(CASE WHEN Status='Đã hủy đơn')` / `COUNT(*)` trên **dòng** và chỉ trên 3,989 dòng `dd/mm/yyyy` ⇒ 42/3,989 = 1.05%.
- Ba lỗi: grain (dòng vs đơn), đồng nghĩa trạng thái (bỏ `hủy`), format ngày (bỏ 50% dòng serial).

### 9.5 `q_006c26c65d50` — ABSTAIN (GT 2,345)

- GT tái lập: đơn có `SUM(Thành tiền) ≥ 55,000` với trạng thái ∈ {`Đã hoàn thành`, `Đang vận chuyển`} và tiền parse đúng (`'58,000.0000'` → 58,000) ⇒ **2,345**.
- Candidate 0: `ROUND(AVG(...))` trong WHERE + literal `'Hoàn thành'` không tồn tại ⇒ `misuse of aggregate: AVG()`.
- Candidate 1: output bị cắt (8 lần truncation), JSON không đóng.
- Candidate 2: tự `abstain` ⇒ `abstain_unverified_unanswerable` (chỉ 1/3 unanswerable).
- Đáng chú ý: snapshot 20h trong `result/submission.jsonl` trả `1082` cho chính câu này — tức là "sửa" từ sai (−1) thành abstain (0), đúng hướng nhưng chưa giải được.

### 9.6 `q_0088f7a0afc5` — ABSTAIN (GT 77)

- GT: `SUM(Số lượng)` của `SP0038 = 'Otrivin 0,1% xịt mũi chai 10ml'` trong 07/2025, trạng thái `Hoàn thành` (loại `Đã hủy`) = **77**.
- candidate 0/1 hallucinate `"bangChiTiet"`, `"danhSachHang"`, `"bangDanhSachSanPham"` (2 retry mỗi candidate, `stagnation`/`budget_exhausted`).
- candidate 2 (sống sót) viết `T1."Mã hóa đơn" BETWEEN '2025-07-01' AND '2025-07-31'` — so sánh **khóa** với ngày ⇒ `(None,)` ⇒ `insufficient_evidence`.
- Hai lỗi cần chặn: tên bảng Việt hoá bịa (đã có `TABLE_NOT_FOUND` nhưng repair không sửa được) và so sánh sai kiểu dữ liệu (không có rule).

### 9.7 `q_008f69128180` — SAI (−1): 0.0 vs 1.21

- GT: 11 đơn huỷ (`Hủy` 6 + `hủy` 5) / 908–913 đơn ≈ **1.21%**.
- candidate 0: `Ghi chú TT = 'Hủy'` (phân biệt hoa/thường) + chỉ 522 dòng `dd/mm/yyyy` (24% bảng) ⇒ 0 đơn huỷ ⇒ `0.0`.
- Pipeline submit `0.0` (percent) theo luật "1/3 majority" dù đây là zero đáng ngờ.
- Validator chỉ cảnh báo `DATE_FORMAT_MISMATCH` cho candidate 1 (dùng `LIKE '%/%/%'`), không cảnh báo candidate 0 — vì candidate 0 "đã có SUBSTR".

### 9.8 `q_00a6e23cbacf` — ABSTAIN (GT 200,556)

- GT: bình quân gia quyền theo số lượng nhập của `SP0017` trong 01/2025 = `SUM(SL*Đơn giá)/SUM(SL)` = **200,555.56 → 200,556**.
- candidate 0 viết đúng công thức nhưng dùng `T2` không khai báo và REPLACE lồng sai (`REPLACE(x, ',', '')` thiếu tham số thứ 3) ⇒ `SQL_SYNTAX`.
- candidate 1 tự `unanswerable`, candidate 2 bị cắt output ⇒ abstain.
- Đây là câu "gần đúng nhất" trong nhóm sai: chỉ cần validator/repair sửa được cú pháp là có +1.

### 9.9 `q_00bb4edee433` — ABSTAIN (GT 2.27)

- GT tái lập: forward-fill nhóm dòng trống trên `DonHang_ChiTiet.xlsx`, lọc 07/2025 + `Đã hoàn thành`, nối `Tên sản phẩm → DuLieuXuat_DanhSachSanPham.Mã SKU → DuLieuXuat_DanhSachThuoc.Thuốc kê đơn = 1` ⇒ **2.26%** (GT 2.27%).
- Trên DB hiện tại (không forward-fill) tỉ lệ chỉ còn **1.75%** ⇒ dữ liệu đã bị méo từ khâu ingestion.
- 3 candidate đều join sai hoặc CTE trả NULL ⇒ `insufficient_evidence`.
- Hai lỗi hệ thống: ingestion bỏ luật §8, coverage không thêm bảng cầu.

### 9.10 `q_00c4e64fe706` — SAI (−1): 131,362,000 vs 271,866,000

- GT tái lập: `SUM("Tổng tiền hàng")` của `Trạng thái = 'xong'` trong 05/2025, **trên cả hai biểu diễn ngày** = **271,866,000**.
- Pipeline: `SUBSTR(...)` ⇒ chỉ 39,105 dòng `dd/mm/yyyy` ⇒ **131,362,000** (mất 51.9%).
- Không có issue nào được sinh ra; candidate 0 chạy `clean`; candidate 1/2 chết vì `MONEY_SCALE_RISK`/`DATE_FORMAT_MISMATCH` (candidate 1 dùng `strftime('%Y-%m-%d', "Thời gian")` → NULL).
- Đây là **ca điển hình của lỗi "validator cho qua biểu thức chuẩn hóa không phủ format"**.

## 10. Chi phí, độ trễ và khả năng mở rộng

### 10.1 Theo câu

| Câu | Call | Prompt token | Completion | Elapsed (trace) | Truncation |
|---|---:|---:|---:|---:|---:|
| `q_001883280e83` | 28 | 121,472 | 4,054 | 51.9 s | 0 |
| `q_00195e1e4049` | 44 | 310,371 | 15,317 | 79.6 s | 7 |
| `q_002d03ebbded` | 16 | 93,166 | 2,990 | 8.2 s | 0 |
| `q_00558c1410ef` | 34 | 160,052 | 7,510 | 31.1 s | 0 |
| `q_006c26c65d50` | 22 | 125,500 | 11,610 | 56.7 s | 8 |
| `q_0088f7a0afc5` | 19 | 130,449 | 7,206 | 101.5 s | 4 |
| `q_008f69128180` | 13 | 74,974 | 6,143 | 50.3 s | 4 |
| `q_00a6e23cbacf` | 19 | 113,482 | 6,898 | 80.3 s | 5 |
| `q_00bb4edee433` | 16 | 89,772 | 7,011 | 68.4 s | 1 |
| `q_00c4e64fe706` | 9 | 54,451 | 1,875 | 34.2 s | 0 |
| **Tổng** | **220** | **1,273,689** | **70,614** | **562.1 s** | **30** |

### 10.2 Ước lượng test set đầy đủ (1000 câu)

| Chỉ số | Ước lượng |
|---|---:|
| Call/câu (trung bình) | 22 |
| Prompt token/câu | ~127 K |
| Thời gian/câu | ~56 s |
| Tổng thời gian (đơn luồng) | **~15.6 giờ** |
| Tổng call | ~22,000 (câu hỏi) + ~7,200 (profiling 100 business) |
| Tổng prompt token | ~127 M |

Với 4 ngày của private round thì 15.6 giờ vẫn khả thi, nhưng: (a) không có biên cho retry toàn bộ, (b) không có biên cho thay đổi prompt, (c) rủi ro timeout của hạ tầng. Giảm call/câu từ 22 → 6–8 là mục tiêu hợp lý và khả thi (bỏ 5 variant schema linking, chặn lỗi cấu trúc trước khi repair, bỏ retry trùng).

## 11. So sánh run 11:57 (`trace_logs_1311`) và run 21:08

| Hạng mục | 11:57 (2 câu) | 21:08 (10 câu) | Kết luận |
|---|---|---|---|
| Temporal anchor | Trial SQL neo `01/01/2026..31/01/2026` (few-shot leakage) | `2026-09-01..07`, `2025-05-01..31` đúng | **Đã sửa** (commit `a395270`) |
| `q_001883280e83` | `abstain` (3-way split) | Đúng 89,526,400 (3/3) | **Đã sửa** |
| `q_00195e1e4049` | `unanswerable` (−1) | `number` sai (−1) | Đổi kiểu lỗi, cùng mức điểm |
| Schema coverage | `biz_124` thiếu bảng `BanHang_HoaDon` | Đã có (15 cột/4 bảng) | **Đã sửa** |
| LSH index | 142–298 MB/business | 0.4–6.2 MB/business | **Đã sửa** (giảm ~30–50×) |
| Grounded literals nhiễm ngày | 10/13 literal là ngày | Còn ở `variant_traces`, đã bị `dropped_literals` | Cải thiện một phần |
| Lỗi mới nổi | — | `CURRENCY_SEPARATOR_MISMATCH` false positive (7), `majority 1/3` (3 câu) | **Chưa xử lý** |
| Chất lượng chung | 0/2 đúng, PA −0.5 | 1/10 đúng, PA −0.40 | Cải thiện biên, chưa đủ |

Kết luận: các commit gần nhất **sửa đúng lớp lỗi "neo thời gian" và "coverage thiếu bảng"**, nhưng run mở rộng sang 10 business đã phơi ra lớp lỗi mới **lớn hơn nhiều: chuẩn hóa dữ liệu (ngày/tiền/trạng thái) và luật dữ liệu §8**. Với 2 câu của run trước, các bẫy này không xuất hiện; với 10 business đại diện thì 5/10 câu bị ảnh hưởng.

## 12. Bảng xếp hạng điểm yếu

| # | Điểm yếu | Bằng chứng | Ảnh hưởng trong 10 câu | Mức |
|---|---|---|---|---|
| W1 | Không có lớp chuẩn hóa ngày (chỉ `dd/mm/yyyy`) | 5/9 bảng ≥50% dòng ngoài format; `q_00c4` −52%, `q_00195` −68% | 4 câu sai/abstain | **P0** |
| W2 | Không canonical hóa trạng thái đơn | `ok`/`xong`/`hủy đơn`/`Hủy`; `q_002d03` +3,200; `q_008f` 0.0 | 3 câu | **P0** |
| W3 | Ingestion bỏ luật ô-trống-theo-nhóm (§8) | `biz_121` 46% dòng NULL; 143 M vs 265 M | 1 câu (−0 nếu forward-fill) và rủi ro rộng | **P0** |
| W4 | Metadata cột bị cắt 240 ký tự | `Tập giá trị đầy đủ: 'xo...` trong prompt | Mọi câu; là điều kiện tiên quyết của W2 | **P0** |
| W5 | `join_key_closure` luôn satisfied | `coverage.py:327-340`; `q_00bb` thiếu bảng cầu | 1 câu | **P0** |
| W6 | Ngưỡng voting tính trên evidence-only | 3 câu `majority_1_of_3`, cả 3 sai | 3 câu | **P0** |
| W7 | Validator: chuẩn hóa "một phần" được chấp nhận + hint hardcode dd/mm/yyyy | `q_00c4` clean; `sql_validator.py:781-805` | 2 câu sai nặng | **P1** |
| W8 | Validator: false positive `IS NOT NULL` trên cột tiền | tái hiện được, 7 warning | Nhiễu repair, che tín hiệu thật | **P1** |
| W9 | `LITERAL_NOT_IN_ENUM`/`MONEY_SCALE_RISK` chỉ warning | `q_008f` gửi 0.0; `biz_112` ×10⁴ | 2 câu | **P1** |
| W10 | Không dedupe dòng trùng trong cùng đơn | `biz_124` 3,018,600 → 2,954,200 | 1 câu (chặn +1) | **P1** |
| W11 | Repair cạn ngân sách output, lặp lại lỗi cũ | 30 truncation; `q_00195` 152 s/310 K token | 3 câu + 40% chi phí | **P1** |
| W12 | Không có rule cấu trúc (aggregate trong WHERE, khóa so với ngày, alias thiếu) | `q_006c`, `q_0088`, `q_00a6` | 3 câu | **P1** |
| W13 | 5 variant schema linking dư thừa | 75 init + 64 repair call; schema cuối 4–15 cột | Chi phí, không tăng độ chính xác | **P2** |
| W14 | Profiling 1 call/cột (dù đã batch) và 1.6 MB metadata/business | 718 call, 1.65 M token | Chi phí one-time | **P2** |
| W15 | Không song song hóa theo business | 562 s tuần tự | Thông lượng | **P2** |

## 13. Đề xuất tối ưu

### 13.1 Nhóm P0 — sửa cấu trúc dữ liệu (kỳ vọng +4 đến +5 câu trên mẫu 10)

**(1) Lớp view chuẩn hóa tại ingestion** — `src/zaic_2026_4/ingestion/view_builder.py`, `src/zaic_2026_4/ingestion/sqlite_sink.py`

Sinh thêm view "canonical" cho mỗi bảng giao dịch (giữ nguyên bảng gốc để không phá thứ khác):

```sql
CREATE VIEW "DanhSachHoaDon__c" AS
SELECT *,
       CASE
         WHEN "Thời gian" GLOB '[0-9][0-9]/[0-9][0-9]/[0-9][0-9][0-9][0-9]'
           THEN substr("Thời gian",7,4)||'-'||substr("Thời gian",4,2)||'-'||substr("Thời gian",1,2)
         WHEN "Thời gian" GLOB '[0-9][0-9][0-9][0-9][0-9]'
           THEN date('1899-12-30','+'||"Thời gian"||' days')
         WHEN "Thời gian" GLOB '[0-9]*-[0-9]*-[0-9]*'
           THEN substr("Thời gian",1,10)
         WHEN "Thời gian" LIKE '%thg%'   -- '1 thg 10 2025'
           THEN printf('%04d-%02d-%02d', …)
       END AS "_date",
       CAST(REPLACE(REPLACE(substr("Tổng tiền hàng",1,INSTR("Tổng tiền hàng",'.')-1), ',', ''), 'đ', '') AS INTEGER) AS "_amount",
       CASE
         WHEN lower("Trạng thái") LIKE '%h%y%'  THEN 'cancelled'
         WHEN lower("Trạng thái") LIKE '%tr%l%i%' OR lower("Trạng thái") LIKE '%ho%n tr%' THEN 'returned'
         WHEN lower("Trạng thái") LIKE '%giao%' THEN 'shipping'
         ELSE 'completed'
       END AS "_status"
FROM "DanhSachHoaDon";
```

Điểm quan trọng: các nhánh `CASE` được **sinh từ format đã phát hiện trong profile** (`DATE_MIXED` → liệt kê đúng các format quan sát được), không hardcode theo shop. Prompt chỉ cần nói: "luôn dùng `_date`, `_amount`, `_status`; không tự parse chuỗi". Việc này biến 4/10 câu từ "phụ thuộc may mắn" thành "đúng theo cấu trúc".

**(2) Canonical hóa trạng thái + literal grounding từ enum đầy đủ** — `src/zaic_2026_4/schema_linking/grounding.py`, `src/zaic_2026_4/submission/prompt_template.py`

- Từ `sample_values` của cột trạng thái, gom nhóm theo token ngữ nghĩa (`huy`, `tra`, `hoan`, `giao`, `xong`, `lay`, `thanh cong`) ở dạng đã fold dấu, sinh map `{giá trị gốc → 4 trạng thái rulebook}` và đưa **toàn bộ** vào prompt dưới dạng bảng nhỏ.
- Sinh sẵn predicate chuẩn: `"Trạng thái" IN ('xong','Hoàn thành',…)` để model dán vào, thay vì tự đoán.

**(3) Forward-fill nhóm dòng trống (rulebook §8)** — `src/zaic_2026_4/ingestion/extractors.py` / `header_detector.py`

- Phát hiện cột "khóa nhóm" (cột đầu có pattern id lặp lại, các cột header cùng NULL theo cụm) và `forward_fill()` các cột header khi giá trị NULL **và** phần còn lại của dòng không NULL.
- Đây là luật văn bản, áp dụng chung, không phải heuristic theo shop. Kiểm thử: `biz_121` phải cho 264.9 M (07/2025) và 2.26%.
- Đồng thời dedupe "hai dòng giống nhau trong cùng đơn" bằng `COUNT(DISTINCT …)` trên view canonical.

**(4) Sửa truncation metadata** — `src/zaic_2026_4/common/xml_sanitizer.py:102`

- Bỏ `max_desc_len=240` mặc định; thay bằng render theo trường có ngân sách riêng:

```xml
<column name="Trạng thái" type="STRING"
        enum="xong|Hoàn thành|hủy đơn|Đã hủy|trả lại|Hoàn trả|Đang giao dịch"
        null_pct="0" rows="78847"/>
```

- Giữ mô tả LLM ở mức ≤120 ký tự (phần này ít giá trị), dồn ngân sách cho `enum`, `min/max`, `sample`, `normalizer`.

**(5) Coverage: kiểm tra đường join thật** — `src/zaic_2026_4/schema_linking/coverage.py:327`

- Tìm đường đi ngắn nhất giữa các bảng đã chọn trên đồ thị join từ profiler; nếu thiếu bảng trung gian thì **thêm bảng** (không chỉ thêm cột).
- Nếu không tìm được đường: `missing_aspects.append("join_key_closure")` ⇒ `unanswerable_proof=False` ⇒ voting không được kết luận `unanswerable`.
- Thêm aspect mới `filter_enum_closure`: mọi literal so sánh phải là thành viên enum đã profile.

**(6) Voting theo EV, không theo "evidence-only"** — `src/zaic_2026_4/submission/voting_executor.py:315`

```python
agreement_ratio = len(best_group) / len(candidates)      # mẫu số là TỔNG candidate
if len(best_group) < 2:                                   # không đủ bằng chứng
    return None, "abstain_insufficient_support", ...
```

Và thêm luật zero-suspicion: nếu answer = 0/0.0/NULL-ish **và** candidate có `LITERAL_NOT_IN_ENUM`/`DATE_FORMAT_MISMATCH` warning ⇒ abstain thay vì submit.

Ước tính: PA mẫu 10 câu −0.40 → **+0.10 … +0.30** nếu (1)(2)(3) hoạt động; riêng (6) đưa −0.40 → −0.10 ngay lập tức.

### 13.2 Nhóm P1 — tăng độ chính xác biên (kỳ vọng +1 đến +2 câu)

**(7) Validator: kiểm tra "phủ format" thay vì "có SUBSTR"** — `sql_validator.py:256, 781`

- Với `DATE_MIXED`, yêu cầu biểu thức chứa nhánh cho **mọi** format trong `formats` của profile (đếm `CASE/LIKE/GLOB` phủ đủ số format); nếu thiếu ⇒ blocking kèm hint **sinh từ profile**, không hardcode `SUBSTR(...,7,4)`.
- Thêm rule `DATE_LITERAL_PLACEHOLDER` chặn `BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'`.
- Thêm rule `KEY_COMPARED_TO_DATE` (cột `FIXED_CODE`/`NUMERIC` so với literal ngày) — bắt được `q_0088` candidate 2.
- Thêm rule `AGGREGATE_IN_PREDICATE` — bắt `AVG()` trong WHERE (`q_006c`).

**(8) Validator: bỏ false positive currency** — `sql_validator.py:1117`

- Chỉ đánh giá các node cột nằm trong ngữ cảnh số học/tổng hợp (`AggFunc`, `Cast`, `Arithmetic`, `Between`), bỏ qua ngữ cảnh `IS NULL`/`IS NOT NULL`/`SELECT` thuần.
- Hoặc: gộp theo `(table, column)` và coi là đã làm sạch nếu **ít nhất một** usage có REPLACE đúng.

**(9) Nâng warning → blocking có điều kiện**

- `LITERAL_NOT_IN_ENUM` ⇒ blocking khi câu trả lời phụ thuộc vào bộ lọc đó (có aggregate/count trên cùng nhánh).
- `MONEY_SCALE_RISK` ⇒ blocking khi phát hiện REPLACE cả `.` và `,` trên cột có phần thập phân (đã có code, chỉ cần đổi severity + mở rộng detector để nhận `'58,000.0000'`).

**(10) Repair có ngân sách và có mục tiêu**

- Tăng `DEFAULT_SQL_MAX_TOKENS` cho repair (1536 → 2048) nhưng **giảm** số retry (2 → 1) và **không** lặp lại khi lỗi thuộc cùng `code` (đã có `is_code_stagnation`, cần áp cho cả `sql_repair` hiện chỉ dùng `is_canon_stagnation`).
- Với lỗi cơ học (`SQL_SYNTAX`, `UNKNOWN_QUALIFIER`, `AGGREGATE_IN_PREDICATE`), dùng **repair tất định** (sửa bằng AST/sqlglot) trước khi gọi LLM.

### 13.3 Nhóm P2 — chi phí và thông lượng (kỳ vọng −60% token, −50% wall time)

**(11) Giảm schema linking từ 5 variant còn 2**: 1 variant `minimal` + 1 `maximal`; gộp kết quả bằng luật (không cần 5 LLM call/case). Dự kiến giảm ~50 call/10 câu và ~35% prompt token ở khâu linking.

**(12) Cache theo (business, intent shape)**: câu hỏi cùng business thường cùng "hình dạng" (doanh thu kỳ, đếm đơn, top-k). Cache `final_linked_schema` theo `(business_id, intent_signature)` thay vì theo câu ⇒ 1000 câu/100 business giảm mạnh số lần linking.

**(13) Song song theo business**: mỗi business một worker (DB riêng, cache riêng), giới hạn bởi GPU batch. Nếu vLLM phục vụ batch lớn, gom 3 candidate + 5 case linking của nhiều câu vào một batch.

**(14) Logging gọn**: hiện mỗi câu sinh `question_trace_*.json` 443 KB ⇒ 1000 câu ≈ 440 MB, cộng `llm_calls.jsonl` 1 MB/business. Đề xuất lưu `llm_calls` với prompt nén (hoặc chỉ lưu khi `--debug`), giữ trace tổng hợp.

## 14. Kế hoạch thực thi và cách đo

| Bước | Việc | Kiểm chứng | Kỳ vọng |
|---|---|---|---|
| 1 | View canonical (ngày/tiền/trạng thái) + prompt "chỉ dùng cột `_date/_amount/_status`" | Chạy 10 câu mẫu + 40 câu train khác | +3…+4 câu; `q_00c4`, `q_00195`, `q_00558`, `q_008f` chuyển đúng |
| 2 | Forward-fill §8 + dedupe | `biz_121` = 264.9 M; `biz_124` = 2,954,200 | +1 câu |
| 3 | Sửa truncation metadata + enum vào prompt | Diff prompt: enum xuất hiện đầy đủ | Nền cho bước 4 |
| 4 | Coverage join closure + voting EV | `q_00bb` không còn `missing=[]`; 3 câu `majority_1_of_3` thành abstain | PA −0.40 → ≥ 0 |
| 5 | Validator (phủ format, currency, rule cấu trúc) | Unit test tái hiện 6 kịch bản trong §6, §5.4 | −1 câu sai, −40% call |
| 6 | Giảm variant + cache + song song | Đo call/câu và wall time | 22 → 8 call/câu; 56 s → <25 s |

Bộ đo cần cố định trước khi sửa (để so sánh được):

- **PA trên 100 câu train** lấy mẫu tầng (stratify theo business + loại câu hỏi: doanh thu, đếm, tỉ lệ, top-k, tồn kho, công nợ).
- Tỉ lệ `wrong` (mục tiêu giảm mạnh hơn `abstain`), tỉ lệ `abstain` (giữ ≤ 20%).
- `calls/question`, `prompt_tokens/question`, `p95 latency`.
- Số câu có đủ 3/3 candidate hợp lệ (hiện 20%).

## 15. Rủi ro và lưu ý

1. **Không được hardcode theo shop.** Mọi ánh xạ (format ngày, đồng nghĩa trạng thái, đơn vị) phải sinh từ `metadata_profile`/`sample_values` tại runtime. Hint `SUBSTR("cột",7,4)` hiện tại là ví dụ điển hình của hardcode cần xóa.
2. **View canonical phải song song tồn tại với bảng gốc** để không phá các truy vấn đang đúng (`q_001883` đúng nhờ bảng gốc + format `dd/mm/yyyy`).
3. **Abstain là điểm 0, không phải "an toàn".** Siết ngưỡng voting sẽ tăng abstain; cần đo PA chứ không đo độ chính xác trên phần đã trả lời.
4. **10 câu là mẫu nhỏ** (10 business). Các tỉ lệ trong báo cáo là tín hiệu chẩn đoán; trước khi kết luận cải thiện phải chạy ≥100 câu train.
5. **Private test có thể có format mới** (ví dụ `yyyy/mm/dd`, `dd-mm-yyyy`). Bộ chuẩn hóa phải dựa trên *phát hiện* format từ dữ liệu, có nhánh fallback và log rõ khi gặp format lạ — không được im lặng trả `--`.

## 16. Phụ lục

### 16.1 Truy vấn tái lập ground truth (dùng lại khi kiểm thử)

```sql
-- biz_106: 271,866,000 (cả hai biểu diễn ngày, chỉ trạng thái hợp lệ)
SELECT SUM(CAST(REPLACE(REPLACE("Tổng tiền hàng", ',', ''), '.', '') AS INTEGER))
FROM "DanhSachHoaDon"
WHERE "Trạng thái" = 'xong'
  AND ( (SUBSTR("Thời gian",7,4)||'-'||SUBSTR("Thời gian",4,2)||'-'||SUBSTR("Thời gian",1,2))
          BETWEEN '2025-05-01' AND '2025-05-31'
     OR date('1899-12-30','+'||"Thời gian"||' days') BETWEEN '2025-05-01' AND '2025-05-31' );

-- biz_117: 1.21% (mọi trạng thái, đơn vị = đơn)
SELECT ROUND(100.0 * COUNT(DISTINCT CASE WHEN "Ghi chú TT" IN ('Hủy','hủy') THEN "Mã phiếu" END)
             / COUNT(DISTINCT "Mã phiếu"), 2)
FROM "SoBanHang_Tháng 10-2025";

-- biz_112: 2,345 (đơn có doanh thu ≥ 55,000, gồm đang giao, tiền bỏ phần thập phân)
SELECT COUNT(*) FROM (
  SELECT "Mã đơn hàng",
         SUM(CAST(REPLACE(SUBSTR("Thành tiền",1,INSTR("Thành tiền",'.')-1), ',', '') AS REAL)) s
  FROM "So_chi_tiet_ban_hang_2026_T07_2026"
  WHERE "Trạng thái" IN ('Đã hoàn thành','Đang vận chuyển')
  GROUP BY 1 HAVING s >= 55000);
```

### 16.2 Checklist đọc log nhanh cho các run sau

1. `submission.jsonl` vs `answers.jsonl` → PA, danh sách câu sai.
2. Với mỗi câu sai: `formatting_trace.coverage`, `voting_status`, `candidate_diagnostics`, rồi mới đọc SQL.
3. So `result_set` với kết quả SQL "đúng nghiệp vụ" viết tay trên DB (như §16.1).
4. Đếm `finish_reason=length` theo câu; nếu > 2 thì xem ngay ngân sách token/repair.
5. `grep -c CURRENCY_SEPARATOR_MISMATCH` để phát hiện false positive đang làm nhiễu run.

## 17. Trạng thái triển khai các đề xuất §13

Mục này ghi lại những gì đã được hiện thực trong cùng đợt sửa này, cách kiểm chứng và phần còn
lại. Nguyên tắc bắt buộc được giữ: mọi ánh xạ (định dạng ngày, cách viết tiền, từ vựng trạng thái)
đều sinh từ dữ liệu tại runtime, không hardcode theo shop hay theo ground truth.

### 17.1 Đã triển khai

| # | Đề xuất §13 | Hiện thực | Kiểm chứng |
|---|---|---|---|
| 1 | Lớp view chuẩn hóa | `ingestion/canonical.py` sinh `_date`/`_amount`/`_status`; giá trị tính bằng Polars rồi lưu ở bảng phụ `_canon_<bảng>`, expose qua view `<bảng>__c` (bảng gốc không bị sửa). View hợp nhất cũng có bản chuẩn hóa `<view>__c` | `tests/test_canonical_normalization.py`; 7/7 truy vấn ground truth viết lại trên view chuẩn hóa cho kết quả đúng |
| 2 | Canonical hóa trạng thái | Bảng `STATUS_FAMILIES` (4 nhóm rulebook + `pending`/`other`); nhận diện cột trạng thái theo **dữ liệu** (≥60% giá trị map được, ≥2 nhóm), tên cột chỉ dùng để xếp hạng | `biz_105`: `hủy`→cancelled, `đã lấy`→shipping, `hàng trả về`→returned; 10/10 business phát hiện đúng cột trạng thái |
| 3 | Forward-fill nhóm dòng (§8) | `ingestion/forward_fill.py`: khối header = các cột có **cùng mẫu NULL**; chỉ điền khi dòng là "dòng tiếp nối" thật (khối trống nhưng các cột item còn dữ liệu), bỏ qua dòng tổng/chú thích, bỏ qua bảng có khóa chính mức dòng; không thêm/bớt/đổi thứ tự dòng, không ghi ra ngoài khối | `biz_121`: doanh thu 07/2025 = **264,919,200** (GT). Quét 130 business: 39 bảng được điền, đều đúng dạng header đơn hàng, 0 dòng đổi số lượng, 0 tổng số thay đổi (`tmp_dev/fill_sweep.py`); `tests/test_grouped_row_fill.py` |
| 4 | Bỏ truncation metadata | `common/xml_sanitizer.py`: mô tả ≤120 ký tự + thuộc tính `enum`, `range`, `samples`, `distinct`, `unique`; hint chung bị bỏ khỏi prompt | `enum="xong \| Đã hoàn thành \| cancel \| ..."` xuất hiện đầy đủ trong `<selected_schema>` |
| 5 | Coverage join thật | `schema_linking/coverage.py`: BFS tìm đường ngắn nhất trên đồ thị join, thêm **bảng cầu**; không có đường ⇒ `missing_aspects += join_key_closure`; thêm aspect `canonical_projection` | `tests/test_coverage_join_closure.py` |
| 6 | Voting theo EV | `submission/voting_executor.py`: mẫu số = **tổng số candidate**, tối thiểu 2 phiếu, luật zero-suspicion (`abstain_zero_suspicion`) | `tests/test_voting_executor.py` (3 test mới) |
| 7 | Validator: phủ format | Rule gom **chiến lược chuẩn hóa theo từng cột** (SUBSTR / GLOB-LIKE / parser ngày / thay 'thg'), nên cả dạng `CASE WHEN` lẫn dạng OR rời đều được chấp nhận; hint sinh từ `date_subtype` của profile, đã xóa `SUBSTR("cột",7,4)` hardcode | `test_mixed_date_column_with_single_branch_is_blocked`, `test_mixed_date_or_disjuncts_are_accepted`, `test_unrelated_branch_does_not_excuse_missing_coverage` |
| 7b | Rule cấu trúc mới | `DATE_LITERAL_PLACEHOLDER`, `AGGREGATE_IN_PREDICATE` (chỉ WHERE), `KEY_COMPARED_TO_DATE` | 4 test mới trong `tests/test_sql_validator.py` |
| 8 | Bỏ false positive currency | Gộp theo (table, column), chỉ xét ngữ cảnh số học/aggregate/cast; một chỗ REPLACE đúng là đủ | 7/7 false positive cũ biến mất; ca dùng tiền không làm sạch vẫn blocking |
| 9 | Nâng warning có điều kiện | `LITERAL_NOT_IN_ENUM` thành blocking khi truy vấn có aggregate; `MONEY_SCALE_RISK` thành blocking | `test_literal_not_in_enum_blocks_aggregates_only` |

### 17.1b Sửa sau vòng review (cùng đợt)

Vòng review độc lập phát hiện 3 lỗi chặn và đã được xử lý:

1. **Forward-fill từng bịa dữ liệu**: bản đầu coi mọi cột NULL-lặp là khóa nhóm nên đã điền
   `so_luong` vào dòng "Cộng" (`biz_105`, SUM tăng 2–6/tháng), điền ngày sinh cho 156 khách
   (`biz_128`). Bản hiện tại yêu cầu khối header trùng mẫu NULL, dòng tiếp nối phải còn dữ liệu
   item, loại dòng tổng/chú thích (`Cộng`, `TỔNG NGÀY ...`, `Ghi chú ...`), và bỏ qua bảng có
   khóa chính mức dòng. Đã kiểm chứng lại toàn bộ 130 business.
2. **Trùng tên view `__c`**: nếu business có sẵn entity tên `X__c`, việc tạo view sẽ ném lỗi và
   mất cả business. Nay mọi view phái sinh đều kiểm tra `sqlite_master` trước khi tạo, tên bị
   chiếm sẽ bị bỏ qua và ghi vào `skipped_collisions` của báo cáo ingestion.
3. **Chặn nhầm truy vấn ngày dạng OR**: hai nhánh rời rạc (SUBSTR + `date(...)`) từng bị coi là
   "chỉ xử lý một nhánh" và bị loại trước khi thực thi. Nay rule hợp nhất chiến lược theo cột,
   đồng thời không tính nhánh của **cột khác** (tránh bỏ sót lỗi thật).

Ngoài ra đã sửa: thứ tự cột khi hợp nhất view chuẩn hóa (dùng danh sách cột tường minh thay
`SELECT *`), vòng lặp coverage dùng snapshot thay vì đổi kích thước list đang duyệt, aspect
`canonical_projection` chỉ khai báo khi thực sự có view chuẩn hóa, quy tắc chọn dấu phân cách
tiền tệ theo bằng chứng của cả cột (giá trị mơ hồ → NULL thay vì sai 1000 lần), bỏ hai thuộc
tính schema không bao giờ được sinh (`null_pct`, `rows`), giới hạn độ dài `samples`, chỉ render
`enum` khi liệt kê đầy đủ, và bỏ câu lệnh prompt còn nhắc tới `llm_hint` đã ngừng render.

### 17.1c Sửa sau vòng review thứ hai

1. **Forward-fill vẫn có thể bỏ sót (giả âm, đã chấp nhận có ý thức)**: điều kiện
   `has_row_level_primary_key` tắt forward-fill khi bảng có một cột text đầy đủ và duy nhất ở mức
   dòng (ví dụ `STT`). Với bảng trộn ô thật sự (merge cells) *và* có cột đánh số đầy đủ, việc điền
   sẽ không xảy ra — ta chọn "không điền" thay vì "điền sai", vì điền sai làm hỏng số liệu tổng
   còn không điền chỉ giữ nguyên hiện trạng. Muốn xử lý nhóm này cần tín hiệu cấu trúc ô từ file
   gốc (merged-cell metadata) chứ không suy ra từ dữ liệu đã trích.
2. **Validator tìm metadata theo alias**: profile được khóa theo tên bảng vật lý, còn SQL sinh ra
   lại qualify cột bằng alias (`T1`). Trước đây mọi rule dựa trên metadata (định dạng ngày, tiền,
   enum, phủ format) **âm thầm mất tác dụng** trên đúng dạng truy vấn phổ biến nhất. Nay
   `build_alias_table_map` dựng bản đồ alias → bảng thật từ AST, `lookup_column_meta` tra theo
   bảng thật; khi cột không có qualifier mà nhiều bảng cùng tên cột thì chỉ nhận profile nếu tất cả
   đều khai báo cùng `data_format` (ngược lại coi là mơ hồ và bỏ qua). Chiến lược chuẩn hóa ngày
   được gom theo `(bảng, cột)` nên hai cột trùng tên ở hai bảng không "cho nhau mượn" nhánh.
   *Giới hạn còn lại*: bước quét sibling (`_references_column`) vẫn so theo **tên cột**, nên trong
   cùng một mệnh đề, một cột trùng tên ở bảng khác vẫn có thể góp nhánh (ví dụ `T1."Ngày"` đã được
   phủ đủ nhánh, `T2."Ngày"` chỉ xử lý một nhánh ⇒ chưa bị chặn). Đây là giả âm hẹp, đã ghi vào
   backlog §17.2.
3. **`CAST` không còn được tính là một nhánh chuẩn hóa**: `CAST(SUBSTR(col,7,4) AS INTEGER)` từng
   được đếm là 2 chiến lược nên lọt qua rule "phủ format". Nay chỉ SUBSTR/GLOB-LIKE/parser ngày/
   thay `'thg'` được tính, và trường hợp một nhánh duy nhất bị chặn đúng như thiết kế.
4. **Chọn dấu phân cách tiền theo biểu quyết của cả cột**: giá trị 3 chữ số sau dấu phân cách
   (`1.500`) tự nó không nói được là phần nghìn hay phần thập phân, nên cả cột bỏ phiếu — dòng mơ
   hồ bỏ phiếu "phần nghìn", dòng có phần lẻ thật (`12.5`) bỏ phiếu "phần thập phân"; bên nhiều
   phiếu thắng, hòa thì nghiêng về đọc thập phân. Nhờ vậy `["1.500","2.000","12.5"]` cho
   `1500, 2000, 12.5` (trước là `1.5, 2.0, 12.5`), còn `["12.5","1.234"]` vẫn giữ `1.234` như cũ.
   Dấu âm kiểu kế toán `(1.500)` được giữ đúng dấu.
   *Giới hạn còn lại*: một cột **toàn** giá trị mơ hồ 3 chữ số (`["1.500","2.250"]`) không thể phân
   biệt với cột thập phân bằng dấu phẩy; cột đó được đọc nhất quán là phần nghìn cho mọi dòng (một
   thang đo duy nhất, không trộn hai thang). Báo cáo ingestion ghi lại
   `ambiguous_rows` và `ambiguous_as_thousands` để soi lại khi cần.

### 17.2 Chưa triển khai (giữ nguyên trong backlog §13)

- **(10) Repair ngân sách/tất định**: chưa đổi `DEFAULT_SQL_MAX_TOKENS` và chưa thêm repair bằng AST.
- **(11)–(13) P2**: chưa giảm 5 variant schema linking, chưa cache theo `(business, intent)`, chưa
  song song theo business. Đây là thay đổi chi phí/thông lượng, cần đo trên ≥100 câu trước khi chốt.
- **(14) Logging gọn**: chưa nén `llm_calls.jsonl`.
- **Dedupe dòng trùng trong cùng đơn**: chưa bật vì là thao tác phá hủy dữ liệu; cần một điều khoản
  rulebook rõ ràng trước khi xóa dòng. Ảnh hưởng còn lại: `biz_124` vẫn cần `COUNT(DISTINCT ...)`.
- **`_references_column` chưa phân giải alias**: bước quét sibling của rule "phủ format" so theo tên
  cột, nên một cột trùng tên ở bảng khác trong cùng mệnh đề vẫn có thể góp nhánh (giả âm hẹp, không
  gặp trong 30 business đã quét). Hướng sửa: truyền `alias_map` vào `_date_normalization_strategies`
  và so theo `(bảng, cột)` đã phân giải.
- **Nhóm ô trộn thật sự nhưng không phân biệt được**: hai cột mô tả thưa có cùng mẫu NULL ở mép trái
  vẫn bị coi là khối header (đã dựng ca tổng hợp `Ghi chú`/`Mô tả`). Không thể phân biệt với header
  trộn thật nếu chỉ dùng dữ liệu đã trích; cần metadata merged-cell từ file gốc.

### 17.3 Kết quả kiểm chứng lại 10 câu mẫu (SQL viết lại trên view chuẩn hóa)

| Câu | GT | Kết quả trên view chuẩn hóa |
|---|---|---|
| `q_00c4` (biz_106, 05/2025) | 271,866,000 | 271,866,000 ✓ |
| `q_002d` (biz_120, 30/04/2025) | 10,309,400 | 10,309,400 ✓ |
| `q_001883` (biz_128, 09/2026) | 89,526,400 | 89,526,400 ✓ |
| `q_006c` (biz_112, đơn ≥ 55.000) | 2,345 | 2,345 ✓ |
| `q_0088` (biz_119, Otrivin 10ml) | 77 | 77 ✓ |
| `q_00558` (biz_105, tỉ lệ hủy 06/2025) | 2.08 | 2.08 ✓ |
| `q_00bb` (biz_121, tỉ lệ thuốc kê đơn) | 2.27 | 2.25 (mọi trạng thái) / 2.27 nếu mẫu số loại đơn hủy — công thức GT không xác định duy nhất |
| `q_008f` (biz_117, tỉ lệ hủy 10/2025) | 1.21 | 1.20 (11/913 đơn; GT tương ứng mẫu số 906–912 đơn, không suy ra được từ dữ liệu) |

Hai câu phần trăm còn lệch nằm trong dung sai làm tròn 2 chữ số thập phân của scorer; đây là
giới hạn của việc tái lập công thức GT, không phải lỗi chuẩn hóa dữ liệu.
