# Phân tích trace logs `trace_logs_1311/` sau các commit

## 0. Tóm tắt điều hành

### 0.1 Phạm vi và cách phân tích

Báo cáo phân tích snapshot `trace_logs_1311/` gồm 27 file, khoảng 489 MB, trong đó phần lớn dung lượng là `lsh_index.pkl` (441 MB cho 2 business) và SQLite DB (32 MB). Các file được phân tích trực tiếp:

- `cache_biz_124/llm_calls.jsonl`, `cache_biz_128/llm_calls.jsonl`
- `cache_biz_124/profiling_trace.json`, `cache_biz_128/profiling_trace.json`
- `cache_biz_124/{metadata_profile,raw_profile}.json`, `cache_biz_128/{metadata_profile,raw_profile}.json`
- `cache_biz_124/schema_linking_result*.json`, `cache_biz_128/schema_linking_result*.json`
- `cache_biz_124/question_trace_q_00195e1e4049.json`, `cache_biz_128/question_trace_q_001883280e83.json`
- `content/result/logs/q_00195e1e4049.json`, `content/result/logs/q_001883280e83.json`
- `content/result/submission.jsonl`
- Hai DB `cache_biz_124/biz_124.db`, `cache_biz_128/biz_128.db` để đối chiếu ngữ nghĩa và ground truth.

Snapshot chỉ có 2 câu hỏi, lấy từ `dataset/train/questions.jsonl`. Vì vậy các con số định lượng là tín hiệu chẩn đoán, không đủ để kết luận thống kê toàn bộ test set. Tuy nhiên cả hai câu đều có ground truth trong `dataset/train/answers.jsonl`, nên có thể đối chiếu chính xác nguyên nhân sai.

Run được thực hiện sau commit `5d3616f` (`feat(tracing): add full pipeline LLM, prompt, reasoning, and repair trace logging in cache_dir`). Commit này chủ yếu thêm logging; các hành vi lỗi phản ánh trạng thái tích lũy của các commit trước đó, đặc biệt `810be2a`, `f29863d`, `cd82658`, `f75f3dc`, `fb26a76`, `5e7b5a1`.

### 0.2 Kết quả run so với ground truth

| Câu | Business | Ground truth | Kết quả trace | Điểm theo scorer hiện tại |
|---|---|---|---|---|
| `q_001883280e83` | `biz_128` | `number = 89,526,400` | `abstain` | 0 (`explicit_abstain`) |
| `q_00195e1e4049` | `biz_124` | `number = 2,954,200` | `unanswerable` | **-1** (`false_unanswerable`) |
| Tổng snapshot | | | | **-1** |

Ground truth được xác nhận độc lập từ DB:

- `biz_128`: tổng `Cần trả NCC` của `DanhSachNhapHang` trong `2026-09-01..2026-09-07` là `89,526,400`. Đây cũng là đáp án của snapshot `content/result` cũ.
- `biz_124`: tổng `Thanh tien` của các thuốc có `Hoạt chất = 'Natri clorid'`, đơn `Đã giao`/`Đang vận chuyển`, khử các dòng trùng hoàn toàn trong cùng đơn, ngày `2026-05-01..2026-05-25`, là `2,954,200`.

Điểm đáng chú ý: `q_00195` bị gán `unanswerable` dù dữ liệu và schema DB gốc hoàn toàn đủ để trả lời; final schema mà schema linker cấp cho candidate mới là thứ thiếu bảng ngày. Trong scorer, `unanswerable` sai là `-1`; nếu pipeline trả `abstain` thì được `0`. Đây là một lỗi logic có tác động điểm số trực tiếp.

### 0.3 Sáu phát hiện nghiêm trọng nhất

1. **Schema linking không có `as_of_date` và bị neo bởi few-shot hardcode**. Cả 4 trial SQL của `biz_128` đều dùng `01/01/2026..31/01/2026`, trùng khớp few-shot trong `constants.py`. Đây là nguồn gốc của tháng sai.
2. **Grounded literals bị nhiễm ngày từ trial SQL và từ calendar**. 10/13 grounded literals ở cả hai business là ngày; formatter lại ghi “Từ trong câu hỏi”, dù câu hỏi không hề chứa các ngày đó. Prompt buộc model dùng các giá trị này.
3. **Few-shot SQL hardcode schema hư cấu gây contamination**. Ở `biz_124`, output/reasoning của 3 candidate nhắc `Ngày đặt`, `Mã đơn`, `Trạng thái đơn`, `don_hang`, `chi_tiet_ban_hang` hàng chục lần. Candidate 1 bị 6 lỗi `COLUMN_NOT_FOUND`; repair lặp lại y nguyên.
4. **Schema recall bỏ sót bảng chứa ngày**. Final schema của `biz_124` không có `BanHang_HoaDon` (bảng duy nhất có `Ngày bán`), nên candidate kết luận đúng theo schema là “unanswerable”. Đây là lỗi coverage, không phải lỗi model đơn thuần.
5. **Truncation + repair kém hiệu quả**. `biz_124`: 2/3 initial candidate bị cắt ở 2,048 token (tổng 3/6 call bị cắt); 3 repair tốn 166.6 s, 40,837 prompt token, không tạo được SQL hợp lệ; một repair trả `unanswerable` và được chọn làm best.
6. **Voting biến 1 candidate unanswerable thành “consensus”**, trong khi 2 candidate còn lại chỉ thất bại. Kết quả là `unanswerable` thay vì `abstain`, làm mất 1 điểm.

### 0.4 Chỉ số nổi bật

| Chỉ số | `biz_124` + `q_00195` | `biz_128` + `q_001883` |
|---|---:|---:|
| Tổng thời gian xử lý | 529.3 s | 209.3 s |
| `business_load` | 141.4 s (26.7%) | 151.7 s (72.5%) |
| `schema_linking` | 23.6 s | 6.9 s |
| `candidate_generation` | 364.2 s (68.8%) | 50.7 s (24.2%) |
| LLM calls của pipeline câu hỏi | 12 | 7 |
| Prompt token của pipeline câu hỏi | 94,951 | 36,350 |
| Prompt token profiling | 169,841 | 179,784 |
| Tổng prompt token | **264,792** | **216,134** |
| Tổng completion token (gồm profiling) | 13,879 | 4,553 |
| Truncated calls | 3/6 candidate calls | 0 |
| Candidate có evidence | 0/3 | 3/3 |
| Kết quả voting | `unanswerable_consensus` | `abstain_low_agreement` |

Tổng hai câu: **738.6 s**, **480,926 prompt token**, **18,432 completion token**, **0/2 câu đúng đáp án**; 1/2 câu tránh được -1 nhờ `abstain`, câu còn lại bị -1 vì `unanswerable` sai.

## 1. Bối cảnh run và đối chiếu lịch sử

### 1.1 Timeline và artifact

- `biz_128`: LLM call đầu tiên lúc `2026-11-13T04:43:42Z`, cuối `04:46:17Z`; `elapsed_seconds = 209.323`.
- `biz_124`: LLM call đầu tiên lúc `2026-11-13T04:47:21Z`, cuối `04:55:06Z`; `elapsed_seconds = 529.293`.
- Hai business được xử lý tuần tự trong một run; `submission.jsonl` có đúng 2 dòng.
- `profiling_trace.json` của cả hai business được tạo trong run này (không phải cache có sẵn), nên `business_load` bao gồm profiling + build index.

Kích thước artifact chính:

| File | `biz_124` | `biz_128` |
|---|---:|---:|
| `llm_calls.jsonl` | 1.11 MB | 0.86 MB |
| `profiling_trace.json` | 0.72 MB | 0.72 MB |
| `question_trace_*.json` | 0.69 MB | 0.35 MB |
| `schema_linking_result*.json` | 0.13 MB × 2 | 0.06 MB × 2 |
| `metadata_profile.json` | 2.61 MB | 1.30 MB |
| `raw_profile.json` | 3.27 MB | 1.61 MB |
| `lsh_index.pkl` | 298.45 MB | 142.77 MB |
| `biz_*.db` | 22.33 MB | 10.19 MB |

`schema_linking_result.json` và `schema_linking_result_q_*.json` giống hệt nhau (cùng kích thước, cùng nội dung). File generic bị ghi đè theo từng câu; trong run nhiều câu chỉ còn kết quả của câu cuối.

### 1.2 Đối chiếu snapshot lịch sử

Trong repo còn snapshot cũ:

- `content/result/logs/q_001883280e83.json` (mtime 2026-11-12 18:28) trả đúng `89,526,400`, voting `majority_2_of_3`; candidate 0 dùng đúng khoảng `2026-09-01..2026-09-07`.
- `result/logs/q_001883280e83.json` (mtime 2026-11-10) cũng trả đúng `89,526,400`.
- Snapshot mới `trace_logs_1311` trả `abstain`.

Đây là bằng chứng regression trên cùng câu hỏi, cùng business. Không nên quy kết toàn bộ cho commit tracing; nhưng trace mới cho thấy candidate generation đã chuyển từ khoảng ngày đúng sang khoảng ngày bị neo bởi grounded literals/few-shot.

Với `q_00195`, snapshot `content` cũ trả `72,332,700` (sai) và snapshot `result` cũ `abstain`; snapshot mới `unanswerable`. Câu này vốn chưa từng được giải đúng, nhưng kiểu lỗi mới nguy hiểm hơn vì `unanswerable` bị tính `-1`.

`content/result/submission.jsonl` của snapshot mới:

```json
{"id": "q_001883280e83", "type": "abstain"}
{"id": "q_00195e1e4049", "type": "unanswerable", "answer": null}
```

Dòng `q_00195` không có `reason`; scorer hiện tại rơi vào nhánh `false_unanswerable` và cho `-1`.

## 2. Phân tích `llm_calls.jsonl`

### 2.1 Cấu trúc record

Mỗi record có các trường: `stage`, `prompt`, `raw_output`, `sampling_params`, `prompt_tokens`, `completion_tokens`, `finish_reason`, `stop_reason`, `latency_ms`, `metadata`, `call_id`, `timestamp`.

Các stage quan sát được:

- `minimal_profile`: profiling từng cột.
- `schema_linking_init`: 4 variant trial SQL.
- `schema_linking_repair`: repair khi literal mismatch.
- `sql_init`: 3 candidate ban đầu.
- `sql_repair`: repair SQL.

Metadata của `schema_linking_*` có `question_id`, `case_name`, `profile_type`; của `sql_*` có `question_id`, `candidate_id`, `seed`, `turn`, `repair_phase`. Metadata profiling có `col_key`, `table_name`, `col_name`, `business_id`.

### 2.2 Bảng metrics theo stage

`biz_124`:

| Stage | Calls | Prompt tok | Completion tok | Prompt avg | Completion avg | Trunc | Latency (s) | Latency/call (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `minimal_profile` | 109 | 169,841 | 4,781 | 1,558 | 43.9 | 2 | 54.10 | 496.4 |
| `schema_linking_init` | 4 | 13,634 | 477 | 3,408 | 119.2 | 0 | 9.54 | 2,384.2 |
| `schema_linking_repair` | 2 | 13,765 | 321 | 6,882 | 160.5 | 0 | 13.86 | 6,930.0 |
| `sql_init` | 3 | 26,715 | 4,539 | 8,905 | 1,513.0 | 2 | 197.57 | 65,856.6 |
| `sql_repair` | 3 | 40,837 | 3,761 | 13,612 | 1,253.7 | 1 | 166.60 | 55,534.5 |
| **Tổng** | **121** | **264,792** | **13,879** | | | **3** | **441.67** | |

`biz_128`:

| Stage | Calls | Prompt tok | Completion tok | Prompt avg | Completion avg | Trunc | Latency (s) | Latency/call (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `minimal_profile` | 83 | 179,784 | 3,159 | 2,166 | 38.1 | 0 | 49.15 | 592.1 |
| `schema_linking_init` | 4 | 11,273 | 343 | 2,818 | 85.8 | 0 | 6.74 | 1,684.9 |
| `sql_init` | 3 | 25,077 | 1,051 | 8,359 | 350.3 | 0 | 50.65 | 16,882.0 |
| **Tổng** | **90** | **216,134** | **4,553** | | | **0** | **106.54** | |

Nhận xét:

- Prompt token của profiling chiếm **64.2%** (`biz_124`) và **83.2%** (`biz_128`) tổng prompt token của business.
- Trong pipeline câu hỏi, candidate generation chiếm **71.1%** (`biz_124`) và **69.0%** (`biz_128`) prompt token.
- `sql_init` của `biz_124` có completion trung bình **1,513 token/call**; 2/3 call chạm trần 2,048. Đây là dấu hiệu model dùng gần hết ngân sách cho reasoning trước khi viết SQL.
- Schema linking khá rẻ so với candidate generation: 23.4 s so với 364.2 s ở `biz_124`.

### 2.3 Latency bị làm phẳng bởi batch

`latency_ms` trong trace là **latency trung bình theo batch**, không phải latency riêng của từng prompt:

- Profiling `biz_124`: 64 call có cùng `latency_ms = 535.882`, 45 call có cùng `440.19`. Đúng với `batch_size = 64` trong `ProfileLLMSummarizer` và 109 cột → 2 batch 64 + 45.
- Schema linking init `biz_124`: cả 4 call cùng `2384.209`; đây là batch 4 prompt, tổng 9.54 s.
- Schema linking repair `biz_124`: cả 2 call cùng `6930.0`; batch 2, tổng 13.86 s.

Hệ quả: `latency_ms` từng record không dùng được để so sánh prompt/model; chỉ tổng theo stage có ý nghĩa. Trace thiếu `batch_size`, `batch_latency_ms`, `batch_id` dù engine đã có các trường này trước khi bị `make_llm_call_record` bỏ đi.

### 2.4 Truncation và stop reason

- `biz_124`: 3 call `finish_reason = "length"`:
  - `sql_init` candidate 0: 2,048 completion token, output là JSON reasoning bị cắt, không có SQL.
  - `sql_init` candidate 2: 2,048 token, cùng hiện tượng.
  - `sql_repair` candidate 2: 2,048 token, vẫn không có SQL.
- `biz_128`: không có truncation; completion 303–378 token/candidate.
- `stop_reason` không nhất quán kiểu:
  - Phần lớn là `"</output>"`.
  - 2 record `sql_init` của `biz_128` có `stop_reason = "248044"` (token id dạng chuỗi).
  - Các call bị cắt có `stop_reason = None`.
  - `finish_reason` là `"stop"` hoặc `"length"`; không có trường chuẩn hóa.

Đề xuất trace: lưu `finish_reason`, `stop_reason_raw`, `stop_reason_normalized`, `batch_size`, `batch_latency_ms`; map token id về stop string.

### 2.5 Chất lượng metadata

- `schema_linking_*` và `sql_*` không có `business_id`, chỉ có `question_id`; khi phân tích nhiều business phải suy ra từ đường dẫn cache.
- Không có `run_id`; file `llm_calls.jsonl` được append qua các lần chạy. Nếu chạy lại cùng cache, record cũ và mới trộn lẫn, không thể tách run.
- Không có `model_name` / `max_model_len` / `quantization`; không thể so sánh cấu hình giữa các run.
- `sampling_params` có `temperature`, `max_tokens`, `stop`, `seed`, `thinking_token_budget`, `structured_outputs`; đủ cho từng call nhưng không có trong summary.
- Prompt được lưu full; 121 record = 797,935 ký tự prompt cho `biz_124`. Prompt còn xuất hiện lại trong `question_trace` và `schema_linking_result`, gây duplication.

## 3. Profiling traces và metadata

### 3.1 `profiling_trace.json`

| Chỉ số | `biz_124` | `biz_128` |
|---|---:|---:|
| `total_columns` | 109 | 83 |
| `total_prompt_tokens` | 169,841 | 179,784 |
| `total_completion_tokens` | 4,781 | 3,159 |
| `total_latency_ms` | 54,105 | 49,146 |
| Prompt token trung bình | 1,558 | 2,166 |
| Completion token trung bình | 43.9 | 38.1 |
| Prompt ký tự trung bình | 4,541 | 6,245 |
| Output rỗng | 7 | 10 |
| Output bị cắt (`length`) | 2 | 0 |
| Output unique | 103 | 74 |

Batch: 109 = 64 + 45; 83 = 64 + 19. Đây là hai batch LLM, không phải 109/83 call riêng lẻ.

Vấn đề:

- **7/109 (6.4%)** và **10/83 (12.0%)** cột có `raw_output` rỗng, `completion_tokens = 3`, `finish_reason = "stop"`. Nhiều khả năng model dừng ngay tại stop sequence `"\n\n"` trong danh sách stop của profiling. Trace không giữ được khoảng trắng raw vì đã `.strip()`.
- **2 cột bị cắt** ở `biz_124`: `BanHang_HangHoa.Mô tả sản phẩm`, `BanHang_KhachHang.Số điện thoại` (160 token). Minimal profile của các cột này không đầy đủ.
- Các cột quan trọng bị rỗng minimal profile:
  - `biz_124`: `BanHang_ChiTietHoaDon.Chi tiết hàng`, `BanHang_HangHoa.Món`, `BanHang_HoaDon.Tong tien`, `BanHang_HangHoa.Mã HH Liên quan`, `BanHang_HangHoa.Thuộc tính`, `BanHang_LichSuGia.Sản phẩm`, `ton_kho.Hàng hóa`.
  - `biz_128`: `DanhSachNhapHang.Mã NCC`, `DanhSachSanPham.Nhóm hàng(3 Cấp)`, `DanhSachSanPham.Thương hiệu`, `DanhSachSanPham.Mã HH Liên quan`, `DanhSachSanPham.Mẫu ghi chú`, `DanhSachSanPham.Vị trí`, `DanhSachSanPham.Thuộc tính`, `DanhSachSanPham.Mô tả chi tiết`, `DanhSachSanPham.Trọng lượng`, `calendar.weekday`.
- Final schema `biz_124` có 11 cột, trong đó 3 cột bị ảnh hưởng: `Chi tiết hàng` (minimal rỗng), `Món` (minimal rỗng), `Mã HH Liên quan` (`data_format = EMPTY`). Candidate prompt dùng `maximal_profile` nên 2 cột đầu vẫn có mô tả; ảnh hưởng trực tiếp là ở schema linker case_a/c, nơi minimal profile rỗng nên không có ngữ nghĩa để nối bảng.

### 3.2 `metadata_profile.json` và `raw_profile.json`

| Chỉ số | `biz_124` | `biz_128` |
|---|---:|---:|
| Số cột metadata | 109 | 83 |
| Số bảng | 14 | 9 |
| `minimal_profile` trung bình | 136 ký tự | 119 ký tự |
| `maximal_profile` trung bình | 396 ký tự | 366 ký tự |
| `sample_values` tổng | 120,411 | 57,731 |
| `sample_values` lớn nhất/cột | 10,000 | 10,000 |
| `undocumented_joins` | 41 | 15 |
| Kích thước metadata | 2.61 MB | 1.30 MB |
| Kích thước raw profile | 3.27 MB | 1.61 MB |

`sample_values` đã bị đẩy lên tới 10,000 giá trị/cột (`DEFAULT_MAX_SAMPLE_VALUES = 10000`), dù phần lớn cột chỉ cần 50–100 giá trị để mô tả. Đây là nguồn phình metadata và LSH index.

Các cột ngày/tiền quan trọng được profile đúng format:

- `BanHang_HoaDon.Ngày bán`: `DATE_MIXED`, subtype `EXCEL_SERIAL, SLASH_DATE, VN_DATE_ABBR`; hint có hướng dẫn dùng `julianday('1899-12-30')` hoặc `LIKE`.
- `DanhSachNhapHang.Thời gian`: `SLASH_DATE`, format `D/M/YYYY`.
- `BanHang_ChiTietHoaDon.Thanh tien`: `CURRENCY`, subtype `THOUSAND_SEPARATOR`.
- `DanhSachNhapHang.Cần trả NCC`: `CURRENCY`, subtype `THOUSAND_SEPARATOR`; dữ liệu thực tế dùng dấu phẩy `,` (ví dụ `227,262,100`), không phải dấu chấm.

Việc profile đúng format là điểm mạnh, nhưng pipeline không chuyển format đó thành biểu thức SQL deterministic hoặc view chuẩn hóa; model vẫn phải tự xử lý, và đã xử lý sai.

### 3.3 LSH index

| Chỉ số | `biz_124` | `biz_128` |
|---|---:|---:|
| Kích thước pickle | 298.45 MB | 142.77 MB |
| Số giá trị index | 120,377 | 57,700 |
| Số cột có value | 95 | 66 |
| Load time đo được | 3.95 s | 2.09 s |
| Top 10 cột chiếm | 71.5% | 92.7% |

Các cột chiếm nhiều nhất là high-cardinality ID/date/amount: `Mã hóa đơn`, `Mã giao dịch`, `Order ID`, `Giờ tạo đơn`, `Thời gian thu`, `Số tiền thu`, `Tổng tiền`. Những cột này gần như không bao giờ là entity mention trong câu hỏi, nhưng vẫn được đưa vào LSH với 10,000 giá trị/cột, làm index phình 10–20 lần và tăng thời gian build/load.

LSH cũng là nguồn gốc của grounded date noise: `calendar.date` có 1,461 giá trị ngày, mọi mention ngày đều match vào `calendar.date` với score cao, dù bảng `calendar` không nằm trong final schema.

### 3.4 `business_load` chưa được chia nhỏ

| Business | `business_load` | Profiling LLM | Phần còn lại |
|---|---:|---:|---:|
| `biz_124` | 141.4 s | 54.1 s | **87.3 s (61.7%)** |
| `biz_128` | 151.7 s | 49.1 s | **102.6 s (67.6%)** |

Phần còn lại gồm DB ingestion, LSH build, FAISS embedding/index build, load model/embedder. Trace hiện chỉ có một con số `business_load`, không có sub-stage. Không thể biết đâu là nút thắt chính (LSH build 120k MinHash hay FAISS embedding 190 doc) nếu không đo lại.

### 3.5 Cache và khả năng mở rộng

`ProfilingOrchestrator.run_pipeline()` chỉ dùng cache khi **cả ba** `metadata_profile.json`, `lsh_index.pkl`, `faiss_index/` đều tồn tại. Nếu thiếu một artifact, nó rebuild toàn bộ, bao gồm gọi lại 109/83 prompt profiling. Không có hash schema/data để invalidate; không reuse metadata độc lập với index.

Ở cấp pipeline, `SubmissionPipeline._business_cache` giữ `BusinessContext` (DB path, metadata, raw profile, LSH index, vector index, linker, voting executor) cho mọi business đã gặp, không có LRU/eviction. Với 128 business × ~150–330 MB cache, bộ nhớ có thể lên hàng chục GB. Đây là rủi ro OOM khi chạy full test set.

## 4. Schema linking traces

### 4.1 Variants và chi phí

| Business | Variants | Calls | Prompt tok | Completion tok | Latency | Retries | Final fields | Harvested | Preserved | Grounded literals |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `biz_124` | a, b, c, e | 6 | 27,399 | 798 | 23.40 s | 2 | 11 | 10 | 4 | 13 |
| `biz_128` | a, b, c, e | 4 | 11,273 | 343 | 6.74 s | 0 | 5 | 3 | 2 | 13 |

`case_d` không xuất hiện trong trace. Code mặc định `cases="abcde"`; trace không ghi lại config `cases`, nên không xác nhận được đây là do CLI `--cases abce` hay do build variant bỏ qua. Đây là lỗi trace reproducibility.

Chi tiết prompt:

- `biz_124`: case_a 4,807 ký tự/1,439 token; case_b 7,376/2,492; case_c 22,142/6,643; case_e 9,254/3,060; SME 1,186 ký tự; rulebook 616 ký tự.
- `biz_128`: case_a 5,291/1,578; case_b 7,221/2,297; case_c 15,889/4,706; case_e 8,543/2,692; SME 494; rulebook 752.

Case_e có SME/rulebook context nhưng **không chứa định nghĩa “tháng này”**. Với `biz_124`, SME nói “Dữ liệu có đến hết 25/05/2026” và rulebook có luật “doanh thu theo hoạt chất ghi đúng như vậy”, nhưng trial SQL vẫn dùng `LIKE '%Natri clorid%'` và tháng 6/2026. Với `biz_128`, rulebook context chỉ có luật nhập hàng, không có luật mốc thời gian; trial SQL dùng tháng 1/2026.

### 4.2 Trial SQL và nguồn gốc tháng sai

`biz_128`, cả 4 variant:

```sql
SELECT SUM(T1."Cần trả NCC") ...
FROM "DanhSachNhapHang" AS T1
WHERE T1."Thời gian" >= '01/01/2026'
  AND T1."Thời gian" <= '31/01/2026'
```

`case_e` thêm `AND T1."Trạng thái" = 'Đã nhập hàng'`, nhưng vẫn giữ tháng 1.

Nguyên nhân:

1. `TaskAlignedSchemaLinker.link_schema()` **không nhận `as_of_date`**. Signature chỉ có `question`, `processed_metadata`, `lsh_indexer`, `vector_indexer`, `raw_profile`, `cases`, `cache_dir`, `question_id`, `trace_enabled`.
2. Prompt của linker không có `<temporal_anchor>`; `'as_of' in prompt` trả `False`, `'temporal'` trả `False`.
3. Few-shot của linker hardcode `2026-01-01..2026-01-31` trong `constants.py` (`FEW_SHOT_SQL_EXAMPLES`). Đây là ngày duy nhất trong prompt gần với “tháng này”, nên model copy nguyên.
4. Rulebook temporal rule (“Tháng này = từ ngày 1 đến hết hôm nay”) không được retrieve vào case_e; query retrieval chỉ trả luật nhập hàng/hoạt chất.

`biz_124`, case_c dùng `2026-06-01..2026-06-30`, ngoài `as_of_date = 2026-05-25` và ngoài vùng dữ liệu; đồng thời bịa cột `BanHang_ChiTietHoaDon.Ngày tháng` (không tồn tại). Case_c có 2 repair; repair thứ nhất tạo SQL chứa chuỗi `<business context>` bị chèn vào giữa mệnh đề, cho thấy repair prompt/parse chưa an toàn.

### 4.3 Final schema coverage

`biz_124` final schema:

```text
BanHang_ChiTietHoaDon."Chi tiết hàng"
BanHang_ChiTietHoaDon."Ma hang"
BanHang_ChiTietHoaDon."Mã giao dịch"
BanHang_ChiTietHoaDon."Thanh tien"
BanHang_ChiTietHoaDon."Đ.vị"
BanHang_HangHoa."Mã HH Liên quan"
BanHang_HangHoa."Món"
BanHang_HangHoa."ma_sp"
BanHang_HangHoa."Đơn vị"
BanHang_Thuoc."Hoạt chất"
BanHang_Thuoc."Mã HH"
```

Thiếu hoàn toàn:

- `BanHang_HoaDon` — bảng duy nhất có `Ngày bán` (và `Order ID`, `Trạng thái`).
- Join path `BanHang_ChiTietHoaDon.Mã giao dịch → BanHang_HoaDon.Order ID`.
- Cột trạng thái đơn để loại hủy/trả.

Candidate 0 đã lập luận đúng theo schema được cấp: không có bảng/cột ngày → `unanswerable`. Đây là lỗi recall của schema linking, không phải lỗi suy luận của model. Final schema còn chứa `BanHang_HangHoa.Mã HH Liên quan` với `data_format = EMPTY`; `preserve_linking_keys` thêm cột này vì tên có token `Mã`, dù cột rỗng 100%. Đây là noise trong schema.

`biz_128` final schema 5 cột là đủ cho câu hỏi (`Cần trả NCC`, `Mã NCC`, `Mã nhập hàng`, `Thời gian`, `Trạng thái`). Vấn đề của `biz_128` không phải coverage mà là literal/date grounding.

### 4.4 Grounded literals: nhiễu và sai nguồn

`biz_124` có 13 grounded literals:

- 3 entry về `Natri clorid` (tốt, nhưng dùng `LIKE` thay vì exact).
- 10 entry ngày thuộc `calendar.date`, trong đó nhiều entry fuzzy:
  - `"2026-06-01" → "2026-06-06"`
  - `"2026-06-30" → "2026-06-05"`
  - `"2026-06-30" → "2026-06-09"`
- Các ngày này không có trong câu hỏi; chúng đến từ trial SQL của linker (`2026-06-01..30`), rồi được LSH fuzzy-match sang nhiều ngày gần nhau trong `calendar`.

`biz_128` có 13 grounded literals:

- 10 entry ngày (`01/01/2026`, `31/01/2026`) thuộc `DanhSachNhapHang.Thời gian`/`DanhSachHoaDon.Thời gian`, gồm các fuzzy mapping:
  - `01/01/2026 → 21/01/2026`, `→ 11/01/2026`
  - `31/01/2026 → 01/01/2026`, `→ 21/01/2026`
- 3 entry trạng thái (`Đã nhập hàng`).

Formatter `format_grounded_entities_xml()` in ra:

```text
- Từ trong câu hỏi: "31/01/2026" -> Cột: DanhSachNhapHang."Thời gian" | Giá trị chuẩn trong DB: "31/01/2026"
```

Câu hỏi thực tế là “tháng này tổng giá trị các phiếu nhập hàng là bao nhiu”, **không có** `31/01/2026`. Đây là false premise. Prompt còn ghi “LƯU Ý BẮT BUỘC: ... BẮT BUỘC sử dụng đúng 'Giá trị chuẩn trong DB'”, nên model bị ép dùng tháng 1.

Cơ chế:

1. `link_schema` harvest literal từ trial SQL (`lits_q`) và query LSH với threshold 0.4.
2. `extract_grounded_literals_from_question` cũng lấy mention từ câu hỏi, nhưng `candidate_columns` chỉ lọc khi `score < 0.75`; các match `calendar.date` score cao vẫn lọt dù không nằm trong schema.
3. Không có trường `source` (`question` vs `trial_sql`); formatter luôn ghi “Từ trong câu hỏi”.
4. Không có bước loại date mention khỏi entity grounding; date nên do temporal resolver xử lý.

Đây là **root cause trực tiếp** của `biz_128` và góp phần lớn vào `biz_124`.

### 4.5 Key preservation và empty columns

`preserve_linking_keys` chỉ thêm key cho các bảng đã được harvest. Nó không mở rộng sang bảng chưa được reference, nên không thể cứu `BanHang_HoaDon`. Nó cũng thêm cột theo heuristic tên (`Mã`, `ID`, `Code`), dẫn đến:

- `BanHang_HangHoa.Mã HH Liên quan` (`EMPTY`, 100% NULL) được preserve.
- `BanHang_ChiTietHoaDon.Đ.vị` (đơn vị, không phải join key) được preserve.
- `biz_128` preserve `Mã NCC`, `Mã nhập hàng` (ít hại nhưng không cần cho SUM).

Đề xuất: chỉ preserve cột nếu `data_format != EMPTY`, `null_count < row_count`, `distinct_count > 1`, và ưu tiên cặp join từ `raw_profile.undocumented_joins`; thêm FK closure 1 hop cho bảng fact.

### 4.6 Few-shot contamination

Hai bộ few-shot đều hardcode schema và ngày:

- `constants.py` (`FEW_SHOT_SQL_EXAMPLES`): bảng `ban_hang`, `hoa_don_chi_tiet`, `khach_hang`; ngày `2026-01-01..31`.
- `prompt_template.py` (`FEW_SHOT_SQL_EXAMPLES_VI`): bảng `don_hang`, `chi_tiet_ban_hang`; cột `Mã đơn`, `Ngày đặt`, `Tổng tiền`, `Trạng thái`, `Mã hàng`, `Tên hàng`, `Số lượng`, `Trạng thái đơn`; ngày `2026-05-01..25`.

Đếm số lần các token này xuất hiện trong output/reasoning của candidate `biz_124`:

| Candidate | `Mã đơn` | `Ngày đặt` | `Trạng thái đơn` | `don_hang` | `chi_tiet_ban_hang` | `Số lượng` |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 48 | 3 | 5 | 3 | 3 |
| 1 | 4 | 14 | 8 | 0 | 0 | 0 |
| 2 | 8 | 12 | 0 | 0 | 0 | 0 |

Candidate 1 final SQL dùng `C1."Mã đơn"`, `C1."Trạng thái đơn"`, `C1."Ngày đặt"` — toàn bộ là cột của few-shot, không tồn tại trong DB. Validator bắt 6 lỗi `COLUMN_NOT_FOUND`, nhưng repair vẫn lặp lại vì few-shot vẫn nằm trong prompt. Đây là contamination trực tiếp, không phải model “tự bịa” ngẫu nhiên.

`biz_128` không bị contamination vì câu hỏi đơn giản và selected schema khớp few-shot format.

Đề xuất: bỏ few-shot schema cụ thể; thay bằng một ví dụ format ngắn với placeholder `<table>`, `<column>`, `<date_start>`, `<date_end>`, hoặc retrieve few-shot từ chính business/schema hiện tại. Không hardcode ngày/tên bảng/cột.

## 5. Candidate generation traces

### 5.1 `biz_124` + `q_00195`: chuỗi thất bại

Stage timings:

```text
business_load        141.4106 s
schema_linking        23.6368 s
candidate_generation 364.2453 s
voting                 0.0000 s
formatting             0.0000 s
total                529.2927 s
```

LLM usage:

```text
schema_linking:       6 calls, 27,399 prompt tok,   798 completion tok,  23.4 s
candidate_generation: 6 calls, 67,552 prompt tok, 8,300 completion tok, 364.2 s, 3 truncated
```

Ba candidate:

| Candidate | Initial | Attempts | Kết quả | Stop reason |
|---|---|---|---|---|
| 0 | JSON reasoning 6,415 ký tự, 2,048 token, bị cắt | 2 | `unanswerable`, `final_sql = ""`, `deliberate_abstain = True` | `deliberate_abstain` |
| 1 | SQL 494 ký tự, parse được nhưng sai cột | 2 | `execution_success = False`, 6 lỗi `COLUMN_NOT_FOUND` | `stagnation` |
| 2 | JSON reasoning 6,043 ký tự, 2,048 token, bị cắt | 2 | `execution_success = False`, SQL vẫn là raw JSON | `stagnation` |

Chi tiết candidate 0:

- Initial output là JSON `{"reasoning": "1. Xác định thời gian...` bị cắt ở 2,048 token; không có `"sql"`.
- Repair prompt 47,802 ký tự / 15,047 token; model trả `{"sql": "", "type": "unanswerable"}`.
- `rank()` chọn attempt 1 vì `executed=True`, `shape_ok=True`, dù `final_sql` rỗng và không có evidence.
- `REPAIR_CONTRACT_TEXT` có dòng: “Nếu dữ liệu phát hành không đủ để trả lời, trả `{"sql": "", "type": "unanswerable"}`.” Đây là lời mời bỏ cuộc khi gặp lỗi schema, và model đã tận dụng.

Chi tiết candidate 1:

- Initial SQL dùng `C1."Mã đơn"`, `C1."Trạng thái đơn"`, `C1."Ngày đặt"` — cột few-shot.
- Validator trả 6 lỗi `COLUMN_NOT_FOUND` với hint liệt kê cột thật; repair trả lại y nguyên SQL (canonical stagnation), tốn 19.4 s và 10,620 prompt token.
- Đây là minh chứng repair không sửa được lỗi do prompt vẫn chứa schema hư cấu.

Chi tiết candidate 2:

- Initial 2,048 token reasoning, bị cắt; repair 2,048 token, vẫn bị cắt; `final_sql` là raw JSON 6,043 ký tự.
- Tổng thời gian candidate 2: 87.7 s + 90.5 s = 178.2 s, không tạo ra SQL.

Tổng kết: 6 LLM call, 67,552 prompt token, 8,300 completion token, 364 s, **0 SQL hợp lệ**. Repair chiếm 166.6 s / 40,837 prompt token và không thành công.

### 5.2 `biz_128` + `q_001883`: 3 candidate đều chạy nhưng sai ngữ nghĩa

Stage timings:

```text
business_load        151.7187 s
schema_linking         6.9306 s
candidate_generation  50.6732 s
voting                 0.0001 s
formatting             0.0001 s
total                209.3227 s
```

Ba candidate đều `execution_success = True`, `has_evidence = True`, nhưng kết quả khác nhau:

| Candidate | Khoảng ngày trong SQL | Kết quả | Đúng? |
|---|---|---:|---|
| 0 | `01/01/2026..01/09/2027`, so sánh chuỗi DD/MM/YYYY | 365,216,100 | Không |
| 1 | `2026-01-01..2026-09-07` | 2,607,075,100 | Không |
| 2 | `2026-01-01..2026-01-31`, chỉ `REPLACE('.', '')` | 1,181 | Không |
| Ground truth | `2026-09-01..2026-09-07` | **89,526,400** | |

Nguyên nhân từng candidate:

- Candidate 0: dùng `SUBSTR(...,1,2)||'/'||...` tạo DD/MM/YYYY rồi so sánh `BETWEEN '01/01/2026' AND '01/09/2027'`. So sánh lexicographic trên DD/MM/YYYY không hợp lệ; khoảng kéo dài 18 tháng. Validator không chặn vì thấy `SUBSTR` (date normalization node) và không kiểm tra format output/literal.
- Candidate 1: chuyển về ISO đúng cú pháp, nhưng lấy `2026-01-01..2026-09-07` (year-to-date) thay vì tháng 9. Đây là ảnh hưởng của grounded January + as_of.
- Candidate 2: chuyển về ISO đúng, nhưng chỉ tháng 1; đồng thời `REPLACE(T1."Cần trả NCC", '.', '')` không xóa dấu phẩy. Dữ liệu là `227,262,100` → `CAST` chỉ đọc phần đầu `227`; tổng thành `1,181`. Đây là lỗi parse tiền tệ nghiêm trọng, validator không chặn.

`MONEY_SCALE_RISK` hiện chỉ warning và chỉ kích hoạt khi REPLACE cả `.` và `,`; trường hợp thiếu dấu phẩy không bị cảnh báo. `DATE_FORMAT_MISMATCH` không bắt được range 2027 hoặc range year-to-date.

### 5.3 Truncation và ngân sách reasoning

- `DEFAULT_SQL_MAX_TOKENS = 2048`, `DEFAULT_THINKING_TOKEN_BUDGET = 1536`.
- Prompt yêu cầu CoT gồm: kiểm tra schema/samples, chuẩn hóa ngày, làm sạch tiền, join/filter, tie-breaking. Model dùng gần hết 2,048 token cho reasoning, không còn chỗ cho JSON `sql`.
- `biz_124`: completion của `sql_init` là 2,048 / 443 / 2,048; candidate 0 và 2 bị cắt.
- `biz_128`: completion 303–378; không cắt vì câu hỏi đơn giản hơn.

Đề xuất: giảm reasoning budget xuống 384–512 token; đặt `max_tokens` 1,536–2,048 nhưng dành ít nhất 512 token cho SQL; hoặc tách 2 pha (plan ngắn → SQL), và retry riêng khi `finish_reason == "length"` với prompt “chỉ xuất SQL”. Không dùng repair full-prefix cho lỗi truncation.

### 5.4 Parser và validation gaps

Parser `parse_structured_sql_output_with_reasoning()`:

- Nếu JSON bị cắt (không kết thúc `}`), `json.loads` fail; hàm fallback `clean_sql_string(raw_output)` và trả raw reasoning như SQL.
- Không emit `OUTPUT_UNPARSEABLE`/`OUTPUT_TRUNCATED` nếu chuỗi fallback không rỗng.
- Vì vậy `initial_sql` của candidate 0 là 6,415 ký tự JSON reasoning, candidate 2 là 6,043 ký tự; trace trông như “SQL” nhưng thực chất là reasoning.
- Candidate 0 có `syntax_valid = True` vì `final_sql` rỗng và được xem là deliberate abstain; điều này che giấu parse failure.
- `initial_reasoning = 0` cho candidate 0/2 dù raw output có reasoning, vì JSON không parse được; `extract_thinking_trace` chỉ tìm `think` tag.

Đề xuất:

- Trả `sql = ""` và issue `OUTPUT_UNPARSEABLE`/`OUTPUT_TRUNCATED` khi JSON không hợp lệ; không fallback raw text thành SQL.
- Thêm `parse_status` (`ok`, `truncated`, `unparseable`, `empty`), `parse_error`, `raw_output` tách khỏi `sql`.
- Parse `reasoning` field ngay cả khi không có `think` tag; lưu `reasoning` và `thinking` riêng.
- `syntax_valid = False` nếu SQL rỗng nhưng type không phải `unanswerable`/`abstain`.

### 5.5 Prompt composition

Candidate prompt `biz_128` dài 26,777 ký tự / 8,359 token:

| Thành phần | Ký tự | Tỷ lệ |
|---|---:|---:|
| `<split_rulebook>` | 14,681 | 54.8% |
| System + hướng dẫn + tail | 5,229 | 19.5% |
| `<few_shot_examples>` | 3,377 | 12.6% |
| `<grounded_entities>` | 1,763 | 6.6% |
| `<selected_schema>` | 1,579 | 5.9% |
| `<temporal_anchor>` | 60 | 0.2% |
| `<user_question>` | 88 | 0.3% |

`biz_124` tương tự: 28,386 ký tự / 8,905 token; `rulebook_xml` 14,680 ký tự.

Rulebook đầy đủ chiếm hơn nửa prompt dù mode là `hybrid`; `format_rulebook_context()` trong hybrid trả **full rulebook + focal sections**, và focal sections có thể rỗng. Đây là prompt bloat và distraction lớn. Với 1,000 câu, riêng rulebook tiêu tốn hàng chục triệu prompt token.

Đề xuất: dùng `retrieval` top-k 3–5 clause (1–2k ký tự), hoặc rulebook nén theo checklist; cache prefix; loại few-shot schema cụ thể. Mục tiêu giảm 30–45% prompt token.

### 5.6 Throughput và batching

- Candidate generation gọi tuần tự 3 candidate (`generate_all_candidates` loop `generate_candidate`), mỗi candidate một `llm_engine.generate(prompt)`.
- Schema linking đã batch 4 prompt; profiling batch 64; candidate thì không.
- `biz_124` 3 initial call mất 197.6 s; nếu batch 3 prompt cùng lúc, thời gian gần với max thay vì sum, có thể giảm ~40–50%.
- Repair 3 call mất 166.6 s; cũng có thể batch.
- Throughput ước tính ~22 token/s từ 2,048 token / 89 s; giảm reasoning budget là cách giảm latency mạnh nhất.

Đề xuất: thêm API batch cho candidate (mỗi request có `SamplingParams` riêng nếu cần seed), batch 3 initial và 3 repair; bật prefix caching (cùng prefix system+rulebook+few-shot); giảm `max_model_len` từ 65,536 xuống 16,384–24,576 để tiết kiệm KV cache.

## 6. Voting và formatting

### 6.1 `biz_128`: abstain do 3-way split

`candidate_signatures`:

```text
cand 0: number, 365216100
cand 1: number, 2607075100
cand 2: number, 1181
```

`VotingExecutor.vote()`:

- `evidence_cands = [0, 1, 2]`.
- `groups`: 3 nhóm, mỗi nhóm 1.
- `agreement_ratio = 1/3 = 0.333`, `min_agreement = 0.5`.
- Kết quả: `abstain_low_agreement`, `winning_candidate_id = None`, `final_answer = None`.

Đánh giá: quyết định abstain là **hợp lý theo scoring** vì cả 3 đáp án đều sai; chọn bừa sẽ bị `-1`. Tuy nhiên đây là dấu hiệu candidate generation thất bại, không phải voting thông minh. Cần verifier/validator để chọn hoặc sửa candidate, không chỉ đếm agreement.

### 6.2 `biz_124`: unanswerable giả

`VotingExecutor.vote()`:

```python
unans_cands = [c for c in candidates if c.deliberate_abstain or (...)]
evidence_cands = [...]
if unans_cands and not evidence_cands:
    winner = unans_cands[0]
    return winner, "unanswerable_consensus", "unanswerable: Candidate consensus or deliberate abstain..."
```

Thực tế:

- `unans_cands = [candidate 0]` (1/3).
- `evidence_cands = []` vì candidate 1/2 execution fail.
- Hàm trả `unanswerable_consensus` dù chỉ 1/3 candidate unanswerable; 2/3 còn lại là lỗi kỹ thuật, không phải “consensus”.
- `abstain` candidates cũng bị gộp vào `unans_cands` và bị biến thành `unanswerable`.

Pipeline sau đó trả `type = "unanswerable"`. Theo scorer, `false_unanswerable` là `-1`; nếu trả `abstain` thì `0`. Đây là lỗi logic có tác động điểm số trực tiếp.

Đề xuất:

- Tách `unanswerable` và `abstain`.
- Chỉ trả `unanswerable` khi:
  - majority candidate (>= 2/3) thực sự unanswerable, **hoặc**
  - schema coverage validator chứng minh thiếu bảng/cột/dữ liệu, kèm `reason`.
- Nếu chỉ một candidate unanswerable và các candidate khác fail kỹ thuật → trả `abstain`, không `unanswerable`.
- Không map `abstain` candidate thành `unanswerable`.
- Không chọn unanswerable attempt làm best nếu target type ban đầu là answerable; thêm `has_sql`/`has_evidence` vào rank trước `executed`.

### 6.3 Scoring impact

| Prediction | Scorer verdict | Score |
|---|---|---:|
| `q_001883`: `abstain` | `explicit_abstain` | 0 |
| `q_00195`: `unanswerable` (không reason) | `false_unanswerable` | **-1** |
| Nếu `q_00195`: `abstain` | `explicit_abstain` | 0 |
| `q_001883`: đáp án đúng | `correct` | +1 |
| `q_00195`: đáp án đúng | `correct` | +1 |

Trên 2 câu, snapshot hiện tại `-1`; nếu sửa voting để trả `abstain` cho `q_00195`, thành `0`; nếu sửa candidate generation để trả đúng cả hai, thành `+2`. Đây là khoảng cải thiện 3 điểm trên mẫu nhỏ, nhưng là tín hiệu mạnh về logic.

## 7. Chất lượng trace/logging

### 7.1 Điểm mạnh

- Trace mới capture được full prompt, raw output, reasoning, thinking, repair turns, attempt history, validation issues, result preview, stage timings, LLM usage.
- `question_trace` có `schema_linking_trace`, `candidate_signatures`, `formatting_trace`; đủ để tái hiện chuỗi quyết định.
- `profiling_trace` có prompt/raw output/token/latency từng cột.
- `llm_calls.jsonl` cho phép phân tích theo stage.
- Không có trace này, các root cause về few-shot, grounded date, truncation, false unanswerable sẽ rất khó thấy.

### 7.2 Thiếu trường quan trọng

- Không có `run_id`, `run_started_at`, `config` (model, `cases`, `rulebook_mode`, seeds, `max_tokens`, `thinking_token_budget`, `temperature`, `max_model_len`, `trace_enabled`).
- Không có `batch_size`, `batch_latency_ms`, `batch_id`; `latency_ms` bị làm phẳng.
- Không có `cache_hit`/`cache_miss`, sub-stage của `business_load` (ingest, profiling LLM, LSH build/load, FAISS build/load).
- Không có `parse_status`/`parse_error`; `initial_sql` bị lẫn raw reasoning.
- Không có `source` cho grounded literals (`question` vs `trial_sql` vs `calendar`).
- Không có `schema_coverage`/`constraint_coverage` để biết final schema có đủ trả lời không.
- Không có `prompt_hash`; prompt full bị lặp.
- `stop_reason` không chuẩn hóa; `finish_reason` không có trong summary.
- `raw_output` rỗng bị mất khoảng trắng do `.strip()`; không biết model đã dừng vì stop sequence nào.

### 7.3 Duplication và dung lượng

- `schema_linking_result.json` và `schema_linking_result_q_*.json` giống hệt; file generic bị ghi đè mỗi câu.
- Prompt xuất hiện 3 lần: `question_trace` (`initial_prompt` + `attempts_history[*].prompt`), `llm_calls.jsonl`, `schema_linking_result`.
- `question_trace` 0.35–0.69 MB/câu; `llm_calls` 0.86–1.11 MB/business; `profiling_trace` 0.72 MB/business; metadata/raw 1.3–3.3 MB/business.
- Với 1,000 câu, log có thể lên hàng GB; chưa tính LSH 30+ GB.
- Đề xuất: `trace_level` (`summary`, `full`, `sampled`), gzip, prompt hash + lazy prompt store, chỉ lưu full prompt cho call lỗi/bất thường; bỏ file schema generic; thêm `run_id` và rotate file theo run.

### 7.4 Đề xuất schema trace tối thiểu

```json
{
  "run_id": "...",
  "business_id": "...",
  "question_id": "...",
  "stage": "sql_init",
  "attempt": 0,
  "prompt_hash": "...",
  "prompt_tokens": 8905,
  "completion_tokens": 2048,
  "finish_reason": "length",
  "stop_reason_raw": null,
  "stop_reason_normalized": null,
  "batch_size": 3,
  "batch_latency_ms": 197570,
  "parse_status": "truncated",
  "parse_error": "json_decode_error",
  "sql": "",
  "target_type": "number",
  "validation_issues": ["OUTPUT_TRUNCATED"],
  "cache_hit": false
}
```

## 8. Danh mục điểm yếu và đề xuất tối ưu

### 8.1 Bảng ưu tiên

| ID | Mức | Điểm yếu | Bằng chứng | Đề xuất | Tác động kỳ vọng | Effort |
|---|---|---|---|---|---|---|
| P0-1 | Critical | Few-shot hardcode schema/ngày gây contamination | `biz_124` candidate nhắc `Ngày đặt` 48 lần; candidate 1 dùng `Mã đơn`/`Trạng thái đơn`; `biz_128` trial SQL tháng 1 | Bỏ few-shot schema cụ thể; dùng placeholder hoặc retrieve ví dụ từ DB hiện tại; không hardcode ngày/bảng/cột | Loại bỏ 6 lỗi `COLUMN_NOT_FOUND`, giảm prompt 3.4k ký tự, giảm repair | 0.5–1 ngày |
| P0-2 | Critical | Schema linking không có `as_of_date` | `'as_of' not in prompt`; cả 4 trial SQL `01/01/2026..31/01/2026` | Thêm `as_of_date` + `<temporal_anchor>` vào mọi variant; temporal resolver deterministic; retrieve luật “tháng này” | Sửa `q_001883`; tránh tháng sai | 1–2 ngày |
| P0-3 | Critical | Grounded literals nhiễm ngày từ trial SQL/calendar | 10/13 literal là ngày; `calendar.date`; false “Từ trong câu hỏi” | Chỉ ground từ question; chỉ cột trong final schema; loại `calendar.*` khỏi date mention; thêm `source`; exact match cho categorical | Loại bỏ false premise, giảm 1.7k ký tự prompt | 1 ngày |
| P0-4 | Critical | Schema recall bỏ sót bảng ngày | `biz_124` final schema thiếu `BanHang_HoaDon`; candidate unanswerable | Coverage validator: temporal intent → thêm date columns + FK closure; status/enum cho revenue; bỏ cột EMPTY; expand schema nếu thiếu | `q_00195` có thể trả lời | 1–2 ngày |
| P0-5 | Critical | False `unanswerable` từ 1/3 candidate | `unans_cands=[0]`, `evidence_cands=[]` → `unanswerable_consensus` | Majority rule; tách abstain/unanswerable; schema proof; không map abstain → unanswerable | Tránh `-1`, ít nhất `0` | 0.5 ngày |
| P0-6 | Critical | Repair contract mời bỏ cuộc | `REPAIR_CONTRACT_TEXT` cho phép `unanswerable`; candidate 0 dùng | Chỉ cho unanswerable khi thiếu schema/dữ liệu có proof; cấm đổi type do lỗi syntax/column | Giảm false unanswerable | 0.5 ngày |
| P1-1 | High | Truncation do reasoning quá dài | 3/6 call `length`; 2,048 token; 0 SQL | Reasoning budget 384–512; reserve SQL; retry SQL-only khi length; hoặc 2-phase | Giảm latency và failure | 1 ngày |
| P1-2 | High | Currency parse không đầy đủ | `Cần trả NCC` dùng `,`; candidate 2 chỉ xóa `.` → 1,181 thay vì 89,526,400 | Validator blocking: kiểm tra separator theo `currency_subtype`; magnitude sanity; normalized numeric view | Sửa candidate 2 và các lỗi tương tự | 1 ngày |
| P1-3 | High | Date validator yếu | Range 2027; year-to-date; DD/MM lexicographic vẫn pass | Parse date predicate; kiểm tra format + range vs as_of + resolved period; block nếu ngoài kỳ | Sửa candidate 0/1 | 1–2 ngày |
| P1-4 | High | Exact ingredient bị `LIKE` | `Hoạt chất` dùng `LIKE '%Natri clorid%'`; rulebook yêu cầu đúng | Validator/prompt: categorical exact dùng `LOWER(col)=LOWER(val)`; LIKE chỉ khi câu hỏi yêu cầu partial | Đúng `q_00195` | 0.5 ngày |
| P1-5 | High | Dedupe và status rule không được thực thi | `q_00195` cần dedupe + status; DB có 1,188 nhóm dòng trùng | Ingestion/view dedupe; validator status filter; prompt rule | Đúng `q_00195` | 1–2 ngày |
| P1-6 | High | Candidate generation tuần tự | 3 call = 197.6 s; batch schema/profiling đã có | Batch 3 candidate + 3 repair; per-request sampling; prefix caching | Giảm 40–50% latency | 1–2 ngày |
| P1-7 | High | Rulebook full 55% prompt | 14,681/26,777 ký tự | Retrieval top-k hoặc checklist; compress | Giảm 30–45% prompt token | 1 ngày |
| P1-8 | High | Profiling empty/truncated | 7/109, 10/83 rỗng; 2 cắt | Retry empty; bỏ stop `\n\n`; max_tokens 256; fallback deterministic; cache | Tăng chất lượng schema linking | 1 ngày |
| P2-1 | Medium | LSH index 298 MB / 120k values | 10 cột chiếm 71.5%; high-cardinality ID/date | Cap 50–100 values/cột; skip cardinality cao; num_perm 64; FTS5/exact index | Giảm 10–20× dung lượng, load nhanh | 1–2 ngày |
| P2-2 | Medium | `business_load` không chia nhỏ | 87–103 s chưa rõ thành phần | Sub-stage timings: ingest, profiling, LSH, FAISS, load | Tối ưu đúng chỗ | 0.5 ngày |
| P2-3 | Medium | Cache rebuild toàn bộ khi thiếu 1 artifact | `run_pipeline` cache condition | Reuse metadata độc lập; hash schema/data; invalidate theo phần | Giảm 170k prompt token khi partial cache | 1 ngày |
| P2-4 | Medium | `_business_cache` không eviction | Giữ LSH/vector/DB cho mọi business | LRU 1–2 business; group questions theo business; đóng DB | Tránh OOM 128 business | 1 ngày |
| P2-5 | Medium | Parser biến raw JSON thành SQL | `initial_sql` 6,415 ký tự raw reasoning; `syntax_valid=True` | `OUTPUT_UNPARSEABLE`/`TRUNCATED`, `sql=""`, parse_status | Trace đúng, repair đúng | 0.5 ngày |
| P2-6 | Medium | Trace thiếu run/config/batch | Không có `run_id`, batch, cache_hit | Schema trace mới; normalize stop_reason; prompt hash; gzip | Debug/scale | 1 ngày |
| P2-7 | Low | Trace duplication | `schema_linking_result` × 2; prompt × 3 | Bỏ file generic; hash/lazy prompt; sample | Giảm log size | 0.5 ngày |

### 8.2 Lộ trình đề xuất

**Tuần 1 — sửa correctness P0**

- Bỏ few-shot hardcode; thay bằng ví dụ format generic.
- Thêm `as_of_date` + temporal resolver vào schema linking và candidate prompt.
- Lọc grounded literals theo source + schema; loại calendar khỏi date.
- Thêm schema coverage validator + FK closure; sửa key preservation.
- Sửa voting unanswerable/abstain + repair contract + rank.
- Chạy lại đúng 2 câu trong trace, kỳ vọng: `q_001883` đúng `89,526,400`; `q_00195` ít nhất `abstain` hoặc đúng `2,954,200`.

**Tuần 2 — sửa validation và normalization**

- Ingestion/view normalization: ISO date, numeric currency, dedupe.
- Date/currency/status/exact-match validators (blocking).
- Parser `OUTPUT_UNPARSEABLE`/`TRUNCATED`.
- Thêm regression tests cho 2 câu và các case date/currency.

**Tuần 3 — tối ưu latency và scale**

- Batch candidate/repair; giảm reasoning budget; prefix caching; max_model_len 16k–24k.
- Rulebook retrieval; giảm prompt token.
- Profiling retry/fallback; cap sample values; LSH 10–20× nhỏ hơn.
- LRU business cache; group questions theo business; sub-stage timings.

### 8.3 Kỳ vọng định lượng

- Candidate latency `biz_124`: 364 s → khoảng 150–200 s nếu batch + giảm reasoning; `biz_128`: 50.7 s → 30–40 s.
- Prompt token câu hỏi: 94,951 → khoảng 50–60k nhờ bỏ full rulebook/few-shot và giảm repair.
- `business_load`: 141–152 s → 30–60 s khi cache hit; lần đầu có thể giảm 20–30% nhờ LSH/profile cap.
- LSH: 298 MB → 15–30 MB; load 4 s → <1 s; memory 128 business từ hàng chục GB xuống vài GB.
- Tránh false `unanswerable`: ít nhất +1 điểm trên `q_00195`; sửa date/grounding có thể +1 trên `q_001883`.

## 9. Phụ lục

### 9.1 Bằng chứng ground truth

`biz_128`:

```sql
SELECT SUM(CAST(REPLACE(REPLACE(REPLACE("Cần trả NCC", 'đ', ''), '.', ''), ',', '') AS INTEGER))
FROM DanhSachNhapHang
WHERE "Trạng thái" = 'Đã nhập hàng'
  AND (SUBSTR("Thời gian", 7, 4) || '-' || SUBSTR("Thời gian", 4, 2) || '-' || SUBSTR("Thời gian", 1, 2))
      BETWEEN '2026-09-01' AND '2026-09-07';
```

Kết quả: `89,526,400`.

`biz_124`:

- Exact `Hoạt chất = 'Natri clorid'` (không dùng `LIKE`, để loại thuốc phối hợp).
- Trạng thái đơn `Đã giao` hoặc `Đang vận chuyển` (loại `Không giao được`, `Đổi trả`).
- Khử các dòng trùng hoàn toàn trong cùng đơn (rulebook §8).
- Ngày `2026-05-01..2026-05-25`, parse cả 3 format: Excel serial, `DD/MM/YYYY`, `D thg M YYYY`.

Kết quả: `2,954,200`.

### 9.2 Lệnh phân tích đã dùng

Các phân tích dùng `uv run python` với Polars/JSON. Ví dụ:

```bash
UV_CACHE_DIR=$PWD/.uvcache uv run python - <<'PY'
import polars as pl
df = pl.read_ndjson("trace_logs_1311/cache_biz_124/llm_calls.jsonl")
print(df.group_by("stage").agg(
    pl.len().alias("calls"),
    pl.col("prompt_tokens").sum().alias("ptok"),
    pl.col("completion_tokens").sum().alias("ctok"),
    (pl.col("finish_reason") == "length").sum().alias("trunc"),
).sort("stage"))
PY
```

### 9.3 Thuật ngữ

- **Grounded literal**: cặp `mention -> canonical value` được inject vào prompt.
- **Trial SQL**: SQL nháp do schema linker sinh ra để harvest field/literal.
- **Evidence candidate**: candidate execute thành công và có ít nhất một cell non-NULL.
- **Final schema**: tập `table.column` đưa vào candidate prompt.
- **False unanswerable**: pipeline trả `unanswerable` khi dữ liệu thực tế trả lời được.

### 9.4 Mã nguồn liên quan

| Khu vực | File | Vai trò |
|---|---|---|
| Prompt/few-shot | `src/zaic_2026_4/constants.py`, `src/zaic_2026_4/submission/prompt_template.py` | Hardcode few-shot, rulebook context, temporal anchor |
| Schema linking | `src/zaic_2026_4/schema_linking/task_aligned_linker.py`, `src/zaic_2026_4/schema_linking/context_builder.py` | Trial SQL, harvest field/literal, final schema |
| Candidate generation | `src/zaic_2026_4/submission/candidate_generator.py` | Prompt, truncation, repair state machine, rank |
| Validation | `src/zaic_2026_4/submission/sql_validator.py` | AST, date/currency/schema checks |
| Voting | `src/zaic_2026_4/submission/voting_executor.py` | Canonical signature, agreement gate, unanswerable |
| Profiling | `src/zaic_2026_4/profiling/profile_summarizer.py`, `src/zaic_2026_4/profiling/profile_manager.py` | Minimal profile, batch, cache |
| Index | `src/zaic_2026_4/profiling/lsh_indexer.py`, `src/zaic_2026_4/profiling/vector_indexer.py` | LSH/FAISS |
| Trace | `src/zaic_2026_4/common/trace_logger.py`, `src/zaic_2026_4/submission/pipeline.py` | Ghi trace, stage timings |
| Ingestion | `src/zaic_2026_4/ingestion/view_builder.py`, `src/zaic_2026_4/ingestion/sqlite_sink.py` | View hợp nhất, nơi nên thêm normalize/dedupe |

### 9.5 Giới hạn của báo cáo

- Snapshot chỉ có 2 câu; các tỷ lệ phần trăm mang tính chẩn đoán, không đại diện toàn bộ test set.
- Một số kết luận về nguyên nhân (ví dụ empty profiling do stop `\n\n`) là suy luận từ trace + code, chưa chạy lại mô hình để xác nhận.
- So sánh snapshot lịch sử `content/` và `result/` có thể khác version DB/cache; chỉ dùng làm tín hiệu regression, không phải đối chứng tuyệt đối.
- Các đề xuất ưu tiên dựa trên tác động correctness trước, latency/size sau.
