# Chẩn đoán run mới nhất (5 câu train, content/result @ 18:36)

Phạm vi: 5 log trong `content/result/logs/`, đối chiếu với
`tmp/cache_<biz>/{metadata_profile.json,schema_linking_result_q_*.json}` và
ground truth `dataset/train/answers.jsonl`.

## 1. Điểm

| id | type | GT | Nộp | Kết quả | s_i |
| --- | --- | --- | --- | --- | --- |
| q_001883280e83 | number | 89,526,400 | 89,526,400 | đúng | +1 |
| q_00195e1e4049 | number | 2,954,200 | 72,332,700 | sai (24.5×) | -1 |
| q_002d03ebbded | number | 10,309,400 | 10,312,600 | sai (lệch 3,200) | -1 |
| q_00558c1410ef | percent | 2.08 | 0.0 | sai | -1 |
| q_006c26c65d50 | number | 2,345 | 0 | sai | -1 |

PA = (+1 -1 -1 -1 -1) / 5 = **-0.6**. Bốn câu sai đều có thể đúng 100% nếu áp
đúng 4 quy tắc rulebook (mục 2) — trần của tập này là +1.0.

Ba lần chạy cùng 5 câu: PA = **-0.4** (run 10/11), **-0.8** (run 12/11 17:13),
**-0.6** (run 12/11 18:36). Câu 1 đổi kết quả giữa các lần
(`list 30 phần tử` → `number`), câu 5 đổi 3 lần (4,660 / 1,082 / 0). Đây là
rủi ro trực tiếp cho Rule 8 (|PA upload − PA re-run| ≤ 0.03).

## 2. Đáp án đúng đã kiểm chứng trên DB của pipeline

Chạy trực tiếp trên `tmp/cache_*/biz_*.db` (chỉ dùng SQL, không hard-code):

| Câu | Biểu thức đúng | Kết quả |
| --- | --- | --- |
| Q1 | `SUM(Cần trả NCC)` với `SUBSTR(Thời gian,7,4)\|\|'-'\|\|... BETWEEN '2026-09-01' AND '2026-09-07'` | 89,526,400 |
| Q2 | Hoạt chất `= 'Natri clorid'` (exact), cửa sổ 2026-05-01..05-25, loại `Không giao được`/`Đổi trả`, **DISTINCT dòng trùng**, `SUM(Thanh tien)` | 2,954,200 |
| Q3 | `SUM(Thành tiền)` ngày 2025-04-30, loại `hủy đơn`/`Huỷ đơn`/`trả hàng` | 10,309,400 |
| Q4 | `COUNT(DISTINCT Mã hóa đơn)` tháng 06/2025, `Status IN ('hủy','Đã hủy đơn')` / tổng, làm tròn 2 chữ số | 2.08 |
| Q5 | `SUM(CAST(REPLACE(Thành tiền,',','') AS REAL))` theo **đơn**, `>= 55000`, loại `Đã hủy đơn`/`Hoàn trả` | 2,345 |

Bốn quy tắc quyết định: (i) chuẩn hoá 3 kiểu ngày (dd/mm/yyyy, `d thg m yyyy`,
Excel serial 5 chữ số); (ii) từ vựng trạng thái theo từng shop (huỷ ≠ trả hàng);
(iii) tiền: chỉ bỏ dấu phân cách nghìn, **giữ phần thập phân**; (iv) ngưỡng theo
đơn phải gộp theo đơn + bỏ dòng trùng (§8 rulebook).

Nếu thiếu (ii) ở Q3 thì lệch đúng 3,200 (hoá đơn `ORD010377`, trạng thái
`hủy đơn`); thiếu (iv) + DISTINCT ở Q2 thì lệch 85,600.

## 3. Chuỗi nguyên nhân theo câu

### Q1 — đúng lần này nhưng 2/3 lần trước sai
- Cả 3 candidate lần này đều SUM đúng kỳ; cand0/cand2 khớp → majority.
- Rủi ro còn nguyên: cand1 (seed 100) lần này cắt `REPLACE('.','')` nên ra
  `7,556`; ở lần chạy trước cand1 dùng `Thời gian BETWEEN '01/09/2026' AND
  '07/09/2026'` → so sánh **lexicographic** trên dd/mm/yyyy, kéo cả 04/2025,
  10/2025, 06/2025… vào và trả `1,099,353,400` (không có lỗi runtime nào).
- Lần chạy 17:13, candidate đúng (89,526,400) bị **random tie-break** loại; câu
  đúng biến thành `list` 30 phần tử → -1.

### Q2 — schema linking bỏ mất bảng header, nên không thể lọc ngày/trạng thái
- `schema_linking_result_q_00195e1e4049.json`: linked = 14 cột, **không có cột
  nào của `BanHang_HoaDon`** (nơi duy nhất có `Ngày bán` + `Trạng thái`), trong
  khi metadata_profile có 109 cột. Có thêm cột ma `BanHang_ChiTietHoaDon."date"`,
  `BanHang_ChiTietHoaDon."BanHang_HoaDon"`, `BanHang_Thuoc."Sản phẩm"` (không
  tồn tại) và nhiễu `calendar."lunar_leap"`.
- Vì không có cột ngày/trạng thái trong prompt, model buộc phải đoán: cand0
  `no such column: T2.Mã HH`, cand1 `no such column: T1.Thanh tien`, chỉ cand2
  chạy được → tổng **toàn thời gian** (72,332,700). Đây là hệ quả của linking,
  không phải lỗi SQL.
- Hai lỗi semantics phụ: dùng `LIKE '%Natri clorid%'` (rulebook: hoạt chất phải
  khớp đúng, thuốc phối hợp là loại riêng) và bỏ §8 dedupe.

### Q3 — sai 3,200 vì literal trạng thái không tồn tại trong dữ liệu
- Profile ghi đúng enum: `'Thành công'(30,050)`, `'ok'(17,413)`, `'Huỷ đơn'(299)`,
  `'Đã trả hàng'(274)`, `'hủy đơn'(179)`, `'trả hàng'(152)`, `'Đang xử lý'(22)`.
- cand0 lọc `NOT IN ('Huỷ','Trả hàng')`, cand1 `NOT IN ('Đã hủy','Trả hàng')` →
  không khớp `'hủy đơn'` → hoá đơn huỷ 3,200 vẫn được cộng.
- Hai candidate **sai giống nhau** nên "majority 2/3" chốt luôn đáp án sai.

### Q4 — 0.0 từ ba lỗi chồng nhau
- Literal ISO so với cột text dd/mm/yyyy: `"Ngày giao dịch" >= '2025-06-01'` →
  đúng **0 dòng** (đã kiểm: 0 rows), nên phép chia ra `NULL`.
- Sai khái niệm: cả 3 candidate lọc `Status IN ('Hoàn trả','Đổi trả')` — đó là
  *trả hàng*, không phải *huỷ*. Hai biến thể huỷ là `'hủy'` + `'Đã hủy đơn'`.
- Kết quả `(None,)` được **cả 3 candidate** trả về → pipeline gọi là
  `unanimous_3_of_3` rồi format thành `percent 0.0` → -1.

### Q5 — literal trạng thái "cũ" được grounding thành chuẩn
- `grounded_literals` của câu này chứa `{'mention': 'xong', 'canonical_value':
  'xong', 'column': 'So_chi_tiet_ban_hang.Trạng thái', 'score': 1.0}`. Giá trị
  `'xong'` có thật nhưng chỉ ở các partition cũ (01/2026 còn 530 dòng); từ
  02/2026 dữ liệu dùng `'Đã hoàn thành'`. Tháng 7/2026 **không có** `'xong'`.
- Vì thế cand0 `TRIM(Trạng thái)='xong'` → rỗng; retry 2 lần vẫn rỗng; cuối cùng
  format thành `number 0`.
- cand2 lại `BETWEEN '01/07/2026' AND '31/07/2026'` (lexicographic, dd/mm/yyyy)
  → quét cả 01–07/2026, trả list ~7,000 mã đơn (log 982 KB).
- Lần chạy trước, lỗi ở tiền: `REPLACE(REPLACE(Thành tiền,',',''),'.','')` trên
  `'58,000.0000'` → `580000000`, ngưỡng `>= 55000` thành vô hiệu → đếm = toàn bộ
  đơn `'Đã hoàn thành'` (4,660).

## 4. Nguyên nhân hệ thống

| # | Vấn đề | Bằng chứng |
| --- | --- | --- |
| S1 | Schema linker **không nhận `as_of_date`** (`grep as_of_date src/zaic_2026_4/schema_linking/` = 0 kết quả; `pipeline.py:247` chỉ truyền `question`). Trial SQL tự bịa kỳ → literal ngày sai được "ground" và tiêm vào prompt sinh SQL | Q1: `01/01/2026`, `31/01/2026`; Q2: `2026-01-01..2026-01-31` |
| S2 | Linked schema = hợp field của trial SQL (256 token), không phải tập cột cần cho câu hỏi → mất bảng/cột thiết yếu | Q2 mất `BanHang_HoaDon`; linked chỉ 4–14 cột trên 32–109 cột có sẵn |
| S3 | Literal grounding không kiểm tra ngữ nghĩa/thời gian: nhận mọi giá trị có trong DB, kể cả giá trị chỉ tồn tại ở partition khác | `'xong'` score 1.0; `'%2026%'` → `calendar.lunar_year` |
| S3b | `minimal_profile` (dùng cho case_a/c) diễn giải lại enum bằng lời, làm rơi hoặc bịa literal; model tin phần prose hơn phần `Tập giá trị đầy đủ` | biz_112 thiếu `'Đã hoàn thành'`, `'Đang vận chuyển'`; biz_105 viết `'Đã giao hàng'`, `'Đã lấy'` (dữ liệu là `'đã lấy'`, `'Đang giao hàng'`) |
| S4 | Enum trạng thái trôi theo thời gian trong cùng một view | biz_112: `xong` 61,992 + `Đã hoàn thành` 48,427; `cancel` 1,344 + `Đã hủy đơn` 954 |
| S5 | So sánh ngày không có guard: ISO literal trên cột dd/mm/yyyy → rỗng; dd/mm/yyyy `BETWEEN` → lexicographic sai | Q4 (0 dòng), Q1 cand1 (1.1 tỷ), Q5 cand2 (Jan–Jul) |
| S6 | **Silent zero**: rỗng/NULL → `0` hoặc `0.0` thay vì abstain (`result_formatter.py:97-104`, `:122-125`) → -1 thay vì 0 | Q4, Q5 mất 2 điểm chỉ vì lỗi này |
| S7 | Vote theo result-set equality; lỗi tương quan vẫn "đồng thuận"; `(None,)`×3 = unanimous; tie-break random bỏ phiếu đúng (`voting_executor.py:136-162`) | Q3, Q4, Q1 (lần 17:13) |
| S8 | Few-shot dạy làm sạch tiền bằng cách bỏ **cả** `.` và `,` (`prompt_template.py:32`); profile chỉ gắn nhãn `CURRENCY (THOUSAND_SEPARATOR)` dù dữ liệu có 4 chữ số thập phân | Q5 lần trước: ×10^4 |
| S9 | `SQLValidator` chỉ kiểm syntax/rule hình thức, không kiểm tồn tại cột/bảng hay literal có trong enum (`sql_validator.py:124-246`) | `H."Ngày đặt"`, `calendar."Status"`, `T2."Mã HH"` lọt qua static check |
| S10 | Năng suất: 121 s/câu (135.7/169.3/122.7/100.6/76.9) → 500 câu ≈ **16.8 h** > giới hạn 6 h | profile build ~2 phút/business (18:25→18:35 cho 5 biz) |
| S11 | Không tái lập được: cùng code (commit 17:00), cùng seed, hai lần chạy cho SQL/đáp án khác nhau; tie-break dùng RNG riêng | PA -0.8 vs -0.6; Q5: 4,660 → 0 |
| S12 | Log phình: dump toàn bộ result_set (982 KB cho 1 câu) | `content/result/logs/q_006c26c65d50.json` |

## 5. Kế hoạch sửa (theo thứ tự tác động)

P0 — không cần đổi model, chỉ cần quy tắc + guard:
1. Truyền `as_of_date` + kỳ suy diễn vào linker; **không ground literal ngày**
   và loại literal ngày bịa khỏi `<grounded_entities>`.
2. Sinh kèm `sql_expr` cho cột ngày/tiền ngay ở profiling (ví dụ
   `CASE ... END` chuẩn hoá 3 kiểu ngày; `CAST(REPLACE(x,',','') AS REAL)`),
   thay vì bắt model tự đoán từ prose.
3. Chuẩn hoá literal trạng thái theo enum của cột (map `'Huỷ'` → `'Huỷ đơn'`,
   gộp nhóm huỷ = `hủy|huỷ|cancel` + `đơn`, nhóm trả = `trả|hoàn|đổi`); áp dụng
   hậu kiểm cho mọi predicate trạng thái trong SQL cuối.
4. Zero-guard: rỗng/NULL/0 → repair có gợi ý cụ thể (định dạng ngày, enum, đơn
   vị tiền); vẫn rỗng → **abstain**, không phát 0. Vote không được coi
   `(None,)`/rỗng là đồng thuận.
5. Linker phải bổ sung cột "bắt buộc theo ý định": cột ngày của bảng grain, cột
   trạng thái, cột tiền — kể cả khi trial SQL quên.

P1 — chất lượng và vận hành:
6. Vote theo checklist (có predicate kỳ đúng kiểu, status dùng enum thật, shape
   khớp `predicted_type`) rồi mới majority; tie-break tất định theo
   (checklist score, candidate_id).
7. Mở rộng `SQLValidator`: kiểm tồn tại bảng/cột bằng schema sqlglot, kiểm định
   dạng literal ngày theo `DATE_MIXED`, kiểm literal trạng thái ⊆ enum, cảnh báo
   biểu thức tiền trên cột có phần thập phân.
8. Tất định: temperature 0 cho candidate (đa dạng bằng permutation + subset
   rulebook), cache profile bất biến, log kèm hash profile/prompt.
9. Năng suất: gọi 1 lần `n=3` cho 3 candidate, giới hạn repair 1 lượt có gợi ý
   đích danh, mục tiêu ≤ 40 s/câu.
10. Cắt result_set trong log (tối đa N dòng + tổng số dòng).

## 6. Kỳ vọng

Áp P0 (1–5): Q3, Q4, Q5 đúng ngay; Q2 đúng khi linker giữ `BanHang_HoaDon`;
Q1 giữ nguyên → **PA +1.0** trên tập 5 câu này, đồng thời triệt tiêu nhóm lỗi
"im lặng trả 0" và "literal trạng thái bịa" — hai nhóm chiếm 3/4 câu sai.

Nên đưa 5 câu này thành fixture hồi quy với 5 đáp án đã kiểm chứng ở mục 2.
