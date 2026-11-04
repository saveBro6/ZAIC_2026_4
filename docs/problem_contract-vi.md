# Bản Hợp Đồng Bài Toán (Problem Contract): DataMate (Zalo AI Challenge 2026)

---

## 1. Tổng Quan & Mục Tiêu Đề Bài

**DataMate** là một bài toán trong khuôn khổ cuộc thi Zalo AI Challenge 2026, yêu cầu xây dựng một hệ thống AI Agent tự động, chạy offline hoàn toàn, có khả năng trả lời các câu hỏi kinh doanh bằng tiếng Việt tự nhiên dựa trên dữ liệu bảng có cấu trúc đa dạng (CSV, Excel, SQLite). Ngữ cảnh dữ liệu gắn liền với hoạt động kinh doanh thực tế của các hộ kinh doanh và doanh nghiệp nhỏ tại Việt Nam (SMBs).

Hệ thống nhận đầu vào gồm các tệp dữ liệu thô của doanh nghiệp, tài liệu mô tả (`README.md`), câu hỏi bằng tiếng Việt tự nhiên, ngày mốc đối chiếu thời gian (`as_of_date`), lịch âm dương (`calendar.csv`) và tập quy tắc nghiệp vụ (`rulebook.md`). Agent phải tính toán ra câu trả lời chính xác tuân thủ định dạng và quy tắc nghiệp vụ, trả về `unanswerable` nếu dữ liệu không đủ căn cứ trả lời, hoặc chủ động `abstain` (từ chối trả lời) để bảo toàn điểm số trước rủi ro bị phạt.

---

## 2. Mô Hình Hóa Bài Toán & Luồng Xử Lý

### 2.1 Bộ Tham Số Đầu Vào (Input)
Với mỗi câu hỏi thứ $i \in \{1, \dots, N\}$, hệ thống nhận một bản ghi:
$$\mathcal{Q}_i = \langle \text{id}_i, \text{business\_id}_i, \text{question}_i, \text{as\_of\_date}_i \rangle$$
Cùng với ngữ cảnh phụ trợ:
1. **Thư mục dữ liệu doanh nghiệp**: `<split>/data/<business_id>/` chứa:
   - Các tệp dữ liệu có cấu trúc: `*.csv`, `*.xlsx`, `*.sqlite`.
   - Tài liệu mô tả: `README.md` (từ điển dữ liệu, ghi chú cấu trúc; chất lượng từ chi tiết đến rất sơ sài).
2. **Quy tắc nghiệp vụ của tập dữ liệu**: `<split>/rulebook.md` quy định cách hiểu mặc định về thuật ngữ, công thức tính doanh thu, giá vốn, quy tắc xếp hạng cho riêng tập (split) đó.
3. **Lịch âm – dương**: `<split>/calendar.csv` ánh xạ ngày dương lịch sang âm lịch và các dịp lễ Tết truyền thống.

### 2.2 Bộ Dữ Liệu Đầu Ra (Output)
Với mỗi câu hỏi $i$, hệ thống phải ghi một dòng JSON vào tệp `submission.jsonl`:
$$\mathcal{A}_i = \langle \text{id}_i, \text{type}_i, \text{answer}_i [, \text{reason}_i] \rangle$$
Trong đó $\text{type}_i \in \{\texttt{"number"}, \texttt{"percent"}, \texttt{"string"}, \texttt{"date"}, \texttt{"boolean"}, \texttt{"list"}, \texttt{"table"}, \texttt{"unanswerable"}, \texttt{"abstain"}\}$.

---

## 3. Quy Cách Dữ Liệu & Cấu Trúc Thư Mục

### 3.1 Cấu Trúc Thư Mục Chuẩn
```text
<split>/
├── data/
│   └── <business_id>/
│       ├── *.csv | *.xlsx | *.sqlite        # Các tệp dữ liệu kinh doanh
│       └── README.md                        # Mô tả doanh nghiệp, schema hoặc ghi chú
├── rulebook.md                              # Quy tắc nghiệp vụ mặc định (THAY ĐỔI THEO TỪNG SPLIT!)
├── calendar.csv                             # Lịch đối chiếu âm - dương (Tết, rằm, mùng một)
├── questions.jsonl                          # Danh sách câu hỏi cần trả lời
└── answers.jsonl                             # Đáp án tham chiếu (chỉ có ở tập Training)
```

### 3.2 Quy Mô & Phân Bổ Tập Dữ Liệu

| Phân đoạn (Split) | Số doanh nghiệp | Số lượng câu hỏi | Đáp án mẫu | Mục đích & Ghi chú |
| :--- | :---: | :---: | :---: | :--- |
| **Training** | 30 | ~3.000 | Có (`answers.jsonl`) | Huấn luyện mô hình, kiểm thử offline, tối ưu prompt/pipeline. |
| **Public Test** | 10–20 | ~1.000 | Không | Vòng 1: Đánh giá trên bảng xếp hạng công khai (tối đa 5 lượt nộp/ngày). |
| **Private Test** | 10–20 | ~500 | Không | Vòng 2: Quyết định thứ hạng chung cuộc. Chứa **loại hình kinh doanh mới** chưa từng xuất hiện. |

### 3.3 Đặc Điểm Dữ Liệu Thực Tế
- **Ngành hàng đa dạng**: Tạp hóa, quán cà phê/trà sữa, nhà hàng/quán ăn, hiệu thuốc, cửa hàng vật liệu xây dựng, tiệm sửa xe máy, cửa hàng mẹ và bé, và các ngành hàng mới ở tập Private test.
- **Thời gian bao phủ**: 12–30 tháng hoạt động thực tế với tính chu kỳ, mùa vụ rõ rệt (Tết, Trung Thu, ngày lễ, ngày nhận lương, cuối tuần).
- **Quy mô bảng**: Từ vài trăm dòng đến khoảng 300.000 dòng mỗi doanh nghiệp.
- **Đa định dạng phức tạp**: Phối hợp nhiều tệp `.csv`, `.xlsx`, `.sqlite` trong cùng một doanh nghiệp. Có nhiều sheet, header tiếng Việt, có dấu/không dấu, format ngày tháng không đồng nhất.
- **Chất lượng `README.md` không đồng đều**: Có doanh nghiệp kèm từ điển dữ liệu rất đầy đủ, có doanh nghiệp chỉ có vài ghi chú sơ sài hoặc gần như không có gì.
- **Câu hỏi thực tế**: Tiếng Việt đời thường, viết tắt, văn phong khẩu ngữ, có hoặc không có dấu thanh.

### 3.4 Cấu Trúc Bản Ghi Chi Tiết

#### `questions.jsonl`
```json
{"id": "q_01234", "business_id": "biz_017", "question": "thang nay hang nao ban cham nhat, ton nhieu ma ban ko dc", "as_of_date": "2026-08-27"}
```
- `id` (chuỗi): Mã định danh câu hỏi.
- `business_id` (chuỗi): Thư mục con tương ứng tại `<split>/data/<business_id>/`.
- `question` (chuỗi): Câu hỏi tự nhiên bằng tiếng Việt.
- `as_of_date` (chuỗi, định dạng `YYYY-MM-DD`): Ngày giả định là "hôm nay" làm mốc giải quyết các mốc thời gian tương đối.

#### `answers.jsonl` (Tập Training)
```json
{"id": "q_06de33fa21cc", "type": "string", "answer": "Xi măng Hà Tiên PCB40 bao 50kg", "alias": ["SP0001"]}
{"id": "q_01241", "type": "unanswerable", "answer": null, "reason": "Dữ liệu không có giá vốn nên không tính được lợi nhuận"}
```
- Trường `alias`: Danh sách tên gọi chấp nhận khác (mã sản phẩm, cách viết khác trong bảng dữ liệu). Tệp nộp bài không cần trường này.

#### `rulebook.md` (Quy định bắt buộc)
- Mỗi split có tệp `rulebook.md` riêng, có thể khác nhau giữa các split.
- **Nguyên tắc động**: Agent phải đọc và áp dụng nội dung `rulebook.md` của chính split đang chạy, **tuyệt đối không được gán cứng (hardcode) quy tắc của tập training**.

---

## 4. Các Loại Câu Trả Lời & Quy Tắc Khớp Đáp Án

Hệ thống chấm điểm so khớp nghiêm ngặt theo kiểu dữ liệu được khai báo:

| Kiểu dữ liệu | Nội dung trả lời | Ví dụ | Quy tắc so khớp (Matching Rule) |
| :--- | :--- | :--- | :--- |
| **`number`** | Giá trị số (tiền VNĐ, số lượng, đếm) | `48250000` | **Khớp chính xác (Exact match)**. Đáp án chuẩn là số nguyên (số tiền, số lượng) hoặc làm tròn 2 chữ số thập phân. |
| **`percent`** | Tỷ lệ phần trăm (tính theo %, không phải số thập phân tỷ lệ) | `-12.45` | **Khớp chính xác sau khi làm tròn 2 chữ số thập phân**. Dấu âm/dương phải tuyệt đối chính xác (ví dụ `-12.45%` ghi là `-12.45`). |
| **`string`** | Tên (sản phẩm, khách hàng, chi nhánh, thứ trong tuần,...) | `"Sting dâu 330ml"` | **Khớp chuỗi chuẩn hóa**. Cả câu trả lời và các dạng được chấp nhận trong ground truth (tên gốc, bí danh `alias`, mã code) đều được chuẩn hóa trước khi so sánh:<br>1. Unicode NFC.<br>2. Chuyển thành chữ thường.<br>3. Ký tự không phải chữ cái, chữ số hoặc khoảng trắng được thay bằng dấu cách.<br>4. Xóa khoảng trắng đầu/cuối, gom khoảng trắng liên tiếp thành 1 dấu cách.<br>*(Lưu ý: Dấu tiếng Việt KHÔNG bị loại bỏ).* |
| **`date`** | Ngày tháng theo chuẩn ISO | `"2026-08-15"` hoặc `"2025-09"` | **Khớp chính xác**. Định dạng `YYYY-MM-DD`. Nếu câu hỏi về tháng thì định dạng `YYYY-MM`. |
| **`boolean`** | Đúng / Sai | `true` hoặc `false` | **Khớp chính xác** (`true` hoặc `false`). |
| **`list`** | Danh sách có thứ tự | `["Cà phê sữa", "Bạc xỉu", "Trà đào"]` | **Khớp chính xác theo từng vị trí (Position-by-position)**. Phải đúng thứ tự câu hỏi yêu cầu (hoặc thứ tự mặc định trong rulebook). Mỗi phần tử khớp theo quy tắc của `string` hoặc `number`. |
| **`table`** | Bảng dữ liệu nhỏ (gồm columns và rows) | `{"columns": ["khach_hang", "so_tien_no"], "rows": [["Chị Lan", 3200000], ["Anh Tú", 1500000]]}` | Cột được so khớp theo tên đã chuẩn hóa (nếu tên cột không khớp thì so theo thứ tự vị trí). Các hàng được so khớp **như một tập hợp không phân biệt thứ tự (set)**. Từng ô dữ liệu so khớp theo quy tắc của `string` hoặc `number`. |
| **`unanswerable`** | Dữ liệu không đủ thông tin để trả lời | `{"type": "unanswerable", "reason": "..."}` | **Đúng khi và chỉ khi đáp án chuẩn là unanswerable**. Nếu cố tình đưa ra đáp án giá trị cho câu hỏi unanswerable sẽ bị tính là sai (`wrong`). Trường `reason` là tùy chọn. |
| **`abstain`** | Chủ động từ chối trả lời | `{"type": "abstain"}` | **0 điểm**. Luôn nhận 0 điểm bất kể đáp án chuẩn là gì. Bỏ sót câu hỏi trong tệp nộp cũng tính là abstain. |

### 4.1 Quy Tắc Xếp Hạng & Xử Lý Đồng Hạng
- **Tiêu chí**: Căn cứ theo câu hỏi; nếu không nêu rõ, áp dụng tiêu chí mặc định trong `rulebook.md` (ví dụ "bán chạy" mặc định theo số lượng bán giảm dần).
- **Phạm vi tính toán**: Chỉ tính trên các đơn hàng hợp lệ (loại bỏ đơn hủy, đơn hoàn trả theo rulebook).
- **Phá vỡ đồng hạng (Tie-breaking)**: Khi các phần tử có giá trị bằng nhau, sắp xếp tăng dần theo mã sản phẩm/mã đối tượng (`code ASC`).
- **Bảo đảm của tập test**: Ban tổ chức cam kết tập test không có trường hợp đồng hạng tại ngưỡng ranh giới thứ $k$ (hoặc hạng 1 đối với câu hỏi dạng "... nhất").
- **Tập ứng viên nhỏ hơn $k$**: Nếu số lượng ứng viên thỏa mãn ít hơn $k$, đáp án chuẩn sẽ chứa toàn bộ các ứng viên hiện có.

---

## 5. Metric Đánh Giá & Cơ Chế Phạt Điểm (Penalized Accuracy)

Chỉ số xếp hạng chính thức trên Leaderboard tính trên $N$ câu hỏi của tập test là **Penalized Accuracy (PA)**:

$$\text{PA} = \frac{1}{N} \sum_{i=1}^{N} s_i$$

Điểm số từng câu $s_i$ được xác định:
$$s_i = \begin{cases}
+1 & \text{khi câu trả lời } i \text{ \textbf{chính xác (correct)}} \\
0 & \text{khi câu trả lời } i \text{ là \textbf{từ chối (abstain)}} \text{ (hoặc không nộp)} \\
-1 & \text{khi câu trả lời } i \text{ \textbf{sai (wrong)}}
\end{cases}$$

### 5.1 Phân Tích Chiến Lược Điểm Số & Quản Trị Rủi Ro
1. **Miền giá trị**: $\text{PA} \in [-1.0, +1.0]$.
2. **Hình phạt đoán mò rất nặng**:
   - Trả lời sai bị trừ $-1$ điểm.
   - Chọn `abstain` được bảo toàn $0$ điểm.
   - **Kỳ vọng toán học**: $\mathbb{E}[s] = P(\text{đúng}) \cdot (+1) + (1 - P(\text{đúng})) \cdot (-1) = 2 \cdot P(\text{đúng}) - 1$.
   - **Ngưỡng hòa vốn**: Hệ thống chỉ nên đưa ra câu trả lời khi ước lượng xác suất đúng $P(\text{đúng}) > 0.50$ (lớn hơn 50%). Khi độ tin cậy thấp hoặc nghi ngờ lỗi schema/code, lựa chọn tối ưu bắt buộc là `abstain`.
3. **Bẫy câu hỏi không thể trả lời (`unanswerable`)**: Cung cấp bất kỳ giá trị nào cho câu hỏi thiếu dữ liệu đều bị phạt $-1$ điểm. Nhận diện đúng `unanswerable` được $+1$ điểm.
4. **Tiêu chí phụ xếp hạng Leaderboard**:
   - Ưu tiên 1: Tổng thời gian suy luận (wall-clock time) thấp hơn trong lần chạy lại của BTC.
   - Ưu tiên 2: Thời điểm nộp bài sớm hơn.

---

## 6. Quy Ước Nghiệp Vụ & Logic Mặc Định Cần Lưu Ý

Nếu câu hỏi không chỉ định công thức riêng, Agent bắt buộc phải tuân theo các quy định mặc định trong `rulebook.md`:

1. **Xử lý mốc thời gian tương đối (`as_of_date`)**:
   - `"tháng này"`: Tính từ ngày 1 của tháng chứa `as_of_date` đến hết ngày `as_of_date` (không tính toàn bộ cả tháng nếu `as_of_date` nằm giữa tháng).
   - `"tuần này"`, `"tuần rồi"`: Tuần chuẩn tính từ **thứ Hai đến Chủ Nhật**.
   - `"cùng kỳ năm ngoái"`: Khoảng thời gian tương ứng đối chiếu với `as_of_date` ở năm liền trước.
2. **Quy tắc tính Doanh thu**:
   - Bằng tổng thành tiền từng dòng sản phẩm sau khi trừ chiết khấu cấp dòng hàng (line-level discount).
   - **Không trừ voucher/giảm giá cấp đơn hàng**.
   - **Không bao gồm thuế VAT và phí vận chuyển**.
   - **Loại trừ toàn bộ đơn hàng bị hủy** và đơn hàng có chứa bất kỳ mặt hàng nào bị trả lại.
3. **Mức độ tiêu thụ ("bán chạy")**:
   - Tiêu chí mặc định là **tổng số lượng sản phẩm bán ra** (không phải tổng tiền doanh thu hay số lần xuất hiện trong đơn).
4. **Phương pháp tính giá vốn**:
   - Giá vốn hàng tồn kho mặc định tính theo **giá vốn bình quân gia quyền theo lô nhập hàng**.
5. **Lịch âm và các dịp lễ**:
   - Khoảng thời gian Tết: Tính từ **ngày 23 tháng Chạp** đến hết **mùng 10 tháng Giêng** (theo lịch âm).
   - Mọi mốc âm lịch phải tra cứu từ tệp `<split>/calendar.csv`.

---

## 7. Ràng Buộc Kỹ Thuật, Hạ Tầng & Giới Hạn Nghiêm Ngặt

Các giải pháp nộp bài sẽ được kiểm thử và chấm điểm tự động bên trong container Docker độc lập với các thông số:

### 7.1 Cấu Hình Phần Cứng & Giới Hạn Thời Gian
- **GPU**: $2 \times$ NVIDIA RTX 5090 (32 GB VRAM/GPU $\rightarrow$ tổng cộng 64 GB VRAM). Kiến trúc Blackwell yêu cầu CUDA $\ge 12.8$ và bản build tương thích PyTorch/vLLM.
- **RAM**: 512 GB System RAM.
- **Giới hạn thời gian (Wall-clock time)**: Toàn bộ quá trình suy luận trên tập Private test (~500 câu hỏi thuộc 10–20 doanh nghiệp) **không được vượt quá 6 giờ**, tính cả thời gian khởi động container và nạp trọng số mô hình.
  - Thời gian trung bình tối đa cho mỗi câu hỏi: $\sim 40$ giây.

### 7.2 Giới Hạn Tham Số Mô Hình (Cực Kỳ Quan Trọng)
- **Tổng số lượng tham số toàn hệ thống**: $\le \mathbf{15\text{ Tỷ}}$ tham số ($15\text{B}$).
  - Áp dụng cho **tất cả** các mô hình sử dụng trong quá trình suy luận cộng lại (Language Models, Embedding models, Rerankers, Classifiers,...).
  - Tính tổng tham số của tất cả mô hình, bất kể chúng được nạp đồng thời hay chạy tuần tự.
  - Với mô hình Mixture-of-Experts (MoE), **tính tổng toàn bộ tham số** của mô hình (sparse + router), không chỉ tính số tham số kích hoạt (active parameters).
- **Lượng tử hóa (Quantization)**: Được phép sử dụng mọi hình thức lượng tử hóa (INT4, FP4, AWQ, GPTQ, GGUF,...).
- **Nguồn gốc mô hình**: Trọng số phải được công bố công khai trước ngày **01/10/2026** với giấy phép hợp lệ.
- **Khai báo**: Mọi mô hình sử dụng phải được liệt kê minh bạch trong `README.md` của giải pháp.

### 7.3 Môi Trường Cách Ly & Tính Toàn Vẹn
- **Không có kết nối mạng (`--net=none`)**: Hệ thống chạy hoàn toàn offline bên trong Docker. Tuyệt đối không gọi API bên ngoài, không tải dữ liệu qua mạng.
- **Dữ liệu mount chỉ đọc (`ro`)**: Thư mục `/data` được gắn dạng read-only. File tạm, index, cache phải ghi vào `/tmp` hoặc `/result`.
- **Không can thiệp thủ công & Cấm gian lận**:
  - Nghiêm cấm gán cứng đáp án (hard-coding) hoặc viết logic riêng rẽ theo từng mã câu hỏi (`id`) hoặc mã doanh nghiệp (`business_id`).
  - Không được dùng tập Public hoặc Private test để huấn luyện hay thu thập đáp án dưới mọi hình thức.
- **Thực thi mã lệnh (Code Execution)**: Agent được phép và khuyến khích sinh mã và thực thi mã (Python, SQLite, Polars/Pandas) ngay trong container để tính toán.

---

## 8. Quy Cách Docker & Giao Diện Xuất Kết Quả

### 8.1 Lệnh Thực Thi Của Ban Tổ Chức
```bash
docker run --gpus all --net none \
  -v /path/to/test_data:/data:ro \
  -v /path/to/result_dir:/result:rw \
  <image_name> /bin/bash /app/predict.sh
```

### 8.2 Cấu Trúc Giao Tiếp Tệp
1. `/data`: Thư mục dữ liệu đầu vào (read-only) gồm:
   - `/data/data/<business_id>/*`
   - `/data/rulebook.md`
   - `/data/calendar.csv`
   - `/data/questions.jsonl`
2. `/result/submission.jsonl`: Tệp kết quả xuất ra dạng JSONL.
3. `/result/logs/`: Thư mục lưu log chi tiết từng câu hỏi. Mỗi câu hỏi cần log:
   - ID câu hỏi.
   - Kết quả dự đoán và kiểu dữ liệu.
   - Thời gian xử lý.
   - Dấu vết suy luận (trace, mã code được sinh, kết quả truy vấn SQL/Python, thông báo lỗi nếu có).

### 8.3 Định Dạng Tệp Nộp Bài (`/result/submission.jsonl`)
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

### 8.4 Kiểm Tra Tính Tái Lập (Reproducibility Contract)
- Điểm chính thức tính dựa trên lượt chạy lại (re-run) của BTC đối với Docker image đã nộp.
- **Tiêu chuẩn tái lập**:
  $$|\text{PA}_{\text{đã nộp}} - \text{PA}_{\text{chạy lại}}| \le 0.03$$
- Nếu độ chênh lệch vượt quá $0.03$, BTC sẽ đối chiếu logs trong `/result/logs/`. Nếu không chứng minh được tính nhất quán, giải pháp sẽ bị hủy kết quả.
- Nếu container bị ngắt giữa chừng do quá 6 giờ hoặc crash, BTC chỉ chấm các câu đã kịp ghi vào `/result/submission.jsonl`; các câu còn lại tính là `abstain`.

---

## 9. Lịch Trình Cuộc Thi & Các Mốc Thời Gian Trọng Yếu

| Giai đoạn / Sự kiện | Mốc thời gian (GMT+7) | Hành động & Yêu cầu |
| :--- | :--- | :--- |
| **Vòng Public Test** | 26/10 – 16/11/2026 | Phát triển giải pháp, chạy thử nghiệm trên tập Public test, nộp file kết quả JSONL (tối đa 5 lượt/ngày). |
| **Đóng nộp Docker giải pháp** | **23:59 ngày 16/11/2026** | Nộp giải pháp hoàn chỉnh (mã nguồn train/infer, weights, `predict.sh`) đóng gói Docker qua link Google Drive công khai + mã checksum SHA256. Tối đa 2 lần nộp (chấm lần cuối cùng). |
| **Vòng Private Test** | 09:00 20/11 – 23:59 23/11/2026 | Tải tập Private test, tự chạy giải pháp đã đóng băng và nộp tệp `submission.jsonl` lên cổng thi. |
| **Thẩm định & Chạy lại** | 24/11 – 01/12/2026 | BTC chạy lại Docker của các đội trên máy chuẩn (2x RTX 5090), rà soát log, kiểm tra tính tái lập và phát hiện gian lận. |
| **Công bố kết quả chung cuộc** | **02/12/2026** | Công bố Top 2 đội đạt giải cao nhất. |

---

## 10. Khuyến Nghị Thiết Kế Hệ Thống & Nguyên Tắc Kỹ Thuật

Để đạt điểm số tối đa theo hợp đồng này, hệ thống Agent cần xây dựng trên các trụ cột:
1. **Xử lý Schema & Rulebook Động**: Cơ chế tự động đọc và phân tích cấu trúc bảng cũng như nội dung `rulebook.md` của từng split; tránh phụ thuộc vào prompt cố định.
2. **Kiến Trúc Sinh Mã Thực Thi (Text-to-SQL / Text-to-Python)**: Chuyển dữ liệu lớn (lên tới 300.000 dòng) vào SQLite in-memory hoặc DuckDB/Polars, sau đó dùng LLM sinh truy vấn tính toán chính xác thay vì nhồi toàn bộ dữ liệu thô vào context của LLM.
3. **Cơ Chế Quyết Định Abstain Theo Độ Tự Tin**: Tích hợp module đánh giá độ tin cậy; chủ động chuyển sang `abstain` khi gặp lỗi truy vấn, dữ liệu mâu thuẫn hoặc điểm tin cậy $\le 50\%$.
4. **Hậu Xử Lý Định Dạng Nghiêm Ngặt**: Kiểm tra tính hợp lệ trước khi ghi kết quả (chuẩn hóa Unicode NFC cho chuỗi, làm tròn 2 chữ số thập phân cho phần trăm, đúng định dạng ISO date, đúng cấu trúc list/table).
5. **Ghi Kết Quả Tức Thời (Stream & Flush)**: Ghi từng dòng kết quả vào `/result/submission.jsonl` và flush ngay lập tức sau mỗi câu hỏi để tránh mất điểm do lỗi sập bất ngờ hoặc vượt thời gian.
