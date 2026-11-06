# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603828
- **Họ và tên:** Nguyễn Tài Minh Huy
- **SEED** (in ở ô đầu notebook): 8004

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ Chưa xác nhận tệp nộp |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ Chưa xác nhận đã tải về |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ Chưa xác nhận đã chụp |

**Kiểm tra trước khi nộp:** Đặt ba tệp trên cạnh báo cáo, bảo đảm notebook lưu output và đổi các ô ☐ thành ☑ sau khi kiểm tra thực tế.

## 2. Kết quả chính

Mình so sánh các phương pháp đã chạy ở Phần 5 theo F1 trên tập test thông thường, F1 trên SUB 40, độ trễ p95 và chi phí API ước tính trên 1.000 tin nhắn.

| Phương pháp | F1 (test) | F1 (SUB 40) | p95 (ms/tin) | USD/1.000 tin (ước tính) |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + Logistic Regression | 0.905 | 0.974 | 0.145 | 0.000 |
| 2. BoW + MLP | 0.947 | 0.974 | 2.825 | 0.000 |
| 3a. MiniLM Embedding + kNN | 0.889 | 0.923 | 129.955 | 0.000 |
| 3b. MiniLM Embedding + Logistic Regression | 0.943 | 0.974 | 49.866 | 0.000 |
| 3c. MiniLM Embedding + MLP Head | 0.938 | 0.950 | 47.971 | 0.000 |
| 4a. Gemini LLM nhỏ, Zero-shot | — | 0.889 | 1213.542 | 0.019 |

**Nhận xét:** BoW + Logistic Regression có độ trễ thấp nhất và F1 trên SUB 40 bằng 0.974, nhưng BoW + MLP đạt F1 cao nhất trên toàn bộ test (0.947). Gemini chỉ được đánh giá trên SUB 40 nên không thể xem F1 này là kết quả trên toàn bộ tập test. Mức USD = 0.000 trong bảng là chi phí API theo phép tính của notebook, **không có nghĩa chi phí hạ tầng/CPU/GPU bằng 0**. Các lần gọi Gemini 4.A và 4.B sau đó có gặp giới hạn API 429, vì vậy chưa có kết quả đầy đủ để so sánh model lớn và Few-shot.

## 3. Dự đoán và thực tế

1. **F1 của BoW + Logistic Regression (`m1_f1`):** Mình dự đoán F1 khoảng **0.80** vì số lượng Spam thấp hơn Ham và một số tin Spam có thể bị bỏ sót. Kết quả thực tế là **0.905** trên tập test, cao hơn dự đoán **0.105**. Lý do là các từ khóa trong dữ liệu SMS thông thường vẫn cung cấp tín hiệu khá mạnh cho mô hình tuyến tính.
2. **Learning Rate (`m1_lr`):** Mình dự đoán LR = **50** sẽ khiến Loss dao động và khó hội tụ. Nhưng sau 300 epochs, Loss cuối khoảng **0.006**, thấp hơn LR = 0.5 (**0.083**) và LR = 0.001 (**0.627**). Mình hiểu rằng Learning Rate lớn không nhất thiết làm mô hình phân kỳ; kết quả còn phụ thuộc hình dạng Loss và đặc trưng dữ liệu. Tuy nhiên, Loss train thấp không đảm bảo F1 test cao hơn.
3. **Stress Test (`hard_eval`):** Mình dự đoán LLM chỉ đạt khoảng **60–80% Accuracy** trên 50 tin khó vì có các trường hợp lừa mô hình và Prompt Injection. Kết quả thực tế trên bộ 50 case được thiết kế là **100% Accuracy** cho cả LLM Zero-shot và LLM có Prompt phòng thủ. Điều này cao hơn dự đoán ít nhất 20 điểm phần trăm, nhưng không chứng minh LLM luôn đạt 100% trên SMS thực tế.

## 4. Phân tích 50 case khó

Bảng Stress Test cho thấy BoW + Logistic Regression đạt **52% Accuracy**, BoW + MLP đạt **60%**, ba phương pháp Embedding đạt **66%**, còn hai biến thể LLM đạt **100%** trên bộ 50 case này. So với khả năng phân loại tốt ở test thường, **BoW + Logistic Regression là phương pháp có Accuracy trên bộ khó thấp nhất**. Đây là so sánh theo Accuracy trên bộ khó; không được lấy số đó trừ trực tiếp cho F1 test thường vì hai chỉ số khác nhau.

- **Nhóm khó 1 – Spam giống tự nhiên (`natural_spam`):** BoW + Logistic Regression chỉ đúng **10%**. Ví dụ trong tập có tin bắt đầu bằng *“Hi there, sorry to bother you, my friend gave me your number...”* (nhãn **spam**). Tin này giống cuộc trò chuyện cá nhân, không có nhiều từ như “FREE” hoặc “WIN”, nên BoW có thể bỏ sót tín hiệu lừa đảo.
- **Nhóm khó 2 – Thao túng/lừa đảo (`manipulation`):** BoW + Logistic Regression chỉ đúng **12%**. Ví dụ trong tập có tin giả danh người thân, bắt đầu bằng *“Hi Mum, I dropped my phone in the toilet, this is my new number...”* (nhãn **spam**). Nội dung tạo cảm giác khẩn cấp, lợi dụng mối quan hệ cá nhân thay vì sử dụng từ khóa quảng cáo quen thuộc; mô hình BoW có thể đoán Ham.

**Giới hạn của bằng chứng:** Hai tin trên là ví dụ cụ thể thuộc hai **nhóm có tỷ lệ sai cao**, nhưng ảnh kết quả chỉ hiển thị tỷ lệ theo nhóm, chưa cho biết dự đoán của BoW trên *từng tin*. Trước khi nộp, mình cần xem output `hard`/`wrong` để xác nhận hai tin này thực sự bị dự đoán sai; nếu không, thay bằng hai tin sai đã được in ra từ notebook. Ngoài ra, MiniLM + kNN chỉ đúng **25%** ở nhóm *Nói về Spam*: câu đang thảo luận hoặc cảnh báo về Spam có thể gần nghĩa với Spam thật nhưng nhãn lại là Ham.

## 5. Thử thách cá nhân

- **SEED và yêu cầu riêng:** 8004 – Thử thách 1: Thay đổi ngưỡng quyết định (Decision Threshold) của Logistic Regression từ 0.5 sang 0.2 và 0.8. So sánh Precision, Recall, F1 và lựa chọn ngưỡng phù hợp cho ngân hàng muốn tránh chặn nhầm tin nhắn OTP.

- **Cách làm:** Mình sử dụng mô hình Logistic Regression đã huấn luyện ở Phần 1 với các trọng số `w` và `b`. Sau đó, mình dùng `predict_proba()` để lấy xác suất Spam trên tập test, lần lượt áp dụng các ngưỡng 0.2, 0.5 và 0.8. Cuối cùng, mình dùng hàm `metrics()` để tính Precision, Recall và F1 mà không cần train lại mô hình.

- **Kết quả:**

| Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.2 | 0.890 | 0.908 | 0.899 |
| 0.5 | 1.000 | 0.827 | 0.905 |
| 0.8 | 1.000 | 0.520 | 0.685 |

- **Giải thích:** Khi giảm ngưỡng xuống 0.2, Recall tăng vì mô hình phát hiện được nhiều Spam hơn, nhưng Precision giảm do có thêm tin Ham bị dự đoán nhầm thành Spam. Khi tăng ngưỡng lên 0.8, Precision vẫn đạt 1.000 nhưng Recall giảm mạnh vì mô hình bỏ sót nhiều Spam.

Mình lựa chọn ngưỡng 0.5 trong ba ngưỡng đã thử vì Precision đạt 100% trên tập test và F1 cao nhất (0.905). Đây là lựa chọn phù hợp hơn cho ngân hàng muốn hạn chế chặn nhầm tin OTP.

## 6. Phương án cho sản phẩm thật

Với yêu cầu **10 triệu SMS/ngày** (trung bình khoảng **116 SMS/giây**) và **p95 dưới 50 ms/tin**, mình chọn **BoW + MLP làm bộ lọc chính**. Mô hình này đạt **F1 test = 0.947** và **p95 = 2.825 ms/tin** trong lab, tốt hơn F1 test của BoW + Logistic Regression (0.905) mà vẫn nằm rất xa ngưỡng 50 ms. BoW + Logistic Regression có thể dùng làm phương án dự phòng nhanh hoặc baseline.

Mình đề xuất kiến trúc **Router hai tầng**: SMS → chuẩn hóa/kiểm tra đầu vào → BoW + MLP → nếu kết quả đủ chắc chắn thì trả nhãn ngay; nếu kết quả không chắc chắn hoặc có dấu hiệu bất thường, chuyển sang hàng đợi để Embedding/LLM kiểm tra thêm. Ngưỡng vùng không chắc chắn (ví dụ xác suất 0.3–0.7) cần được hiệu chỉnh trên validation; không xem xác suất thô là độ tin cậy đã được hiệu chuẩn. Vì LLM Zero-shot đo được **p95 khoảng 1.214 giây**, không nên đặt API LLM vào đường phản hồi đồng bộ có SLA 50 ms. Những tin chưa thể quyết định ngay phải có chính sách tạm thời, kiểm tra sau hoặc chuyển xử lý thủ công phù hợp.

Về chi phí, theo **giả định đề tính tay**: 100 input tokens và 1 output token/SMS, đơn giá $0.10 và $0.50 cho mỗi triệu tokens, nếu dùng LLM cho toàn bộ **10 triệu SMS/ngày** thì ước tính **$105/ngày**, tương đương **$3.150 cho 30 ngày**. Đó chỉ là chi phí token theo giả định, chưa gồm hạ tầng, lưu trữ, giám sát, retry và vận hành; đơn giá thực tế cần kiểm tra lại. Phương án hai tầng giúp giảm số SMS phải gọi LLM, nhưng có đánh đổi về độ phức tạp hệ thống, bảo mật nội dung tin nhắn và việc xử lý kết quả chậm.

## 7. AI log và điều đã học

Trong **Phần 4 (khu vực notebook cho phép dùng AI hỗ trợ viết code)**, mình có sử dụng AI như sau:

| Phần | Mình hỏi AI gì (prompt) | AI trả lời/viết gì | Mình đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 4.1 – Gọi LLM | Cách cấu hình Gemini API và viết hàm `call_openai()` theo OpenAI-compatible API? | Hướng dẫn thêm `GEMINI_API_KEY`, chọn provider `gemini`, gọi Chat Completions và đo latency. | Mình cấu hình và gọi thử thành công Gemini; notebook hiển thị câu trả lời và thời gian phản hồi. |
| 4.2 – Prompt | Làm thế nào tạo Zero-shot/Few-shot prompt phân loại Spam/Ham? | Gợi ý hàm `build_classification_prompt()`. | Mình sửa định dạng kết thúc `Message: """..."""` rồi chạy lại test. |
| 4.3 – Đọc nhãn | Cách xử lý các câu trả lời `Spam.`, `Not spam`, `Label: spam`? | Gợi ý `parse_label()` dùng biểu thức chính quy. | Mình bổ sung xử lý phủ định và tiền tố `Label:`; kiểm tra các lỗi test đã chỉ ra. |
| 4.4 – Batch | Cách gọi API nhiều tin, tính latency, chi phí và xử lý nhãn không hợp lệ? | Gợi ý `llm_classify_batch()` cùng `retry_with_backoff()`. | Mình đối chiếu yêu cầu `None` → nhãn 0, tăng `n_invalid`, và kiểm tra bằng API giả lập. |
| 4.A–4.B – Giới hạn API | Vì sao Gemini báo `429 RESOURCE_EXHAUSTED`? | Giải thích hạn mức requests/phút và cách giãn các lần gọi. | Mình đọc traceback, xác nhận lỗi quota 15 requests/phút, không xem đó là lỗi mô hình. |

**Điều mình học được:** Mình hiểu rõ hơn chu trình `zero_grad()` → forward → loss → backward → optimizer step và lý do ReLU giúp MLP học quan hệ phi tuyến. Mình cũng nhận ra F1 trên test thường, độ trễ, chi phí và độ bền trước các trường hợp khó phải được xem cùng nhau, không thể chọn mô hình chỉ vì Accuracy cao. Cuối cùng, mình hiểu rằng LLM có thể bắt đầu bằng Zero-shot mà không cần train trên nhãn Spam/Ham, nhưng vẫn cần kiểm thử độc lập, kiểm soát prompt injection, quota và chi phí khi triển khai thực tế.
