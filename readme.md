# 🛡️ Ứng dụng Web Học Máy Phát Hiện Giao Dịch Gian Lận (Streamlit App)

Ứng dụng này được phát triển nhằm mục đích chuyển đổi quy trình thử nghiệm các mô hình học máy phân loại từ định dạng Jupyter Notebook sang nền tảng giao diện Web tương tác trực quan thông minh. Hệ thống hỗ trợ đắc lực cho các chuyên viên phân tích tài chính trong việc phát hiện sớm các giao dịch bất thường, nguy cơ gian lận hoặc rủi ro vỡ nợ tín dụng (`default`).

---

## ✨ Các Tính Năng Chính Của Ứng Dụng

1. **Cấu hình tham số linh hoạt (Sidebar):** Cho phép người dùng tùy biến tải tệp dữ liệu huấn luyện, lựa chọn thay đổi thuật toán học máy cùng các siêu tham số mô hình một cách chủ động thông qua các thanh trượt trước khi bấm chạy huấn luyện.
2. **Tổng quan dữ liệu khoa học:** Thống kê chi tiết số hàng, số cột, dung lượng file mẫu cùng bảng mô tả trực quan các phân phối thống kê (`describe`).
3. **Trực quan hóa dữ liệu tương tác:** Sử dụng thư viện đồ họa kéo thả hiện đại `Plotly` giúp hiển thị cấu trúc lưới phân phối tần suất của nhãn mục tiêu và các biến đặc trưng số.
4. **Kiểm định mô hình chuyên sâu:** Tái lập chính xác quy trình chấm điểm kiểm nghiệm trên tập dữ liệu Test ẩn (tỷ lệ 80/20) tương tự quy trình xây dựng trong Notebook nguyên bản. Hiển thị điểm số Accuracy, Precision, Recall, F1-Score cùng bảng Classification Report và biểu đồ nhiệt Confusion Matrix trực quan.
5. **Hỗ trợ 2 chế độ dự báo nâng cao:**
   - **Nhập dữ liệu trực tiếp:** Nhập các thông số của một giao dịch đơn lẻ thông qua Form để hệ thống chấm điểm rủi ro ngay lập tức kèm xác suất cụ thể.
   - **Dự báo tệp hàng loạt:** Tải lên một bảng chứa danh sách nhiều khách hàng/giao dịch mới, ứng dụng tự động kiểm tra định dạng cấu trúc, dự đoán nhãn hàng loạt và cho phép xuất ngược tệp CSV kết quả.

---

## 🛠️ Cấu Trúc File Dữ Liệu Đầu Vào Kỳ Vọng

Để ứng dụng phân tích chính xác và không bị lỗi hệ thống, tệp dữ liệu tải lên tại Sidebar hoặc Tab dự báo cần tuân thủ cấu trúc định dạng sau:
- **Định dạng cho phép:** Tệp tin đuôi `.csv` hoặc `.xlsx`.
- **Danh sách các biến đặc trưng bắt buộc (Features):** Bao gồm đầy đủ 14 cột dữ liệu số định danh từ `X_1`, `X_2`, `X_3`, ..., cho đến `X_14`.
- **Cột nhãn mục tiêu (Chỉ bắt buộc đối với tệp huấn luyện mẫu tại Sidebar):** Cột mang tên `default` chứa giá trị nhị phân (`0`: Giao dịch bình thường / an toàn; `1`: Giao dịch có dấu hiệu gian lận hoặc vỡ nợ).

---

## 🚀 Hướng Dẫn Cài Đặt Và Vận Hành Ứng Dụng

### Bước 1: Chuẩn bị môi trường máy tính
Đảm bảo máy tính của bạn đã cài đặt sẵn công cụ quản lý thư viện Python (Phiên bản khuyến nghị `Python >= 3.9`).

### Bước 2: Cài đặt các thư viện bổ trợ cần thiết
Di chuyển con trỏ dòng lệnh Terminal/Command Prompt vào thư mục chứa dự án có 3 file này và khởi chạy lệnh cài đặt hàng loạt:
```bash
pip install -r requirements.txt
