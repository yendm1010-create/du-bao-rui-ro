import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ==============================================================================
# LỆNH STREAMLIT ĐẦU TIÊN
# ==============================================================================
st.set_page_config(layout="wide", page_title="Hệ Thống Phát Hiện Gian Lận", page_icon="🛡️")

# ==============================================================================
# CÁC HÀM CACHE & BỔ TRỢ DÙNG CHUNG
# ==============================================================================
@st.cache_data
def load_data(file_bytes, file_name):
    """Đọc dữ liệu mẫu từ tệp tải lên (CSV hoặc Excel)"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# Định nghĩa danh sách các biến đặc trưng dựa theo Notebook
FEATURE_COLS = [f'X_{i}' for i in range(1, 15)]
TARGET_COL = 'default'

# Initialize session states
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'cm' not in st.session_state:
    st.session_state.cm = None
if 'report_df' not in st.session_state:
    st.session_state.report_df = None

# ==============================================================================
# THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải dữ liệu huấn luyện mẫu
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện mẫu (dataset1.csv)", 
        type=["csv", "xlsx"],
        help="Chọn tệp CSV hoặc Excel chứa thông tin các biến X_1 đến X_14 và cột mục tiêu 'default'."
    )
    
    st.divider()
    
    # Lựa chọn mô hình dựa theo notebook công cụ sử dụng
    model_choice = st.selectbox(
        "Lựa chọn mô hình huấn luyện",
        options=["Logistic Regression", "Decision Tree", "Random Forest"],
        index=2, # Mặc định chọn Random Forest vì có hiệu năng tốt nhất trong notebook
        help="Chọn một trong ba thuật toán phân loại mà bạn muốn huấn luyện và đối sánh."
    )
    
    st.subheader("Tham số mô hình AI")
    
    # Thay đổi tham số động theo mô hình được lựa chọn
    params = {}
    if model_choice == "Logistic Regression":
        params['max_iter'] = st.slider("Số lượng vòng lặp (max_iter)", min_value=100, max_value=2000, value=100, step=50, help="Số lượng vòng lặp tối đa cho các bộ tối ưu hóa hội tụ.")
        params['C'] = st.slider("Hệ số nghịch đảo điều hòa (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.05, help="Giá trị càng nhỏ, mức độ điều hòa dữ liệu (regularization) càng mạnh.")
        
    elif model_choice == "Decision Tree":
        params['random_state'] = st.number_input("Random State", value=32, step=1, help="Đảm bảo tính tái lập kết quả phân tách cây giống như notebook.")
        params['criterion'] = st.selectbox("Tiêu chí đánh giá (criterion)", options=["gini", "entropy", "log_loss"], index=0, help="Hàm đo lường chất lượng phân tách các nhánh cây.")
        
    elif model_choice == "Random Forest":
        params['n_estimators'] = st.slider("Số lượng cây phân loại (n_estimators)", min_value=10, max_value=500, value=100, step=10, help="Số lượng cây quyết định trong rừng cây.")
        params['random_state'] = st.number_input("Random State", value=32, step=1, help="Đảm bảo tính tái lập kết quả của mô hình ngẫu nhiên.")
        
    st.divider()
    
    # Duy nhất nút này kích hoạt hành động huấn luyện dữ liệu
    train_clicked = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# ==============================================================================
# THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# ==============================================================================
st.title("🛡️ Ứng dụng Học Máy Phát Hiện Giao Dịch Gian Lận & Rủi Ro Vỡ Nợ")
st.caption("Ứng dụng hỗ trợ phân tích dữ liệu giao dịch tài chính, trực quan hóa phân phối biến và áp dụng các thuật toán phân loại (Logistic Regression, Decision Tree, Random Forest) để dự đoán nhãn rủi ro giao dịch.")

if uploaded_file is None:
    st.info("💡 Vui lòng tải tệp dữ liệu huấn luyện mẫu (ví dụ: `dataset1.csv`) tại thanh Sidebar bên trái để bắt đầu khám phá và kích hoạt mô hình.")
    st.stop()
else:
    file_bytes = uploaded_file.read()
    df = load_data(file_bytes, uploaded_file.name)
    st.caption(f"📁 **Đang sử dụng tệp dữ liệu nguồn:** `{uploaded_file.name}` | Kích thước dữ liệu thô: {df.shape[0]} dòng, {df.shape[1]} cột.")

st.divider()

# ==============================================================================
# KHỐI HUẤN LUYỆN (Chạy khi bấm nút và lưu kết quả vào session_state)
# ==============================================================================
if train_clicked and df is not None:
    with st.spinner("🔄 Đang xử lý phân tách dữ liệu và huấn luyện mô hình..."):
        # Kiểm tra xem tệp dữ liệu cấu trúc có đầy đủ các cột yêu cầu hay không
        missing_cols = [col for col in FEATURE_COLS + [TARGET_COL] if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Tệp dữ liệu thiếu các cột bắt buộc sau: {missing_cols}")
        else:
            X = df[FEATURE_COLS]
            y = df[TARGET_COL]
            
            # Phân tách tập Train/Test theo tỉ lệ 80/20 và random_state=32 đúng như notebook cấu hình
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
            
            # Khởi tạo mô hình dựa trên cấu hình người dùng chọn
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=params['max_iter'], C=params['C'])
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier(random_state=params['random_state'], criterion=params['criterion'])
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
                
            # Huấn luyện mô hình
            model.fit(X_train, y_train)
            
            # Dự đoán kiểm định trên tập test
            y_pred = model.predict(X_test)
            
            # Tính toán các chỉ số đo lường hiệu năng mô hình
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            
            # Lưu trữ toàn bộ kết quả vào st.session_state để tái dùng độc lập tại các tab nội dung
            st.session_state.trained_model = model
            st.session_state.model_name = model_choice
            st.session_state.metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
            st.session_state.cm = cm
            st.session_state.report_df = report_df
            
            st.success(f"🎉 Huấn luyện thành công mô hình **{model_choice}**! Chuyển sang Tab 'Kết quả huấn luyện' để xem thống kê chi tiết.")

# ==============================================================================
# THÂN ỨNG DỤNG CẤU TRÚC CHIA TAB NỘI DUNG
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🔬 Kết quả huấn luyện & Kiểm định", 
    "🔮 Sử dụng mô hình dự báo"
])

# ------------------------------------------------------------------------------
# TAB 1: TỔNG QUAN DỮ LIỆU
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Thống kê cấu trúc tệp dữ liệu")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Tổng số dòng dữ liệu (Rows)", f"{df.shape[0]:,}")
    with col_m2:
        st.metric("Tổng số cột đặc trưng (Columns)", f"{df.shape[1]}")
    with col_m3:
        file_size_mb = len(file_bytes) / (1024 * 1024)
        st.metric("Dung lượng tệp nguồn", f"{file_size_mb:.2f} MB")
        
    st.markdown("### 🔍 Xem dữ liệu thô đầu tiên (Head 5)")
    st.dataframe(df.head(5), use_container_width=True)
    
    st.markdown("### 📋 Thống kê mô tả các biến đặc trưng của mô hình AI")
    # Chỉ mô tả các biến thực tế đi vào mô hình (X_1 đến X_14 và default)
    available_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns]
    st.dataframe(df[available_cols].describe(), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: TRỰC QUAN HÓA DỮ LIỆU
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Phân tích trực quan hóa phân phối biến")
    
    # Ưu tiên hiển thị phân phối biến mục tiêu 'default' nếu có sẵn trong tệp
    if TARGET_COL in df.columns:
        st.markdown("#### Phân phối nhãn mục tiêu rủi ro (0: Bình thường, 1: Gian lận/Vỡ nợ)")
        target_counts = df[TARGET_COL].value_counts().reset_index()
        target_counts.columns = ['Nhãn', 'Số lượng']
        target_counts['Nhãn'] = target_counts['Nhãn'].astype(str)
        fig_target = px.bar(target_counts, x='Nhãn', y='Số lượng', color='Nhãn',
                            color_discrete_map={'0': '#2ca02c', '1': '#d62728'},
                            height=350, text_auto=True)
        st.plotly_chart(fig_target, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Trực quan hóa cấu trúc lưới (2x2) các biến số đầu vào")
    
    # Bộ chọn hiển thị đa biến nếu danh sách nhiều hơn 4 biến
    selected_features = st.multiselect(
        "Chọn 4 biến số đặc trưng bạn muốn hiển thị đồ thị đồ họa:",
        options=FEATURE_COLS,
        default=FEATURE_COLS[:4],
        max_selections=4,
        help="Hãy chọn tối đa 4 biến để hệ thống hiển thị phân phối dạng biểu đồ trực quan."
    )
    
    if len(selected_features) > 0:
        # Tạo lưới biểu đồ 2x2 bằng st.columns
        rows_cols = [st.columns(2), st.columns(2)]
        flat_cols = [rows_cols[0][0], rows_cols[0][1], rows_cols[1][0], rows_cols[1][1]]
        
        for idx, feature in enumerate(selected_features):
            if feature in df.columns:
                with flat_cols[idx]:
                    fig_hist = px.histogram(df, x=feature, marginal="box", 
                                            title=f"Phân phối tần suất biến {feature}",
                                            color_discrete_sequence=['#1f77b4'], height=300)
                    st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("⚠️ Vui lòng chọn ít nhất một biến đặc trưng từ hộp danh sách trên để vẽ biểu đồ.")

# ------------------------------------------------------------------------------
# TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH
# ------------------------------------------------------------------------------
with tab3:
    if st.session_state.trained_model is None:
        st.info("ℹ️ Hệ thống hiện tại chưa có mô hình được lưu trữ. Vui lòng chọn thuật toán và ấn nút **'Huấn luyện mô hình'** tại thanh Sidebar bên trái.")
    else:
        st.subheader(f"📊 Kết quả đánh giá mô hình chuyên sâu: **{st.session_state.model_name}**")
        
        # Trình diễn Metric dạng số đơn lẻ
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Độ chính xác tổng thể (Accuracy)", f"{st.session_state.metrics['Accuracy']:.4f}")
        with m_col2:
            st.metric("Độ chính xác dự báo đúng lớp 1 (Precision)", f"{st.session_state.metrics['Precision']:.4f}")
        with m_col3:
            st.metric("Tỷ lệ bắt sót lớp gian lận (Recall)", f"{st.session_state.metrics['Recall']:.4f}")
        with m_col4:
            st.metric("Điểm cân bằng F1-Score", f"{st.session_state.metrics['F1-Score']:.4f}")
            
        st.markdown("---")
        
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.markdown("#### Báo cáo chi tiết phân loại (Classification Report)")
            st.dataframe(st.session_state.report_df, use_container_width=True)
            
        with c_right:
            st.markdown("#### Ma trận nhầm lẫn biểu diễn trực quan (Confusion Matrix)")
            cm_data = st.session_state.cm
            # Trực quan Confusion Matrix bằng biểu đồ nhiệt Figure Factory từ Plotly
            z = cm_data
            x = ['Dự báo Bình Thường (0)', 'Dự báo Gian Lận (1)']
            y = ['Thực tế Bình Thường (0)', 'Thực tế Gian Lận (1)']
            z_text = [[str(y) for y in x] for x in z]
            
            fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Blues')
            fig_cm.update_layout(height=350, margin=dict(t=30, b=30, l=30, r=30))
            st.plotly_chart(fig_cm, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: SỬ DỤNG MÔ HÌNH DỰ BÁO
# ------------------------------------------------------------------------------
with tab4:
    if st.session_state.trained_model is None:
        st.info("ℹ️ Chức năng dự báo yêu cầu mô hình học máy cần phải được huấn luyện trước. Hãy nhấn nút kích hoạt tại Sidebar.")
    else:
        st.subheader("🔮 Chấm điểm & Dự báo rủi ro giao dịch mới")
        
        predict_mode = st.radio(
            "Chọn phương thức nhập dữ liệu đầu vào khách hàng:",
            options=["Chế độ 1: Nhập trực tiếp qua bảng mẫu", "Chế độ 2: Tải file danh sách hàng loạt (Cấu trúc tương tự X_test)"],
            horizontal=True
        )
        
        # ----------------------------------------------------------------------
        # CHẾ ĐỘ 1: NHẬP TRỰC TIẾP QUA FORM
        # ----------------------------------------------------------------------
        if "Chế độ 1" in predict_mode:
            st.markdown("##### Điền các thông số kỹ thuật của giao dịch cụ thể cần phân tích rủi ro:")
            
            # Tính toán các giá trị mặc định của form dựa theo dữ liệu trung vị thực tế để biểu mẫu thân thiện hơn
            default_input_vals = {}
            for col in FEATURE_COLS:
                if col in df.columns:
                    default_input_vals[col] = float(df[col].median())
                else:
                    default_input_vals[col] = 0.0

            # Sử dụng st.form để bao bọc các widget tránh hiện tượng load lại trang liên tục khi nhập số
            with st.form("single_prediction_form"):
                # Gom cụm các biến nhập liệu vào các cột khác nhau cho giao diện gọn đẹp
                f_cols = st.columns(4)
                input_data = {}
                
                for idx, col_name in enumerate(FEATURE_COLS):
                    col_target_block = f_cols[idx % 4]
                    with col_target_block:
                        if col_name in df.columns:
                            min_v = float(df[col_name].min())
                            max_v = float(df[col_name].max())
                        else:
                            min_v, max_v = -100.0, 100.0
                            
                        input_data[col_name] = st.number_input(
                            f"Thông số {col_name}",
                            min_value=min_v,
                            max_value=max_v,
                            value=default_input_vals[col_name],
                            format="%.4f"
                        )
                
                submit_predict = st.form_submit_button("🔍 Tiến hành Dự báo", type="primary", use_container_width=True)
                
            if submit_predict:
                # Chuyển đổi dữ liệu nhập vào thành DataFrame đúng định dạng schema cấu trúc
                input_df = pd.DataFrame([input_data])
                
                # Thực hiện dự đoán
                pred_class = st.session_state.trained_model.predict(input_df)[0]
                
                st.markdown("---")
                st.subheader("📊 Kết quả phân tích từ hệ thống AI:")
                if pred_class == 1:
                    st.error("🚨 **CẢNH BÁO:** Hệ thống nhận diện đây là một giao dịch có chỉ số **RỦI RO CAO / GIAN LẬN / VỠ NỢ (Nhãn 1)**. Cần kiểm tra rà soát ngay lập tức.")
                else:
                    st.success("✅ **AN TOÀN:** Giao dịch được phân loại ở trạng thái **BÌNH THƯỜNG / AN TOÀN (Nhãn 0)**.")
                    
                # Tính xác suất nếu mô hình được chọn có hỗ trợ predict_proba
                try:
                    pred_proba = st.session_state.trained_model.predict_proba(input_df)[0]
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.metric("Xác suất An Toàn (Lớp 0)", f"{pred_proba[0]*100:.2f}%")
                    with col_p2:
                        st.metric("Xác suất Rủi Ro (Lớp 1)", f"{pred_proba[1]*100:.2f}%")
                except:
                    pass

        # ----------------------------------------------------------------------
        # CHẾ ĐỘ 2: TẢI FILE HÀNG LOẠT
        # ----------------------------------------------------------------------
        else:
            st.markdown("##### Tải lên tệp chứa danh sách nhiều giao dịch cùng lúc để thực hiện phân tích hàng loạt:")
            bulk_file = st.file_uploader(
                "Tải lên tệp dữ liệu cần chấm điểm dự đoán (Yêu cầu có các cột đặc trưng từ X_1 đến X_14)",
                type=["csv", "xlsx"],
                key="bulk_uploader"
            )
            
            if bulk_file is not None:
                bulk_bytes = bulk_file.read()
                bulk_df = load_data(bulk_bytes, bulk_file.name)
                
                if bulk_df is not None:
                    # Kiểm tra sự đồng nhất của các trường dữ liệu
                    missing_features = [col for col in FEATURE_COLS if col not in bulk_df.columns]
                    
                    if missing_features:
                        st.error(f"❌ Tệp tin tải lên không hợp lệ. Đang thiếu các cột biến đặc trưng cấu trúc sau: {missing_features}")
                    else:
                        # Trích xuất đúng các cột X đầu vào
                        X_bulk = bulk_df[FEATURE_COLS]
                        
                        # Dự đoán hàng loạt
                        bulk_preds = st.session_state.trained_model.predict(X_bulk)
                        
                        # Gắn cột kết quả dự đoán vào DataFrame hiển thị kết quả
                        result_df = bulk_df.copy()
                        result_df['Dự_Báo_Default'] = bulk_preds
                        
                        try:
                            bulk_probas = st.session_state.trained_model.predict_proba(X_bulk)
                            result_df['Xác_Suất_Rủi_Ro_Lớp_1'] = bulk_probas[:, 1]
                        except:
                            pass
                        
                        st.success(f"⚡ Đã hoàn tất dự báo cho toàn bộ {result_df.shape[0]} bản ghi giao dịch trong tệp!")
                        
                        # Thống kê tổng quan số lượng nhãn dự báo được
                        fraud_count = int(np.sum(bulk_preds == 1))
                        normal_count = int(np.sum(bulk_preds == 0))
                        
                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                            st.metric("Số giao dịch An Toàn dự kiến (Lớp 0)", normal_count)
                        with col_r2:
                            st.metric("Số giao dịch Cảnh Báo nguy hiểm (Lớp 1)", fraud_count)
                        
                        st.markdown("#### Bảng dữ liệu kết quả phân tích chi tiết:")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Chuyển đổi tệp kết quả đầu ra thành CSV để người dùng có thể tải về máy tiện dụng
                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_output_bytes = csv_buffer.getvalue().encode('utf-8-sig')
                        
                        st.download_button(
                            label="📥 Tải xuống kết quả dự báo toàn bộ (.CSV)",
                            data=csv_output_bytes,
                            file_name=f"ket_qua_du_bao_gian_lan_{st.session_state.model_name}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
