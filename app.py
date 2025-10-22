import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import altair as alt

# Load model
model_path = "models/plant_disease_model.h5"
model = tf.keras.models.load_model(model_path)
num_classes_model = model.output_shape[-1]

# Load tên lớp từ classes.txt 
classes_file = "models/classes.txt"
with open(classes_file, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines() if line.strip() != ""]

# Đồng bộ classes với số lớp model 
if len(classes) < num_classes_model:
    for i in range(len(classes), num_classes_model):
        classes.append(f"Unknown_{i}")
elif len(classes) > num_classes_model:
    classes = classes[:num_classes_model]

st.title("Plant Disease Detection (Batch Mode)")

# Hàm preprocess ảnh 
def preprocess_image(image, target_size=(128, 128)):
    img = image.convert("RGB")
    w, h = img.size
    min_dim = min(w, h)
    left, top = (w - min_dim) / 2, (h - min_dim) / 2
    right, bottom = (w + min_dim) / 2, (h + min_dim) / 2
    img = img.crop((left, top, right, bottom)).resize(target_size)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    return img_array

# Hàm xác định trạng thái
def get_status(class_name):
    if "healthy" in class_name.lower():
        return "Khỏe mạnh", "success", "#2ca02c"
    else:
        return "Có thể bị bệnh", "error", "#d62728"

# Upload nhiều ảnh
uploaded_files = st.file_uploader(
    "Chọn nhiều ảnh lá cây",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        with st.container():
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Ảnh: {uploaded_file.name}", use_container_width=True)

            preds = model.predict(preprocess_image(image))[0]
            # Đảm bảo preds có cùng số lớp với classes
            if len(preds) != len(classes):
                st.error(f"Lỗi: số lớp dự đoán ({len(preds)}) không khớp với classes.txt ({len(classes)})")
                continue

            class_idx = np.argmax(preds)
            confidence = preds[class_idx]

            status, status_type, _ = get_status(classes[class_idx])
            st.markdown(f"**Dự đoán chính:** {classes[class_idx]} ({confidence*100:.2f}%)")
            if status_type == "success":
                st.success(status)
            else:
                st.error(status)

            results.append({
                "Ảnh": uploaded_file.name,
                "Dự đoán chính": classes[class_idx],
                "Xác suất (%)": confidence*100,
                "Tình trạng": status
            })

    # Tổng hợp kết quả
    if results:
        st.subheader("Tổng hợp kết quả dự đoán")
        df_results = pd.DataFrame(results)
        st.dataframe(df_results)
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("Export CSV", csv, "plant_disease_results.csv", "text/csv")

    # Bar chart ảnh đầu tiên
    st.subheader(f"Xác suất từng lớp cho {uploaded_files[0].name}")
    first_preds = model.predict(preprocess_image(Image.open(uploaded_files[0])))[0]

    # Đồng bộ số lớp giữa preds và classes
    if len(first_preds) < len(classes):
        first_preds = np.append(first_preds, [0]*(len(classes)-len(first_preds)))
    elif len(first_preds) > len(classes):
        first_preds = first_preds[:len(classes)]

    df_chart = pd.DataFrame({
        'Lớp': classes,
        'Xác suất (%)': first_preds*100
    })
    df_chart['Màu'] = df_chart['Lớp'].apply(lambda x: "#2ca02c" if "healthy" in x.lower() else "#d62728")
    df_chart = df_chart.sort_values("Xác suất (%)", ascending=False)

    chart = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X('Xác suất (%):Q', title='Xác suất (%)'),
        y=alt.Y('Lớp:N', sort='-x', title='Lớp'),
        color=alt.Color('Màu:N', scale=None),
        tooltip=['Lớp', 'Xác suất (%)']
    ).properties(height=500, width=800)

    st.altair_chart(chart)
