import os
import shutil
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# 设置 Eager Execution
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# 临时目录管理函数
def clear_temp_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

# 数据批次生成函数
def generate_data_batches(data_dir, batch_size, target_size):
    rescale_factor = 1/255.0
    data_generator = ImageDataGenerator(rescale=rescale_factor)

    return data_generator.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )


def process_and_display_images(model, uploaded_files, temp_dir):
    try:
        # 清理并重新创建临时目录
        clear_temp_directory(temp_dir)

        # 保存上传的文件到临时目录的子目录并转换为 NPY 格式
        npy_data = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 将图像读取为 NumPy 数组并调整大小为 (256, 256)
            image = Image.open(file_path)
            image = image.resize((256, 256))
            image_array = np.array(image)

            # 将图像数组添加到列表中
            npy_data.append(image_array)

        # 转换为 NPY 格式并进行预测
        npy_data = np.array(npy_data)
        print("Input shape:", npy_data.shape)  
        predictions = model.predict(npy_data, batch_size=1, verbose=1)
        print("Predictions shape:", predictions.shape)
        predictions = np.where(predictions > 0.5, 1, 0)

        # 创建图表
        num_files = len(uploaded_files)
        fig, ax = plt.subplots(num_files, 2, figsize=(12, 6*num_files))

        # 显示原始图像和预测结果
        for i, uploaded_file in enumerate(uploaded_files):
            original_image = Image.open(uploaded_file)
            original_image = original_image.resize((256, 256))  # 调整原始图像大小为 (256, 256)
            pred_image = Image.fromarray((predictions[i].squeeze() * 255).astype(np.uint8))

            # 如果只有一张图片，则直接使用 ax 对象，不进行索引操作
            if num_files == 1:
                ax[0].imshow(original_image)
                ax[0].set_title("Original Image")

                ax[1].imshow(pred_image, cmap='gray')
                ax[1].set_title("Predicted Mask")
            else:
                ax[i, 0].imshow(original_image)
                ax[i, 0].set_title("Original Image")

                ax[i, 1].imshow(pred_image, cmap='gray')
                ax[i, 1].set_title("Predicted Mask")

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"当处理图像时有错误发生: {e}")



# Streamlit 应用程序
def main():
    # 加载模型
    model_path = 'model_isic18.h5'  # 模型路径
    model = tf.keras.models.load_model(model_path)

    st.title("皮肤病变分割工具")

    image1 = Image.open("1.jpg")
    image2 = Image.open("2.jpg")
    image3 = Image.open("3.jpg")

    st.markdown("### 分割示例结果")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image1, use_column_width=True)

    with col2:
        st.image(image2, use_column_width=True)

    with col3:
        st.image(image3, use_column_width=True)

    roc = Image.open("roc.jpg")
    pr = Image.open("pr.jpg")

    st.markdown("### 测试集评估结果")
    col_1, col_2 = st.columns(2)
    with col_1:
        st.image(roc, use_column_width=True)

    with col_2:
        st.image(pr, use_column_width=True)

    # 上传接口
    uploaded_files = st.file_uploader("上传皮肤图片", accept_multiple_files=True, type=['jpg', 'jpeg'])

    # 临时目录
    temp_dir = './temp_app_data/test'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # 按钮事件处理
    if uploaded_files and st.button("处理图像"):
        with st.spinner("处理图像中..."):
            process_and_display_images(model, uploaded_files, temp_dir)
            st.success("处理完成!")

if __name__ == "__main__":
    main()
