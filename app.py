# app.py
from flask import Flask, request, render_template, send_from_directory, jsonify
import os
from ultralytics import YOLO
from PIL import Image
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 加载 YOLOv8 模型（首次会自动下载）
model = YOLO('yolov8n.pt')  # 可替换为自定义训练好的口罩检测模型

@app.route('/')
def index():
    return '''
    <h2>口罩检测系统 (YOLOv8 + Flask)</h2>
    <form action="/detect" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">上传并检测</button>
    </form>
    '''

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "Empty filename", 400

    # 保存上传文件
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = str(uuid.uuid4()) + '.' + ext
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 使用 YOLOv8 进行推理
    results = model(filepath)

    # 保存带检测框的结果图
    result_img_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    results[0].save(filename=result_img_path)  # YOLOv8 自动保存可视化结果

    # 返回结果页面（显示原图和检测图）
    return f'''
    <h3>检测完成！</h3>
    <p>原始图像：<br><img src="/uploads/{filename}" width="400"></p>
    <p>检测结果：<br><img src="/results/{filename}" width="400"></p>
    <a href="/">再试一次</a>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
