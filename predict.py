import gradio as gr
import timm
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import pandas as pd
import cv2
import numpy as np
# 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
test_transform = transforms.Compose(
                [transforms.Resize(300),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
                ])

def create_model(model_name):
    avail_pretrained_models = timm.list_models("*"+model_name+"*")
    if model_name in avail_pretrained_models:
        model = timm.create_model(model_name,num_classes=1000,pretrained=True)
    else:
        print("请重新输入Model的名字")
    return model


def inference(img,model_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = create_model(model_name)
    model = model.eval()
    model = model.to(device)
    img = Image.fromarray(img)
    input_img = test_transform(img)
    input_img = input_img.unsqueeze(0).to(device)
    # 执行前向预测，得到所有类别的 logit 预测分数
    pred_logits = model(input_img) 
    pred_softmax = F.softmax(pred_logits, dim=1) # 对 logit 分数做 softmax 运算
    # 解析出类别
    top_n = torch.topk(pred_softmax, 10)
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()
    # 解析出置信度
    confs = top_n[0].cpu().detach().numpy().squeeze()
    # 载入imagecsv文件
    df = pd.read_csv('imagenet_class_index.csv')
    idx_to_labels = {}
    for idx, row in df.iterrows():
        idx_to_labels[row['ID']] = [row['wordnet'], row['class']]
    # 用 opencv 载入原图
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    predict_class = {}
    for i in range(3):
        class_name = idx_to_labels[pred_ids[i]][1] # 获取类别名称
        confidence = confs[i] * 100 # 获取置信度
        text = '{:<15} {:>.4f}'.format(class_name, confidence)
        if i ==0:
            predict_class['class_name'] = class_name
            predict_class['confidence'] = confidence
        # !图片，添加的文字，左上角坐标，字体，字号，bgr颜色，线宽
        img_bgr = cv2.putText(img_bgr, text, (20, 40 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_ = Image.fromarray(img_bgr)
    return img_,predict_class



demo = gr.Interface(
    fn=inference,
    inputs=[gr.components.Image(shape=(256, 256)),gr.inputs.Dropdown(choices=["vgg19", "vgg16", "resnet18"], label="model")],
    outputs=[gr.components.Image(shape=(256,256)),gr.components.Textbox()],
    title="智能计算系统",
    description="智能计算系统Web推理应用,加载ImageNet1K预训练模型-VGG19,部署预测.",
)

demo.launch(share=True)





