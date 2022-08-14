import logging
import torch
import torch.nn as nn
import numpy as np
from six import BytesIO
import os
import json

import torchvision.models as models
from torchvision import transforms
import cv2

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

JSON_CONTENT_TYPE = 'application/json'
PNG_CONTENT_TYPE  = 'image/png'
JPG_CONTENT_TYPE  = 'image/jpeg'
NPY_CONTENT_TYPE  = 'application/x-npy'

class ImageTransform(object):
    def __init__(self, resize, mean, std):
        self.data_trasnform = {
            'train': transforms.Compose([
                # データオーグメンテーション
                transforms.RandomHorizontalFlip(),
                # 画像をresize×resizeの大きさに統一する
                transforms.Resize((resize, resize)),
                # Tensor型に変換する
                transforms.ToTensor(),
                # 色情報の標準化をする
                transforms.Normalize(mean, std)
            ]),
            'valid': transforms.Compose([
                # 画像をresize×resizeの大きさに統一する
                transforms.Resize((resize, resize)),
                # Tensor型に変換する
                transforms.ToTensor(),
                # 色情報の標準化をする
                transforms.Normalize(mean, std)
            ])
        }
    
    def __call__(self, img, phase='train'):
        return self.data_trasnform[phase](img)

def model_fn(model_dir):
    """モデルのロード."""
    logger.info('START model_fn')
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2)
    # モデルのパラメータ設定
    with open(os.path.join(model_dir, 'bird_torch.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))
    logger.info('END   model_fn')
    return model

def input_fn(request_body, content_type=PNG_CONTENT_TYPE):
    """入力データの形式変換."""
    logger.info('START input_fn')
    logger.info(f'content_type: {content_type}')
    logger.info(f'request_body: {request_body}')
    logger.info(f'type: {type(request_body)}')
    if content_type == PNG_CONTENT_TYPE:#PNGの場合の処理
        #受信した画像データをバイナリデータとして受け取る
        stream = BytesIO(request_body)
        #バイナリデータをNumpy配列に変換
        input_data = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        #Numpy配列を画像形式のTensorに変換する
        input_data = cv2.imdecode(input_data, 1)
        #Tensorを画像に変換する
        transform_PIL = transforms.ToPILImage()
        input_data = transform_PIL(input_data)
    elif content_type == JPG_CONTENT_TYPE:#JPGの場合。PNGと同様の処理です
        stream = BytesIO(request_body)
        input_data = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        input_data = cv2.imdecode(input_data, 1)
        transform_PIL = transforms.ToPILImage()
        input_data = transform_PIL(input_data)
    elif content_type == NPY_CONTENT_TYPE:
        stream = BytesIO(request_body)
        transform_PIL = transforms.ToPILImage()
        input_data = transform_PIL(np.load(stream))
    else:
        # TODO: content_typeに応じてデータ型変換
        logger.error(f"content_type invalid: {content_type}")
        input_data = {"errors": [f"content_type invalid: {content_type}"]}
    logger.info('END   input_fn')
    return input_data

def predict_fn(input_data, model):
    """推論."""
    logger.info('START predict_fn')

    if isinstance(input_data, dict) and 'errors' in input_data:
        logger.info('SKIP  predict_fn')
        logger.info('END   predict_fn')
        return input_data
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 説明変数の標準化
    # リサイズ先の画像サイズ
    resize = 300
    # 今回は簡易的に(0.5, 0.5, 0.5)で標準化
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = ImageTransform(resize, mean, std)
    img_transformed = transform(input_data, 'valid').unsqueeze(0)

    # 推論
    with torch.no_grad():
        logger.info('END   predict_fn')
        return model(img_transformed.to(device))

def output_fn(prediction, accept=JSON_CONTENT_TYPE):
    """出力データの形式変換."""
    logger.info('START output_fn')
    logger.info(f"accept: {accept}")

    if isinstance(prediction, dict) and 'errors' in prediction:
        logger.info('SKIP  output_fn')
        response = json.dumps(prediction)
        content_type = JSON_CONTENT_TYPE
    elif accept == JSON_CONTENT_TYPE:
        #[0]だと「可愛い」判定、[1]だと「かっこいい」判定です
        pred = []
        pred += [int(l.argmax()) for l in prediction]
        m = nn.Softmax(dim=1)#「可愛い」と「かっこいい」
        response = json.dumps({"results": pred[0], "eval":{'kawaii':m(prediction)[0].tolist()[0], 'kakkoii':m(prediction)[0].tolist()[1]}})
        content_type = JSON_CONTENT_TYPE
    else:
        #[0]だと「可愛い」判定、[1]だと「かっこいい」判定です
        pred = []
        pred += [int(l.argmax()) for l in prediction]
        m = nn.Softmax(dim=1)#「可愛い」と「かっこいい」
        response = json.dumps({"results": pred[0], "eval":{'kawaii':m(prediction)[0].tolist()[0], 'kakkoii':m(prediction)[0].tolist()[1]}})
        content_type = JSON_CONTENT_TYPE

    logger.info('END   output_fn')
    return response, content_type


if __name__ == '__main__':
    logger.info("process main!")
    pass