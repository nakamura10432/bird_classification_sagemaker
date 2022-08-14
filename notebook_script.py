# エンドポイントの作成、デプロイ
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# モデルの作成
pytorch_model = PyTorchModel(model_data="s3://*****/models/bird_torch.tar.gz",
                             role=role,
                             framework_version='1.11.0',
                             py_version="py38",
                             entry_point="entry_point.py")

# デプロイパラメータ
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048, #利用するメモリサイズ。インスタンスの最大メモリより大きくはできない
    max_concurrency=2
)
deploy_params = {
    'instance_type'          : 'ml.t3.medium', 
    'initial_instance_count' : 1,
    'serverless_inference_config' : serverless_config #ここでサーバレスインスタンスの利用設定を行う
}

# デプロイ
predictor = pytorch_model.deploy(**deploy_params)

#テスト
from PIL import Image
#入力データ 
file_name = 'sample_1.jpeg'
img = Image.open(file_name)
# 推論
results = predictor.predict(img)
print(results)