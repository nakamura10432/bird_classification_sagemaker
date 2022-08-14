## SageMakerの画像認識エンドポイント用スクリプト

Resnet50を用いた画像の二値分類モデルをSageMakerのエンドポイントで利用するためのスクリプトです。

フォルダ構成は下記の通り。

- entry_point.py：エンドポイントのデプロイ用スクリプト
- predict_request.py：ローカルからエンドポイントへアクセステストをするためのスクリプト
- notebook_script.py：Jupyter Notebook内での実行スクリプト

詳しくは下記のURLで説明記事をあげています。

https://www.data-flake.com/2022/08/14/amazon-sagemaker-serverless-inference/