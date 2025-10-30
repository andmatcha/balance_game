Balance Game (Skeleton)
=======================

実行:
python -m balance_game.app
または python scripts/run.py

依存（必須）: numpy, opencv-python, mediapipe（MediaPipeは必須）

アセット:
balance_game/assets/images/glass.png を配置（PNG, 透明背景推奨）。未配置時はデバッグ描画。

キー操作:
- SPACE: タイトルで開始 / 結果からタイトルに戻る
- S: スタート（3秒カウントダウン、代替）
- R: リセット（タイトルへ）
- 1/2/3: 難易度切替（easy/normal/hard）
- Q: 終了

uv での実行手順:
1) 依存インストール（pyproject.toml 準拠、MediaPipe含む）
uv sync

2) 実行
uv run balance-game
# または
uv run -m balance_game.app

