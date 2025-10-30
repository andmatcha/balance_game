## カメラバランスゲーム 要件定義・設計ガイド

### 概要
PC接続のWebカメラで上半身の姿勢を推定し、頭上に縦長の長方形（仮想物体）を表示します。Sキーで3秒カウントダウン後にゲーム開始。頭の傾きや左右方向の動きに応じて物体が物理演算で自然に傾き、許容傾き超過でゲームオーバー。一定時間バランス維持でクリアします。

### 目的/ゴール
- シンプルで実装容易な画像認識を用いてリアルタイムのインタラクティブ体験を実現
- 可読性・保守性・カスタマイズ性に優れたモジュール構成
- CPUのみ・一般的なWebカメラで十分なパフォーマンス

### スコープ
- 入力: Webカメラ映像（RGB）
- 検出: 上半身ランドマーク（鼻/両肩/両肘/両手首/両人差し指 MCP/PIP/DIP/TIP）＋顔ランドマーク（頭頂部・顎先）
  - 人差し指は MediaPipe Hands を併用して MCP/PIP/DIP/TIP を推定（未検出時はPoseのTIPにフォールバック）
- 表示: 頭上に縦長の長方形（回転・スケール、アルファ合成）
- ルール: 角度ダイナミクスに基づく安定判定、一定時間クリア
- UI: タイマー、難易度、ステータス（Ready/Countdown/Playing/Fail/Clear）、カウントダウン表示、頭頂部と顎先の点＋直線、頭頂部を通る直交ガイドライン（顔幅程度）、肩/肘/手首/指先の点と腕ライン描画、人差し指の各関節点（MCP/PIP/DIP/TIP）と接続線

### 想定環境
- OS: macOS 14+（darwin 24.6.0）、Windows 10/11、Ubuntu 22.04 以降を想定
- Python: 3.9+（本プロジェクトは 3.9.12 仮想環境で検証）
- カメラ: UVC対応の一般的なWebカメラ

### 依存ライブラリ（必須）
- OpenCV: `opencv-python>=4.8`
- MediaPipe Pose: `mediapipe>=0.10`（CPUで動作する姿勢推定・必須）
- NumPy: `numpy`
-（任意）設定: `pydantic` / `pyyaml`
-（任意）効果音: `pygame` または `playsound`

既存コード活用:
（該当なし）

### 技術選定と根拠（シンプル重視）
- 姿勢推定は MediaPipe Pose を必須とし、導入と実装が容易でCPUでも実用的
- 描画は OpenCV のアルファ合成で完結（GPU不要）
- 物理は簡易な回転剛体モデル（角速度/角加速度・減衰）で表現

### 機能要件
- フレーム処理: 目標 30fps（端末性能に応じて可変）
- ランドマーク: 鼻（頭基準）、両手首、両肩の2D座標
- オーバーレイ: 頭上/左右手首にPNG（回転・スケール対応、アルファ合成）
- 判定: 体の傾き・急加速（ジャーク）をしきい値で監視し、しきい値超過で「落下」
- クリア: 安定状態が設定秒数連続達成
- 難易度: Easy/Normal/Hard（許容傾き・ジャーク・クリア秒数を変更）
- UI: タイマー、難易度表示、状態、ヒント用ガイドライン
- ログ/保存（任意）: スクリーンショット、CSVログ

### 非機能要件
- モジュール分割（単一責務）で可読性・保守性を確保
- 設定の外部化と難易度切替の容易さ
- 時間基準更新（フレームレート変動にロバスト）
- CPUのみで動作、依存は最小限

### システム構成とデータフロー
1. Camera: Webカメラからフレーム取得
2. PoseDetector: MediaPipeでランドマーク抽出（Pose＋FaceMesh＋Hands、フォールバック無し）
3. GameLogic: 難易度・状態管理、カウントダウン、安定判定、スコア・タイマー更新
4. OverlayRenderer: 長方形の回転・スケール・アルファ合成
5. UI: テキスト/カウントダウン描画
6. App: ループ制御・入出力の統合

### データモデル（インタフェース草案）
```python
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Point2D:
    x: float
    y: float

@dataclass
class Keypoints2D:
    nose: Optional[Point2D]
    left_wrist: Optional[Point2D]
    right_wrist: Optional[Point2D]
    left_shoulder: Optional[Point2D]
    right_shoulder: Optional[Point2D]
    left_elbow: Optional[Point2D]
    right_elbow: Optional[Point2D]
    left_index: Optional[Point2D]
    right_index: Optional[Point2D]
    head_top: Optional[Point2D]   # FaceMesh由来（近似）
    chin: Optional[Point2D]       # FaceMesh由来

@dataclass
class StabilizerConfig:
    max_tilt_deg: float      # 許容傾き（小さいほど難しい）
    max_jerk: float          # 許容ジャーク（px/ms）
    clear_seconds: float     # クリアに必要な連続安定時間

@dataclass
class GameConfig:
    target_fps: int
    difficulty: str          # "easy" | "normal" | "hard"
    stabilizer: StabilizerConfig

class PoseDetector:
    def detect(self, frame_bgr) -> Tuple[Keypoints2D, dict]:
        ...
```

### ディレクトリ構成（提案）
```text
balance_game/
  assets/
    images/           # （任意）将来のPNGアセット用。現仕様は長方形のみ
    sounds/           # （任意）
  balance_game/
    __init__.py
    app.py            # メインループ
    config.py         # 設定（難易度・しきい値・入出力）
    camera.py         # カメラ抽象
    pose_detector.py  # MediaPipe検出（必須）
    overlay.py        # 回転・スケール・アルファ合成
    physics.py        # 角度ダイナミクス/安定判定
    game_logic.py     # 状態・カウントダウン・スコア・遷移
    ui.py             # テキスト/カウントダウン描画
    types.py          # dataclass/型定義
    utils/
      geometry.py
      timing.py
      drawing.py
  scripts/
    run.py            # 実行エントリ（例: python -m balance_game.app）
  tests/
    test_physics.py
    test_overlay.py
  README.md
```

既存資産の参照:
（該当なし）

### 設定項目（例）
```python
EASY   = StabilizerConfig(max_tilt_deg=18.0, max_jerk=0.8, clear_seconds=5.0)
NORMAL = StabilizerConfig(max_tilt_deg=12.0, max_jerk=0.5, clear_seconds=10.0)
HARD   = StabilizerConfig(max_tilt_deg= 8.0, max_jerk=0.3, clear_seconds=15.0)
```
備考: max_tilt_deg は「物体（長方形）の許容傾き閾値」として利用する。

### 物理モデルと判定（簡易回転剛体）
1. 頭の傾き φ（deg）: 両肩の中点→鼻ベクトルと正上方向の角度差。
2. 水平移動速度 v_x（px/s）: 鼻の x 位置の時間微分。
3. 角加速度 α: `α = k_tilt * rad(φ) + k_move * (v_x / s_norm) - c_damp * ω`。
   - ω: 角速度、θ: 角度（長方形の傾き）。オイラー法で更新。
4. 判定: `abs(deg(θ)) <= max_tilt_deg` を連続達成で安定、閾値超過でゲームオーバー。
5. 初期値: θ=0, ω=0。微小な減衰 `c_damp` を付与し自然減衰。

### 実行方法（uv）
```bash
# 依存関係をインストール（pyproject.toml 準拠、MediaPipe含む）
uv sync

# 実行
uv run balance-game
# または
uv run -m balance_game.app
```

キー操作:
- S: 3秒カウントダウン開始 → PLAYING へ遷移
- R: リセット（READY へ）
- Q: 終了
- 1/2/3: 難易度切替（easy/normal/hard）



### リスクとフォールバック
- MediaPipeは必須: インストール/環境依存の問題があれば `uv sync` の再実行、Python 3.9系の利用、アーキテクチャ互換の確認を推奨
- 低照度・逆光で精度低下: 明るさ自動調整、履歴平滑化（移動平均/指数平滑）
- パフォーマンス低下: 入力解像度の縮小、検出間引き（Nフレームに1回）

### 今後の拡張案
- マルチプレイ（2人同時）
- 物体パラメータや重心の種類を増やす（難易度の多様化）
- ポーズ全体（肘/膝等）への拡張、ミニゲーム化


