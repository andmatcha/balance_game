## カメラバランスゲーム 要件定義・設計ガイド

### 概要
PC接続のWebカメラで両手の人差し指先端を検出し、各指先の直上に横長の長方形（物理オブジェクト）を落下させてバランスを取ります。SPACE でタイトルから準備画面へ、両指が検出されると自動で3秒カウントダウンしゲーム開始。物体は pymunk による剛体物理で動作し、一定時間安定を維持するとクリア、失敗条件が0.5秒以上続くとゲームオーバーになります。

### 目的/ゴール
- シンプルで実装容易な画像認識を用いてリアルタイムのインタラクティブ体験を実現
- 可読性・保守性・カスタマイズ性に優れたモジュール構成
- CPUのみ・一般的なWebカメラで十分なパフォーマンス

### スコープ
- 入力: Webカメラ映像（RGB）
- 検出: 上半身ランドマーク（鼻/両肩/両肘/両手首/両人差し指 MCP/PIP/DIP/TIP）＋顔ランドマーク（頭頂部・顎先）
  - 人差し指は MediaPipe Hands を併用して MCP/PIP/DIP/TIP を推定（未検出時はPoseのTIPにフォールバック）
- 表示: 頭上に縦長の長方形（回転・スケール、アルファ合成）
- 表示: 両手の人差し指先端の直上に横長の長方形を各1個（合計2個）表示。pymunk剛体の姿勢に基づきOpenCVで描画
- ルール: 失敗条件の不安定状態が0.5秒以上継続でゲームオーバー。安定が `clear_seconds` 連続でクリア
- UI: タイマー、難易度、ステータス（Ready/Countdown/Playing/Fail/Clear）、カウントダウン表示。指先の骨格表示は省略（必要に応じて再追加可能）

### 想定環境
- OS: macOS 14+（darwin 24.6.0）、Windows 10/11、Ubuntu 22.04 以降を想定
- Python: 3.9+（本プロジェクトは 3.9.12 仮想環境で検証）
- カメラ: UVC対応の一般的なWebカメラ

### 依存ライブラリ（必須）
- OpenCV: `opencv-python>=4.8`
- MediaPipe Pose: `mediapipe>=0.10`（CPUで動作する姿勢推定・必須）
- NumPy: `numpy`
- Pymunk: `pymunk`（剛体物理シミュレーション）
-（任意）設定: `pydantic` / `pyyaml`
-（任意）効果音: `pygame` または `playsound`

既存コード活用:
（該当なし）

### 技術選定と根拠（シンプル重視）
- 姿勢推定は MediaPipe（Pose/FaceMesh/Hands）を使用
- 描画は OpenCV で完結（GPU不要）
- 物理は Pymunk の剛体エンジンを採用（指先＝KINEMATIC円、長方形＝Dynamic剛体）。高速移動対策としてソルバ反復増加とサブステップを実施

### 機能要件
- フレーム処理: 目標 30fps（端末性能に応じて可変）
- ランドマーク: 両手の人差し指（MCP/PIP/DIP/TIP）を中心に使用。鼻/頭頂部/顎先は取得するがロジックでは未使用
- 物理オブジェクト:
  - 横長長方形×2（左手/右手）。各指先のxを初期xとして、画面上端外（中心y=−rect_half_h−1）から落下開始
  - 指先は KINEMATIC円（半径≈12px）として毎フレーム追従（速度も設定）
  - 長方形は Dynamic剛体（質量≈6.0、重力(0,1200)、摩擦0.9、反発0.0、幅≈360px・高さ≈20px）
  - 物理安定性: ソルバ反復（≈30）とサブステップでトンネリングを抑制
- 判定:
  - 失敗条件（どちらか一方でも）を0.5秒以上連続で満たすと FAIL
    1) 一度でも可視になった後に完全に画面外（AABBが画面矩形と非交差）
    2) 水平方向の完全逸脱（AABBの[min_x,max_x]が指x±指半径の帯と非交差）
    3) 長方形全体が指より下（AABBのmin_y > 指y + 指半径 + マージン）
  - クリア: 安定（上記失敗条件を満たさない）が `clear_seconds` 連続で CLEAR
- 準備/開始:
  - Title→SPACEで Prepare。Prepare 中に両指TIPが9フレーム連続で検出されると自動で3秒カウントダウン→Playing
  - S キーで即カウントダウン開始（デバッグ用）
- UI: HUD（タイマー、難易度、状態、平均傾きなど）、タイトル・準備・リザルト画面
- ログ/保存（任意）: スクリーンショット、CSVログ

### 画面/遷移とUI
- 画面（Screen）:
  - Title: タイトル表示、難易度表示、操作ガイド（1/2/3, SPACE）。
  - Prepare: カウントダウンの準備画面（両指検出の案内）。
  - Countdown: 中央に残り秒数（3→2→1）。終了で自動的に Playing へ遷移。
  - Playing: HUD（タイマー・難易度・傾き等）＋安定時間の進捗。Fail/Clear 判定。
  - Result: GAME CLEAR/OVER とスコア（Elapsed/Stable）表示。SPACE で Title へ。
- 遷移:
  - Title -(SPACE)-> Prepare -(両指9フレーム)-> Countdown -(3s経過)-> Playing -(Clear/Fail)-> Result -(SPACE)-> Title
  - 難易度切替（1/2/3）は常時可。Reset（R）は Title に戻す。S キーで即カウントダウン開始。
- 実装指針:
  - `GameStatus` は COUNTDOWN/PLAYING/CLEAR/FAIL を管理。
  - `Screen` は Title/Prepare/Countdown/Playing/Result を管理（`app_core.ScreenManager`）。
  - `ui.py` に `draw_title`, `draw_result`, `draw_prepare`, `draw_hud` を用意。

### UI実装方針（ライブラリ選定）
- ベースライン: OpenCV のみで実装（追加依存なし、最小実装・低コスト）。
- 代替案（将来拡張時）:
  - Pygame(+pygame-menu): メニュー/効果音/フォント強化に適。OpenCV→Surface 変換が必要。
  - GUI系（PyQt/DearPyGui）: 高機能だが本要件には過剰。

### 非機能要件
- モジュール分割（単一責務）で可読性・保守性を確保
- 設定の外部化と難易度切替の容易さ
- 時間基準更新（フレームレート変動にロバスト）
- CPUのみで動作、依存は最小限

### システム構成とデータフロー
1. Camera: Webカメラからフレーム取得
2. PoseDetector: MediaPipeでランドマーク抽出（Pose＋FaceMesh＋Hands）
3. GameLogic: 難易度・状態管理、カウントダウン、安定/失敗の時間管理
4. Physics: pymunk Space の更新（指先＝KINEMATIC円、長方形＝Dynamic剛体）。`physics.update/draw`
5. UI: HUD/タイトル/準備/リザルトの描画
6. AppCore: `GameApp`/`ScreenManager` によるループ制御・画面遷移・入力処理

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
    app.py            # エントリポイント（GameApp.run を呼び出し）
    app_core.py       # ループ制御/画面遷移/入力処理（GameApp, ScreenManager, handle_key_input）
    config.py         # 設定（難易度・しきい値・入出力）
    camera.py         # カメラ抽象
    pose_detector.py  # MediaPipe検出（必須）
    physics.py        # 指先追従 + 剛体長方形の物理/安定判定（pymunk）
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

### 物理モデルと判定（pymunk）
1. 指先: KINEMATIC円（r≈12）。毎フレーム座標と速度を設定して剛体に追従。
2. 長方形: Dynamic剛体（m≈6.0、摩擦0.9、反発0.0）。サイズは幅≈360×高さ≈20。
3. 環境: 重力(0,1200)、ソルバ反復≈30、サブステップで `dt ≤ 1/240` を維持。
4. スポーン: ゲーム開始時、各指のxを用い、中心y=−rect_half_h−1（画面上端外）から落下開始。
5. 失敗条件（いずれか、0.5s連続）:
   - 一度でも可視になった後の完全不可視（AABBが画面矩形と非交差）
   - 指帯からの水平全逸脱（AABBの[min_x,max_x]が指x±指半径の帯と非交差）
   - 長方形全体が指より下（min_y > 指y + 指半径 + マージン）
6. 安定/クリア: 上記失敗が連続0.5s未満なら安定。`clear_seconds` 連続で CLEAR。

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
- SPACE: Title→Prepare（開始準備）/ Result→Title
- S: 3秒カウントダウン開始
- R: リセット（Title へ）
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


