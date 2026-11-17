# ドキュメント用スクリーンショットの作り方

`docs/images/` 配下の画像は、デモデータを流し込んだスクラッチサーバーに対して自動撮影されています。このファイルが撮影リストと再撮影手順の正本です — UI を変えたときに、存在しない画面がドキュメントに残り続けないようにするためのものです。

> 英語版: [screenshots.md](screenshots.md)

## 大原則

- **デモデータのみ、必ずスクラッチサーバーで。** プロジェクトのショットはグリッド全体を写すため、そのサーバーにある全プロジェクト名が公開リポジトリに入ります。撮影ツールはスクラッチか実運用かを判別できません — 専用の `CLS_PROJECTS_DIR` を与えるのは撮影者の責任です。
- 両言語を 1920×1080 で撮ります。英語は `docs/images/`、日本語は `docs/images/ja/` に置き、各ドキュメントは自分の言語のセットを埋め込みます。
- 写真主体のショットは JPEG(ノイズが PNG を無効化するため)、平坦な UI パネル主体のショットは PNG のままにします。

## 再撮影

```bash
# 1. デモ画像。合成セットを作るか...
python scripts/make_demo_images.py --out C:\scratch\demo_imgs
#    ...任意のフォルダを接頭辞で指定してもよい:
#      ok_*            正常として取り込む
#      ng_<種類>_<n>   その種類の不良として取り込む
#      probe_*         検査タブにドロップする(取り込まない)

# 2. 専用の空データディレクトリでスクラッチサーバーを起動
set CLS_PROJECTS_DIR=C:\scratch\demo_projects
.venv-windows\Scripts\python.exe -m uvicorn apps.api.app.main:app --port 8792

# 3. 撮影。ツールはバンクタブと同じ手順 — 取り込み → ラベル → 組み立て —
#    を API で行ってから UI を操作します。
cd apps/ui
node e2e/_tools/docs-screenshots.mjs --api http://localhost:8792 \
    --images C:\scratch\demo_imgs --lang en
node e2e/_tools/docs-screenshots.mjs --api http://localhost:8792 \
    --images C:\scratch\demo_imgs --lang ja

# 4. README のヒーロー GIF (5 フレームのコマ送り)
node e2e/_tools/docs-demo-gif.mjs --api http://localhost:8792 \
    --images C:\scratch\demo_imgs
python e2e/_tools/assemble-gif.py
```

実行と実行の間にデモプロジェクトを削除してください(`DELETE /api/v1/projects/{id}`)。ツールは毎回自分でプロジェクトを作るので、汚れたサーバーに対して 2 回目を回すとプロジェクトのショットにカードが 4 枚並びます。

出力は `apps/ui/e2e/screenshots/docs/<lang>/` に入ります。英語は `docs/images/`、日本語は `docs/images/ja/` にコピーし、写真主体のものは JPEG に変換してコミットします。

## 撮影リスト (すべて自動撮影)

| ファイル | 内容 | 使用先 |
|---|---|---|
| `hero.gif` | バンク作成 → 分離度評価 → OK プローブ → NG プローブとヒートマップ | README, README.ja |
| `hero.jpg` | 検査タブ、焦げた部品が NG 判定、欠陥上のヒートマップ | first-run |
| `projects.png` | デモカードが並ぶプロジェクトグリッド、片方選択 | first-run |
| `bank_label.jpg` | バンクタブ: ラベル済みリスト、選択画像、3 ステップすべて ✓ | first-run |
| `bank_marks.jpg` | バンクタブで不良画像を開き、矩形を 2 つマークした状態 | first-run |
| `check_histogram.png` | スイープ後の分離度評価、分布としきい値 | first-run |
| `check_map.png` | 特徴分離マップ、スコアで色付けされた点 | first-run |
| `inspect_queue.jpg` | OK / NG 集計付きの検査リストとヒートマップビューア | (リストのみ) |
