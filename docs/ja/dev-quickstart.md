# 開発者クイックスタート

> 英語版: [../dev-quickstart.md](../dev-quickstart.md)

## 構成

```
apps/api/        FastAPI backend
  app/routers/     projects, bank, staging, score, images, inspections,
                   captures, hardware, system, root
  app/core/        state, scoring glue, schemas, settings, security
  tests/           API tests (pytest, TestClient, temp data dir)
packages/clscore/ memory-bank core — no FastAPI dependency
  clscore/        bank.py (3-tier bank), scoring.py, compress.py,
                   feature_extractor.py, incident.py, projection.py, fsio.py
  tests/           core tests (pytest, CPU-only)
apps/ui/         React + TypeScript + Vite frontend
  src/components/  Develop (Teach/Check), Operator (Inspect), Projects,
                   Settings
  e2e/             Playwright suite (see below)
scripts/         user-facing CLIs (installers, launcher,
                   reduce_bank.py bank-shrink maintenance)
```

設計境界: `clscore` は HTTP や UI について一切関知しません。採点関数は純粋関数なので、結果を単体で再現できます。状態・ロック・永続化・設定は API 層が持ちます。

## ソースから実行する

バックエンド (Python 3.10+):
```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r apps/api/requirements.txt -r apps/api/requirements-dev.txt
uvicorn apps.api.app.main:app --port 8791
```

フロントエンド (Node 18+):
```bash
cd apps/ui
npm install
npm run dev        # Vite dev server, proxies /api to :8791
```

または一度だけビルドして API に配信させることもできます: `npm run build` → `http://localhost:8791/ui/` を開きます。

## テスト

```bash
# Python — API + core (fast, CPU-only, temp data dir; never touches real data)
python -m pytest apps/api/tests packages/clscore/tests -q

# TypeScript
cd apps/ui && npx tsc --noEmit && npx eslint src --ext .ts,.tsx

# Everything (skips missing tools gracefully)
bash scripts/test.sh          # Windows: scripts\test.bat
```

### E2E (Playwright)

```bash
cd apps/ui
npx playwright install chromium   # once
npm run test:e2e
```

- ビルド済み UI を配信している稼働中のサーバー `http://localhost:8791` に対して実行します — 先にサーバーを起動してください。
- スイートは**追記のみ**です: 一意な名前の `zz-e2e-*` プロジェクトを作成し、その中で作業し、終了後に削除します。他のプロジェクトには一切触れませんが、CI 相当の実行では使い捨ての `CLS_PROJECTS_DIR` インスタンスの利用を推奨します。
- リリース時の手動チェックは [E2E チェックリスト](e2e-checklist.md)にあります。

## 規約

- Apache-2.0 です。全ソースファイルに SPDX ヘッダーを付けます。
- コミットメッセージは Conventional Commits 形式です(`feat:` / `fix:` / `docs:` / `chore:`)。
- バンクへの変更はディスク上でアトミックでなければならず(`clscore/fsio.py` のヘルパーを使用)、キャッシュ済みの GPU テンソルを無効化する必要があります(`state.mark_dirty()`)。
- 採点のセマンティクスを変える変更は、必ず評価キャッシュのフィンガープリント(`routers/bank.py`)に反映してください — 古いキャッシュが最新として提示される状態はデータ破損として扱います。
- UI: 新しいビューは既存タブの操作系を踏襲します(ズーム / パン / キー — [ユーザーガイド](user-guide.md)の表を参照)。配色は色覚多様性に配慮した状態を保ちます(色は形状/ラベルと併用し、青と紫の組み合わせは避けます)。
- pytest は常にグリーンかつ CPU のみで維持します。GPU 依存の挙動はユニットテストではなくライブスモークで確認します。

## CI

GitHub Actions(`.github/workflows/ci.yml`)で TypeScript、ESLint、UI ビルド、Ruff、pytest、pip-audit、lockfile のドリフトチェックを実行します。ローカルでの再現は `scripts/test.sh` に加え、lockfile については `python -m piptools compile` です。
