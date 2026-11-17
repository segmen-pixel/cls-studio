# デプロイ

> 英語版: [../deployment.md](../deployment.md)

Cls-Studio はローカルファーストです: デフォルト設定は `127.0.0.1` のみにバインドし、画像をどこにも送信せず、アカウントも不要です。このページでは、そのデフォルトから先のすべてを説明します。

## データ配置とバックアップ

すべてのユーザーデータは `CLS_PROJECTS_DIR`(デフォルト `~/Documents/ClsStudio/projects`)配下に保存されます:

```
projects/<project-id>/
  banks/<bank-id>/
    bank.npy               # normal-tier features
    critical/<label>.npy   # defect exemplars (+ .meta.npz sidecars)
    negative/<label>.npy
    bank_meta.json         # manifest incl. per-image row index
    runtime_config.json    # saved verdict recipe
    eval_cache.json        # separation-check cache (derived)
    ivf_index.npz          # compression index (derived, auto-rebuilt)
    _images/               # lossless thumbnails of taught images
    _staging/              # staged-but-untaught drops
  captures/
  inspections/             # persisted inspection log
```

**バックアップ = projects ディレクトリのコピー**です。サーバーがアイドルのときに行ってください。全書き込みはアトミック(tmp に書いてから置換)なので、書き込み途中にコピーしても読み込み可能な状態が得られます。派生ファイル(`eval_cache.json`、`ivf_index.npz`)はいつでも再生成できます。バンク単体の移動は [インポート/エクスポート](import_export.md)で行えます。

`CLS_PROJECTS_DIR` はリポジトリツリーの**外**に置いてください — そうでない場合サーバーは起動を拒否します(リポジトリ操作からユーザーデータを保護するためです)。

## 環境変数

| 変数 | デフォルト | 用途 |
|---|---|---|
| `CLS_PROJECTS_DIR` | `~/Documents/ClsStudio/projects` | データルート |
| `CLS_DB_PATH` | データルートと同じ場所 | メタデータ用 SQLite |
| `CLS_MODELS_DIR` | `<repo>/models` | 予約済みのモデルディレクトリ。サーバーは起動時に `<dir>/registry` を作成するだけです。DINOv2 の重みはここには**入りません** — torch.hub が `TORCH_HOME` 配下にキャッシュします |
| `CLS_TORCH_DEVICE` | `auto` | `auto` / `cpu` / `cuda:N`。実行時にも `PUT /api/v1/hardware/torch/device` で切替可能 |
| `CLS_API_TOKEN` | 未設定 | 設定すると、すべての API 呼び出しで `X-API-Token` ヘッダーでの送信が必須になる共有シークレット |
| `CLS_HOST` | `127.0.0.1` | バインドアドレス |
| `CLS_MAX_PATCHES_PER_IMAGE` | `2048` | 全層に適用される画像 1 枚あたりのパッチ上限(0 = 無制限)。NG の欠陥マークは間引き後も生き残ります — マーク付きの行は常に保持されます |
| `CLS_CAPACITY_SMALL/_MEDIUM/_LARGE` | 350k / 1.4M / 4M | 設定の容量プランの背後にある行数上限 |
| `CLS_INSPECTION_LOG_CAP` | `200` | プロジェクト毎に保持する検査結果の件数 |
| `CLS_MAX_UPLOAD_TOTAL_MB` | `2048` | 複数ファイル一括アップロード 1 リクエストあたりに受け付ける合計サイズ(MB)。ファイル 1 個あたり 200 MB の上限は引き続き適用されます |
| `CLS_MAX_UPLOAD_FILES` | `1024` | 複数ファイル一括アップロード 1 リクエストで受け付ける最大ファイル数 |
| `CLS_MAX_ARCHIVE_MB` | `65536` | パッケージ読み込み(バンク / プロジェクトの zip)のサイズ上限 — これらはディスクへストリーム書き込みされるため、上記の画像単位の上限は適用されません |
| `CLS_COMPILE_MODE` | `reduce-overhead` | 特徴抽出器の `torch.compile` モード: `default` / `reduce-overhead` / `max-autotune`、または `off` でコンパイルをスキップ |
| `CLS_RUNTIME_DIR` | `~/.cls-studio/runtime` | ランタイムプロセスレジストリ(`procs.json`)用のディレクトリ |
| `CLS_OPENVINO_IR_DIR` | `./ov_ir` | OpenVINO バックエンドが `.xml` / `.bin` の IR ペアを探す場所 |
| `CLS_BG_BACKBONE_DIR` | 未設定 | 蒸留済み BG-aware エンコーダのチェックポイントを置くディレクトリ。デフォルトなし: 未設定の場合、BG バックボーンは読み込みを拒否します |
| `CLS_STATE_DIR` / `CLS_DATA_DIR` | 未設定 | 旧起動スクリプト向けに残しているレガシーエイリアス — `CLS_PROJECTS_DIR` が未設定のとき、データルートは `<value>/projects` になります |

## LAN 公開

現場ネットワーク上のオペレーター端末から使う場合:

1. 設定 → ネットワーク → **LAN 公開**を有効にします(オプトインが書き込まれ、再起動が必要です。以後サーバーは `0.0.0.0` にバインドします)。
2. `CLS_API_TOKEN` を設定します — 未設定のままだと、すべての LAN クライアントが認証なしの完全な読み書きアクセスを持ちます。トークンなしで LAN 公開が有効な場合、サーバーは起動時に警告をログに出します。
3. OS のファイアウォールで TCP 8791 を開けます(Windows インストーラーには補助スクリプト `scripts/windows/_setup_firewall.ps1` が同梱されています)。

複数のクライアントが同時にサーバーを使えます。アクティブなバンクはリクエスト毎にバインドされます(`X-Bank-Binding`): 操作の途中で別のクライアントがバンクを切り替えても、影響を受ける書き込みは誤ったバンクに着地せず明確な 409 エラーで失敗し、開いているタブは自動で再バインドされます。

**Cls-Studio を直接インターネットに公開しないでください。** 信頼できる LAN を越える用途では、TLS を終端して実際の認証を行うリバースプロキシの背後に置いてください — 下記の [リバースプロキシ](#リバースプロキシ) と [SECURITY.md](../../SECURITY.md) を参照してください。

## リバースプロキシ

Cls-Studio 自体にはユーザーアカウントも TLS もありません。複数の信頼できるマシンから到達可能なデプロイには、TLS を終端して認証を行うプロキシを前段に置く必要があります。セキュリティ境界はプロキシであり、アプリはその背後でループバック(127.0.0.1)に留まります。

基本ルール:

- プロキシが同一ホストで動く場合は `CLS_HOST=127.0.0.1` のままにしてください。明示的な `CLS_HOST` は設定画面の LAN オプトインより常に優先されるため、これがアプリをネットワークに出さない確実な方法です。`0.0.0.0` はプロキシが別マシンで動く場合にのみ使い、その際はファイアウォールで TCP 8791 をプロキシのアドレスだけに絞ってください。
- `/api` だけでなくオリジン全体をプロキシしてください。オリジンは起動方法によって変わります:
  - ネイティブインストール — `http://127.0.0.1:8791`: 1 つのポートが `/ui/` で UI を、`/api/v1/*` で REST を提供します。
  - Docker compose — `http://127.0.0.1:5173`: UI コンテナが `/` で SPA を提供し、API パスは既に API コンテナへ転送されます。
- WebSocket の配管は不要です。本リリースが公開するのはプレーンな HTTP リクエスト/レスポンスのみです — WebSocket エンドポイントもストリーミングの口もないため、プロキシに `Upgrade` の処理は要りません。
- リクエストボディの上限は、画像 1 枚ではなく複数ファイルドロップ全体に合わせてください。バッチインポートは、選択したすべてのファイルを *1 つ* の multipart リクエストで送信します: 200 MB の上限はそのリクエスト内のファイル 1 個を制限し、リクエスト全体は `CLS_MAX_UPLOAD_TOTAL_MB`(デフォルト 2048 MB)と `CLS_MAX_UPLOAD_FILES`(デフォルト 1024)で制限されます。プロキシの上限は合計側の上限に合わせてください。プロキシに 2 GB ものボディを受けさせたくない場合は、両方をまとめて下げます。
- `CLS_API_TOKEN` を設定し、`X-API-Token` ヘッダーはプロキシに注入させてください。こうすればシークレットがブラウザに渡ることはありません — 同梱の UI は自らこのヘッダーを送りません。有効なトークンのない `/api/v1/*` へのリクエストは `401` で拒否されます。

nginx、ネイティブインストールの場合:

```nginx
server {
    listen 443 ssl;
    server_name cls-studio.example.internal;

    ssl_certificate     /etc/ssl/certs/cls-studio.crt;
    ssl_certificate_key /etc/ssl/private/cls-studio.key;

    # Authentication happens here — cls-studio has none of its own.
    auth_basic           "cls-studio";
    auth_basic_user_file /etc/nginx/cls-studio.htpasswd;

    # Whole multi-file upload request, not one file: keep in sync with
    # CLS_MAX_UPLOAD_TOTAL_MB (default 2048).
    client_max_body_size 2048M;

    location / {
        proxy_pass http://127.0.0.1:8791;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;

        # Shared secret injected server-side; never seen by the browser.
        proxy_set_header X-API-Token "<value of CLS_API_TOKEN>";
    }
}
```

Caddy の同等設定:

```caddy
cls-studio.example.internal {
    basic_auth {
        operator <bcrypt-hash>
    }
    # Whole multi-file upload request; keep in sync with
    # CLS_MAX_UPLOAD_TOTAL_MB (default 2048).
    request_body {
        max_size 2048MB
    }
    reverse_proxy 127.0.0.1:8791 {
        header_up X-API-Token "<value of CLS_API_TOKEN>"
    }
}
```

## Docker

```bash
docker compose up --build
```

- UI は `http://localhost:5173/`、API は UI コンテナが中継します。
- ユーザーデータは名前付きボリューム `cls-studio-data` にあります(再ビルドしても残ります)。
- ポートはすべて `127.0.0.1` のみに公開されます。それ以上が必要な場合は、意図を持って `docker-compose.yml` を編集してください。

## GPU に関する補足

- デバイスは `CLS_TORCH_DEVICE`(または実行時の `PUT /api/v1/hardware/torch/device`)で決まります。切替は即時に反映され、モデルとキャッシュ済みのバンクテンソルが新しいデバイス上に再読み込みされます。
- VRAM 使用量は normal バンクが支配的です: デフォルトの圧縮ではバンクは int8 で常駐し(パッチ行あたり ≈ 0.75 KB)、設定の容量プランは 小 / 中 / 大 の上限でそれぞれおよそ 0.26 / 1.05 / 3 GB の常駐量に相当します。
- Windows(WDDM)では `nvidia-smi` のメモリ表示は当てになりません。アプリ内のヘルスパネルが報告する `torch.cuda.memory_allocated` が正確です。

## 知っておきたい運用挙動

- **教示または再起動後の最初の採点**では、圧縮済みバンクテンソルと(まれに)IVF インデックスが再構築されます — 大きなバンクでは一度だけ数秒の遅延が発生します。以降の採点は高速です。
- **圧縮設定の変更**は次回の採点から適用され、分離度評価のキャッシュを無効化し、保存済みの判定レシピに要再確認を付けます。ディスク上のバンクは常にフル精度のままです。
- **検査ログ**はプロジェクト毎に上限があります。長期に必要なものは書き出してください。
