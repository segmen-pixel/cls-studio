// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import type { TabId } from "./types";

export type TutorialMode = "beginner" | "intermediate" | "expert";

export type TutorialAdvance =
  | { type: "click"; selector: string }
  | { type: "tabEnter"; tabId: TabId }
  | { type: "manual" };

export type TutorialStep = {
  id: string;
  /** CSS selector for the target element. Null = centered modal (no spotlight). */
  targetSelector: string | null;
  /** Tab the user should be on for this step; if different, we auto-switch. Undefined = any tab. */
  requireTab?: TabId;
  titleJa: string;
  titleEn: string;
  bodyJa: string;
  bodyEn: string;
  /** Condition to auto-advance to the next step. Manual = user must press Next. */
  advanceOn: TutorialAdvance;
  /** Tooltip placement relative to the target. */
  placement?: "top" | "bottom" | "left" | "right" | "center";
  /** Which modes include this step. Undefined = all modes. */
  modes?: TutorialMode[];
  /** Shown on the welcome step only; renders mode-select buttons instead of Next. */
  isModeSelect?: boolean;
  /**
   * When set, the tutorial programmatically clicks the matching element
   * once the step is shown. Use to auto-open a dialog whose contents the
   * next steps will spotlight. One-shot per step activation — re-entering
   * the same step does NOT re-click.
   */
  onEnterClickSelector?: string;
};

/**
 * All tutorial steps in a single ordered list. Filter by mode at runtime via `modes`.
 * Steps with `modes` undefined appear in every mode.
 *
 * Flow mirrors the real Cls-Studio workflow: create a project → import,
 * label and assemble on the バンク (bank) tab → run the evaluation on the
 * 学習 (develop) tab → read heatmaps and the NG度 score → tune the threshold
 * → save the verdict config on the 検査 (operator) tab.
 */
export const TUTORIAL_STEPS: TutorialStep[] = [
  {
    id: "welcome",
    targetSelector: null,
    titleJa: "Cls-Studio へようこそ！",
    titleEn: "Welcome to Cls-Studio!",
    bodyJa: "ハンズオンチュートリアルを始めます。自分に合ったモードを選んでください。途中でスキップもできます。",
    bodyEn: "Let's take a hands-on tour. Pick the mode that matches your experience. You can skip any time.",
    advanceOn: { type: "manual" },
    placement: "center",
    isModeSelect: true,
  },
  {
    id: "projects-tab",
    targetSelector: '[data-tutorial-step="projects-tab"]',
    titleJa: "プロジェクトタブ",
    titleEn: "Projects tab",
    bodyJa: "まずは「プロジェクト」タブを開きます。プロジェクトは画像・メモリバンク・判定設定をひとまとめにする作業単位で、検査対象（製品・工程）ごとに分けて管理します。各プロジェクトは独立したフォルダに保存されます。",
    bodyEn: "Start by opening the Projects tab. A project is a workspace bundling images, memory banks, and verdict settings — keep one per product or inspection target. Each project lives in its own folder.",
    advanceOn: { type: "tabEnter", tabId: "projects" },
    placement: "bottom",
  },
  {
    id: "create-project",
    targetSelector: '[data-tutorial-step="create-project-btn"]',
    requireTab: "projects",
    titleJa: "新規プロジェクトを作成",
    titleEn: "Create a new project",
    bodyJa: "「新規プロジェクト」ボタンで作成します。名前とメモを付けておくと後で探しやすくなります。既存のプロジェクトを開きたい場合はこのステップを飛ばし、下のグリッドから選択してください。",
    bodyEn: "Click New Project to create one — a name and a memo make it easy to find later. If you already have a project, skip this and pick it from the grid below.",
    advanceOn: { type: "manual" },
    placement: "bottom",
  },
  {
    id: "bank-tab",
    targetSelector: '[data-tutorial-step="bank-tab"]',
    titleJa: "「バンク」タブへ",
    titleEn: "Switch to the Bank tab",
    bodyJa: "次は「バンク」タブ。判定の土台になる記憶をここで作ります。右側に番号付きの 3 ステップが並んでいて、順に片付ければ完成です。終わったステップには ✓ が付きます。",
    bodyEn: "Next: the Bank tab, where you build the memory the verdict is made against. Three numbered steps run down the right-hand side; work through them in order and each ticks as it finishes.",
    advanceOn: { type: "tabEnter", tabId: "bank" },
    placement: "bottom",
  },
  {
    id: "bank-import",
    targetSelector: '[data-tutorial-step="bank-step-1"]',
    requireTab: "bank",
    titleJa: "① 画像を入れる",
    titleEn: "① Get the images in",
    bodyJa: "「+ 画像をインポート」で写真を選びます。一度に全部で構いません。各画像はエンコーダを 1 回だけ通ってパッチが保存されます — 時間がかかるのはここだけで、後からラベルを付け直しても二度と払いません。",
    bodyEn: "Press + Import images and pick your photos — all at once is fine. Each image goes through the encoder exactly once and its patches are stored. This is the only slow part, and relabelling later never repeats it.",
    advanceOn: { type: "manual" },
    placement: "left",
  },
  {
    id: "bank-label",
    targetSelector: '[data-tutorial-step="bank-step-2"]',
    requireTab: "bank",
    titleJa: "② 1 枚ずつ判断する",
    titleEn: "② Judge them one at a time",
    bodyJa: "左のリストで行を選び、1 / 2 / 3 キー（またはボタン）で 正常 / 不良 / 過検知抑制 を割り当てます。不良には種類（「傷」「焦げ」など）も付けてください。特徴は動かないので、ラベルは何度でも変えられます。不良画像は開いて「欠陥をマーク」から欠陥部分に矩形を描くと、そのパッチが見本になります。",
    bodyEn: "Select rows in the list and press 1 / 2 / 3 (or use the buttons) to assign Normal, Defect or Suppress FP. Give each defect a kind (\"scratch\", \"burnt\", ...). The features never move, so labels are free to change. Open a defect image and press Mark defect to drag rectangles over the defect itself — those patches become the exemplars.",
    advanceOn: { type: "manual" },
    placement: "left",
  },
  {
    id: "bank-assemble",
    targetSelector: '[data-tutorial-step="bank-step-3"]',
    requireTab: "bank",
    titleJa: "③ 判断をバンクに反映する",
    titleEn: "③ Fold the labels into the bank",
    bodyJa: "ここが要点です: **ラベルを変えただけでは検査は何も変わりません**。「バンクを組み立てる」を押して初めて反映されます。ラベルを 1 つでも変えるとこのステップの ✓ は自動的に外れるので、それがバンクとラベルがずれている合図です。",
    bodyEn: "This is the one that matters: **changing labels alone changes nothing inspection sees**. Press Assemble the bank to fold them in. Change any label afterwards and this step unticks itself — that is the signal that the bank and the labels have drifted apart.",
    advanceOn: { type: "manual" },
    placement: "left",
  },
  {
    id: "develop-tab",
    targetSelector: '[data-tutorial-step="develop-tab"]',
    titleJa: "「学習」タブへ",
    titleEn: "Switch to the Teach tab",
    bodyJa: "次は「学習」タブ。組み上げたバンクが OK と NG をどれだけ見分けられるかを、ここで確かめます。名前に反してニューラルネットの訓練は一切ありません — 記憶した特徴との距離を測るだけなので、数十枚・数分から始められます。",
    bodyEn: "Next: the Teach tab, where you find out how well the bank you just assembled can tell OK from NG. Despite the name there is no neural-network training at all — it only measures distance to what was memorised, so tens of images and a few minutes are enough to start.",
    advanceOn: { type: "tabEnter", tabId: "develop" },
    placement: "bottom",
  },
  {
    id: "develop-run-eval",
    targetSelector: '[data-tutorial-step="develop-run-eval"]',
    requireTab: "develop",
    titleJa: "評価を実行",
    titleEn: "Run the evaluation",
    bodyJa: "緑の「▶ 評価を実行」が一気通貫ボタンです。バンク内の全画像の採点 → マップ更新 → ヒートマップの準備までまとめて実行します。バンクを組み立て直したら、まずこれを押すのが基本サイクルです。右下の「検証」で、1 枚だけ除外するか同一ロットごと除外するかを選べます。",
    bodyEn: "The green Run Evaluation button does everything in one pass: scores every image in the bank, refreshes the maps, and pre-renders heatmaps. After reassembling the bank, pressing this is the core loop. The Validation control below picks whether one image or a whole lot is held out.",
    advanceOn: { type: "manual" },
    placement: "right",
  },
  {
    id: "develop-viewer",
    targetSelector: '[data-tutorial-step="develop-viewer"]',
    requireTab: "develop",
    titleJa: "ビューアの操作",
    titleEn: "Viewer controls",
    bodyJa: "リストで画像を選ぶと右のビューアに表示されます。ホイールでカーソル中心にズーム、ドラッグ（または Space＋ドラッグ）でパン、ダブルクリックでフィットに戻ります。↑/↓ キーで前後の画像へ移動でき、Shift＋↑/↓ で選択を広げられます。",
    bodyEn: "Pick an image in the list to show it in the viewer. Scroll to zoom around the cursor, drag (or Space+drag) to pan, double-click to fit. Use ↑/↓ to step between images, Shift+↑/↓ to extend the selection.",
    advanceOn: { type: "manual" },
    placement: "left",
  },
  {
    id: "develop-heatmap",
    targetSelector: '[data-tutorial-step="develop-heatmap-toggle"]',
    requireTab: "develop",
    titleJa: "ヒートマップの読み方",
    titleEn: "Reading the heatmap",
    bodyJa: "「H」ボタンで原画像⇔ヒートマップを切り替えます（ズームは維持されるので見比べに便利）。色は 青＝OK 水準、白＝判定しきい値ちょうど、朱＝しきい値超え（NG）。OK 画像はほぼ全面が青になり、欠陥箇所だけが朱く浮き上がるのが理想です。",
    bodyEn: "The H button toggles between the original image and the heatmap (zoom is kept, so comparing is easy). Colors: blue = OK level, white = exactly at the verdict threshold, vermilion = above threshold (NG). Ideally an OK image is almost entirely blue and only defects glow vermilion.",
    advanceOn: { type: "manual" },
    placement: "left",
  },
  {
    id: "develop-info",
    targetSelector: '[data-tutorial-step="develop-info"]',
    requireTab: "develop",
    titleJa: "NG度 0-100",
    titleEn: "NG degree 0-100",
    bodyJa: "画像情報パネルの大きな数字が「NG度」。0＝OK 画像の標準的なスコア、50＝判定しきい値ちょうど、100＝ヒートマップが飽和する水準です。50 を超えると NG 側。ヒートマップの色と同じ物差しなので「赤く見える＝数字が高い」が常に一致します。下には top-k などの生スコアも表示されます。",
    bodyEn: "The big number in the info panel is the NG degree: 0 = a typical OK image, 50 = exactly at the verdict threshold, 100 = where the heatmap saturates. Above 50 means NG. It shares the heatmap's scale, so \"looks red\" and \"scores high\" always agree. Raw scores (top-k, etc.) are listed below.",
    advanceOn: { type: "manual" },
    placement: "left",
  },
  // --- Intermediate+: separation check ---
  // Marking is covered by the bank-label step, on the tab it lives on.
  {
    id: "develop-separation",
    targetSelector: '[data-tutorial-step="develop-separation"]',
    requireTab: "develop",
    titleJa: "分離度評価としきい値",
    titleEn: "Separation check & threshold",
    bodyJa: "OK と NG のスコア分布をヒストグラムで重ね、分離の良さを AUROC で示します。完全分離なら取り違えゼロで判定可能。しきい値は自動提案されますが、スライダーで手動調整でき（↺ で自動値に戻す）、しきい値以下の NG＝見逃し、しきい値超えの OK＝過検知として表示されます。",
    bodyEn: "Overlays the OK and NG score histograms and reports separation quality as AUROC — full separation means zero mix-ups. A threshold is auto-suggested, but the slider lets you adjust it (↺ restores the suggestion); NG below the threshold shows as misses, OK above it as false positives.",
    advanceOn: { type: "manual" },
    placement: "bottom",
    modes: ["intermediate", "expert"],
  },
  // --- Expert: α boost, feature map, bank management ---
  {
    id: "develop-alpha",
    targetSelector: '[data-tutorial-step="develop-alpha"]',
    requireTab: "develop",
    titleJa: "α ブースト",
    titleEn: "α boost",
    bodyJa: "α ブーストは、NG 見本（マークした領域、無ければ自動選出）に近いパッチのスコアを底上げする仕組みです。OK との差が小さい微妙な欠陥を持ち上げて分離を改善したいときに使います。0 で無効。上げすぎると過検知が増えるので、分離度評価を見ながら調整してください。",
    bodyEn: "The α boost raises the score of patches similar to your NG exemplars (marked regions, or auto-picked ones). Use it to lift subtle defects that barely differ from OK and improve separation. 0 disables it; too high increases false positives, so tune it while watching the separation check.",
    advanceOn: { type: "manual" },
    placement: "bottom",
    modes: ["expert"],
  },
  {
    id: "develop-map",
    targetSelector: '[data-tutorial-step="develop-map"]',
    // The map lives behind a toggle; without this the spotlight lands on a
    // figure that has not been switched on yet.
    onEnterClickSelector: '[data-tutorial-step="develop-map-toggle"]',
    requireTab: "develop",
    titleJa: "特徴分離マップ",
    titleEn: "Feature-separation map",
    bodyJa: "バンク内の特徴を 2 次元に投影した散布図です。OK と NG の点群が分かれていれば判定しやすいデータ、混ざっていれば追加の教え込みや α 調整が必要なサイン。粒度（パッチ/画像）や色分け（種別/スコア）を切り替えられ、評価後は過検知・見逃しの画像に印が付きます。",
    bodyEn: "A 2-D projection of the bank's features. Separated OK/NG clusters mean easy verdicts; mixed clusters signal you need more teaching or α tuning. Switch granularity (patch/image) and coloring (tier/score); after an evaluation, false-positive and missed images get marked on the map.",
    advanceOn: { type: "manual" },
    placement: "bottom",
    modes: ["expert"],
  },
  // --- Operator tab: run inspections & save the verdict config ---
  {
    id: "operator-tab",
    targetSelector: '[data-tutorial-step="operator-tab"]',
    titleJa: "検査タブへ",
    titleEn: "Switch to the Inspection tab",
    bodyJa: "仕上げは「検査」タブ。現場のオペレーター向けの運用画面で、「学習」タブで作ったバンクとしきい値を使って新しい画像を判定します。",
    bodyEn: "Finish on the Inspection tab — the operator-facing screen that judges new images using the bank and threshold you built on the Teach tab.",
    advanceOn: { type: "tabEnter", tabId: "operator" },
    placement: "bottom",
  },
  {
    id: "operator-drop",
    targetSelector: '[data-tutorial-step="operator-drop"]',
    requireTab: "operator",
    titleJa: "画像をドロップして判定",
    titleEn: "Drop an image to judge it",
    bodyJa: "検査したい画像をドロップ（またはクリックで選択）すると即採点され、ヒートマップと OK / NG の判定が表示されます。スコアがしきい値を超えたら NG です。",
    bodyEn: "Drop an image (or click to pick one) and it's scored immediately, showing the heatmap and an OK / NG verdict. A score above the threshold means NG.",
    advanceOn: { type: "manual" },
    placement: "right",
  },
  {
    id: "operator-save",
    targetSelector: '[data-tutorial-step="operator-save"]',
    requireTab: "operator",
    titleJa: "判定設定を保存",
    titleEn: "Save the verdict config",
    bodyJa: "判定しきい値は分離度評価の提案値で初期化され、ここで手動調整もできます。「判定設定を保存」でしきい値と α がバンクに保存され、バンクの書き出しパッケージにも含まれます。これで学習から運用までの一巡が完了です。",
    bodyEn: "The verdict threshold initializes from the separation check's suggestion and can be adjusted here. Save verdict config stores the threshold and α into the bank — included in the bank's export package. That completes the full loop from training to operation.",
    advanceOn: { type: "manual" },
    placement: "left",
  },
  {
    id: "done",
    targetSelector: null,
    titleJa: "チュートリアル完了！",
    titleEn: "Tutorial complete!",
    bodyJa: "お疲れさまでした。ヘッダーの▶︎ボタンでいつでもこのチュートリアルを再生できます。説明モードを ON にすると、ボタンにカーソルを合わせるだけで機能説明が表示されます。",
    bodyEn: "Nicely done. Replay this tutorial any time via the ▶︎ button in the header. Enable description mode for hover-to-read explanations on every button.",
    advanceOn: { type: "manual" },
    placement: "center",
  },
];

export function getStepsForMode(mode: TutorialMode): TutorialStep[] {
  return TUTORIAL_STEPS.filter((s) => !s.modes || s.modes.includes(mode));
}
