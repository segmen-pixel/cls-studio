// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// The import dialog. The store keeps a display NAME per entry and the blob on
// disk is {entry_id}.{ext}, so the name is the only thing that tells one lot
// from another in the labelling grid -- and 200 files called image_001.png
// from four different lots is unrecoverable once ingested. This is where the
// name is decided, before the encoder runs.
import React, { useCallback, useRef, useState } from "react";

import { useI18n } from "../i18n";
import { ACCENT, BORDER, DANGER, INK, MUTED, PANEL_2, RULE, TYPE } from "../ui/tokens";

export type BankImportDialogProps = {
  /** Mounted only while true; all staged state clears on every close. */
  open: boolean;
  onClose: () => void;
  /** Handed the already-renamed files (zips unrenamed). Dialog closes first. */
  onImport: (files: File[]) => void;
};

// What the server can actually take: cv2.imdecode formats, plus a zip, which
// goes to /store/ingest_zip instead. GIF is out on purpose -- imdecode returns
// None for it, so accepting one only produces a "could not be decoded" toast.
const STAGE_RE = /\.(png|jpe?g|bmp|tiff?|webp|zip)$/i;
const ZIP_RE = /\.zip$/i;
const ACCEPT = "image/*,.png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp,.zip";

const isZip = (f: File) => ZIP_RE.test(f.name);
const stageFilter = (f: File) =>
  STAGE_RE.test(f.name) || (f.type.startsWith("image/") && f.type !== "image/gif");

export default function BankImportDialog({ open, onClose, onImport }: BankImportDialogProps) {
  const { t } = useI18n();
  const [prefix, setPrefix] = useState("");
  const [suffix, setSuffix] = useState("");
  const [pending, setPending] = useState<File[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const dragDepth = useRef(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const renameFile = useCallback((file: File): File => {
    // A zip's own name is discarded by the server and the member names inside
    // are untouched, so renaming it would only mislead the preview.
    if (isZip(file)) return file;
    const dot = file.name.lastIndexOf(".");
    const base = dot > 0 ? file.name.slice(0, dot) : file.name;
    const ext = dot > 0 ? file.name.slice(dot) : "";
    return new File([file], `${prefix}${base}${suffix}${ext}`, { type: file.type });
  }, [prefix, suffix]);

  const addFiles = useCallback((fileList: FileList | File[], folderName?: string) => {
    const arr = Array.from(fileList).filter(stageFilter);
    if (arr.length === 0) return;
    // Append, never replace: two successive drops accumulate.
    if (folderName) {
      setPending((prev) => [
        ...prev,
        ...arr.map((f) => new File([f], `${folderName}_${f.name}`, { type: f.type })),
      ]);
    } else {
      setPending((prev) => [...prev, ...arr]);
    }
  }, []);

  // Chrome returns at most 100 entries per readEntries call, so the loop is
  // load-bearing rather than defensive: without it a folder of 400 images
  // stages exactly 100 and looks like it worked.
  const readDirectoryEntries = useCallback(
    async (entry: FileSystemDirectoryEntry, pathPrefix = ""): Promise<File[]> => {
      const reader = entry.createReader();
      const files: File[] = [];
      const readBatch = (): Promise<FileSystemEntry[]> =>
        new Promise((resolve, reject) => reader.readEntries(resolve, reject));
      let batch: FileSystemEntry[];
      do {
        batch = await readBatch();
        for (const child of batch) {
          if (child.isFile) {
            const file = await new Promise<File>((resolve, reject) =>
              (child as FileSystemFileEntry).file(resolve, reject),
            );
            if (!stageFilter(file)) continue;
            // Sub-path joined with "_", never "/": the name has to stay one
            // path segment or the grid renders a broken label, and the server
            // truncates at the last "/" anyway.
            files.push(pathPrefix
              ? new File([file], `${pathPrefix}${file.name}`, { type: file.type })
              : file);
          } else if (child.isDirectory) {
            files.push(...await readDirectoryEntries(
              child as FileSystemDirectoryEntry, `${pathPrefix}${child.name}_`,
            ));
          }
        }
      } while (batch.length > 0);
      return files;
    },
    [],
  );

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    const items = e.dataTransfer.items;
    if (items && items.length > 0) {
      const entries: FileSystemEntry[] = [];
      for (let i = 0; i < items.length; i++) {
        const entry = items[i].webkitGetAsEntry?.();
        if (entry) entries.push(entry);
      }
      const dirs = entries.filter((x) => x.isDirectory) as FileSystemDirectoryEntry[];
      if (dirs.length > 0) {
        const all: File[] = [];
        for (const dir of dirs) {
          const found = await readDirectoryEntries(dir);
          // One folder: its name becomes the prefix for every file, so it is
          // applied once below. Several: prepend each here, or two lots merge.
          if (dirs.length > 1) {
            for (const f of found) all.push(new File([f], `${dir.name}_${f.name}`, { type: f.type }));
          } else {
            all.push(...found);
          }
        }
        for (const fe of entries.filter((x) => x.isFile) as FileSystemFileEntry[]) {
          const file = await new Promise<File>((resolve, reject) => fe.file(resolve, reject));
          if (stageFilter(file)) all.push(file);
        }
        addFiles(all, dirs.length === 1 ? dirs[0].name : undefined);
        return;
      }
    }
    if (e.dataTransfer.files.length > 0) addFiles(e.dataTransfer.files);
  }, [addFiles, readDirectoryEntries]);

  const removeFile = useCallback((idx: number) => {
    setPending((prev) => prev.filter((_, i) => i !== idx));
  }, []);

  const reset = useCallback(() => {
    setPending([]);
    setPrefix("");
    setSuffix("");
    dragDepth.current = 0;
    setDragOver(false);
  }, []);

  const handleClose = useCallback(() => { reset(); onClose(); }, [reset, onClose]);

  const handleSubmit = useCallback(() => {
    if (pending.length === 0) return;
    const renamed = pending.map(renameFile);
    onImport(renamed);
    reset();
    onClose();
  }, [pending, renameFile, onImport, reset, onClose]);

  if (!open) return null;

  const firstImage = pending.find((f) => !isZip(f));
  const previewName = firstImage
    ? renameFile(firstImage).name
    : `${prefix}example${suffix}.png`;

  const fieldLabel: React.CSSProperties = {
    // display:flex and marginBottom:0 are overrides, not decoration:
    // .settings-panel label is block with a 4px margin (projects.css:409).
    display: "flex", flexDirection: "column", flex: 1, gap: 6,
    marginBottom: 0, fontSize: TYPE.md, color: MUTED,
  };
  const fieldInput: React.CSSProperties = {
    padding: "6px 9px", border: RULE, borderRadius: 6,
    background: PANEL_2, color: INK, fontSize: TYPE.md,
  };
  const nameCell: React.CSSProperties = {
    flex: 1, minWidth: 0, overflow: "hidden",
    textOverflow: "ellipsis", whiteSpace: "nowrap",
  };

  return (
    // Every one of the four drag events is stopped here. The dialog renders
    // inside .bank-tab, which carries whole-tab drop handlers, and React
    // synthetic events bubble along the React tree -- so an unguarded drop on
    // the modal imports the files UNRENAMED behind it, and an unguarded
    // dragenter strands the tab's depth counter with its overlay left up.
    <div
      className="settings-overlay"
      onClick={handleClose}
      onDragEnter={(e) => { e.preventDefault(); e.stopPropagation(); }}
      onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
      onDragLeave={(e) => { e.stopPropagation(); }}
      onDrop={(e) => { e.preventDefault(); e.stopPropagation(); }}
    >
      <div className="settings-panel bank-import-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2>{t("bank.importTitle")}</h2>
          <button className="ghost" onClick={handleClose}
                  data-desc={t("common.close")} data-desc-pos="bottom">×</button>
        </div>

        <p className="muted" style={{ fontSize: TYPE.md, marginTop: 0 }}>{t("bank.importHint")}</p>

        <div style={{ display: "flex", gap: 16, marginBottom: 14 }}>
          <label style={fieldLabel}>
            <span>{t("bank.importPrefix")}</span>
            <input type="text" value={prefix} placeholder="prefix_"
                   onChange={(e) => setPrefix(e.target.value)} style={fieldInput} />
          </label>
          <label style={fieldLabel}>
            <span>{t("bank.importSuffix")}</span>
            <input type="text" value={suffix} placeholder="_suffix"
                   onChange={(e) => setSuffix(e.target.value)} style={fieldInput} />
          </label>
        </div>

        <div style={{ fontSize: TYPE.md, color: MUTED, marginBottom: 14 }}>
          {t("bank.importPreview")}:{" "}
          <code style={{ background: PANEL_2, padding: "3px 8px", borderRadius: 4, fontSize: TYPE.base }}>
            {previewName}
          </code>
        </div>

        <div
          onDragEnter={(e) => { e.preventDefault(); e.stopPropagation(); dragDepth.current += 1; setDragOver(true); }}
          onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
          onDragLeave={(e) => {
            e.stopPropagation();
            dragDepth.current = Math.max(0, dragDepth.current - 1);
            if (dragDepth.current === 0) setDragOver(false);
          }}
          onDrop={(e) => {
            e.preventDefault(); e.stopPropagation();
            dragDepth.current = 0; setDragOver(false);
            void handleDrop(e);
          }}
          onClick={() => fileInputRef.current?.click()}
          style={{
            border: `2px dashed ${dragOver ? ACCENT : BORDER}`,
            borderRadius: 8, padding: "36px 20px", textAlign: "center",
            cursor: "pointer", color: MUTED, fontSize: TYPE.md,
            background: dragOver ? "rgba(34, 199, 219, 0.07)" : "transparent",
            transition: "border-color .15s, background .15s",
          }}
        >
          <input ref={fileInputRef} type="file" accept={ACCEPT} multiple hidden
                 onChange={(e) => { if (e.target.files) addFiles(e.target.files); e.target.value = ""; }} />
          <input
            ref={folderInputRef} type="file" hidden
            // @ts-expect-error webkitdirectory is not in React's type defs
            webkitdirectory=""
            onChange={(e) => {
              const picked = e.target.files;
              if (!picked || picked.length === 0) return;
              const top = (picked[0].webkitRelativePath || "").split("/")[0] || undefined;
              const renamed: File[] = [];
              for (const f of Array.from(picked)) {
                if (!stageFilter(f)) continue;
                const parts = (f.webkitRelativePath || f.name).split("/");
                // parts[0] is the top folder and is applied as the prefix, so
                // only the segments between it and the basename are folded in.
                const sub = parts.length > 2 ? `${parts.slice(1, -1).join("_")}_` : "";
                const base = parts[parts.length - 1];
                renamed.push(sub ? new File([f], `${sub}${base}`, { type: f.type }) : f);
              }
              addFiles(renamed, top);
              e.target.value = "";  // so re-picking the same folder fires again
            }}
          />
          {pending.length === 0
            ? t("bank.importDropHint")
            : t("bank.importStaged").replace("{n}", String(pending.length))}
        </div>

        <div style={{ display: "flex", justifyContent: "center", marginTop: 8 }}>
          <button
            onClick={(e) => { e.stopPropagation(); folderInputRef.current?.click(); }}
            style={{ padding: "6px 10px", borderRadius: 6, border: RULE, background: "transparent", color: INK, cursor: "pointer" }}
          >
            {t("bank.importSelectFolder")}
          </button>
        </div>

        {pending.length > 0 && (
          <div style={{ marginTop: 14, maxHeight: 240, overflowY: "auto", fontSize: TYPE.md }}>
            {pending.map((f, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 0" }}>
                <span style={{ ...nameCell, color: MUTED }}>{f.name}</span>
                {isZip(f) ? (
                  <span style={{ ...nameCell, color: MUTED }}>{t("bank.importZipRow")}</span>
                ) : (
                  <>
                    <span style={{ color: MUTED, flex: "none" }}>→</span>
                    <span style={{ ...nameCell, color: ACCENT, fontWeight: 500 }}>
                      {renameFile(f).name}
                    </span>
                  </>
                )}
                <button
                  onClick={() => removeFile(i)}
                  aria-label={t("bank.importRemoveRow")}
                  data-desc={t("bank.importRemoveRow")}
                  style={{ background: "none", border: "none", color: MUTED, cursor: "pointer", fontSize: TYPE.xl, padding: "0 4px", flex: "none" }}
                  onMouseEnter={(e) => { e.currentTarget.style.color = DANGER; }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = MUTED; }}
                >×</button>
              </div>
            ))}
          </div>
        )}

        <div style={{ display: "flex", justifyContent: "flex-end", gap: 10, marginTop: 18 }}>
          <button
            onClick={handleClose}
            style={{ padding: "6px 10px", borderRadius: 6, border: RULE, background: "transparent", color: INK, cursor: "pointer" }}
          >
            {t("common.cancel")}
          </button>
          <button
            onClick={handleSubmit}
            disabled={pending.length === 0}
            style={{
              padding: "6px 10px", fontSize: TYPE.md, borderRadius: 6, border: "none",
              background: ACCENT, color: "#fff", fontWeight: 600,
              cursor: pending.length ? "pointer" : "default",
              opacity: pending.length ? 1 : 0.5,
            }}
          >
            {t("bank.importConfirm").replace("{n}", String(pending.length))}
          </button>
        </div>
      </div>
    </div>
  );
}
