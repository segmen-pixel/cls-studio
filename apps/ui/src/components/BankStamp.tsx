// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
//
// Which bank the tab is judging with, printed over the image it is judging.
//
// Every project on this build holds exactly one bank, so "which bank" is not a
// name anybody picks -- it is what the bank was assembled FROM: a label set, a
// count per tier, a row count, and whether the labels have moved since. The
// check tab and the inspection tab both reach verdicts with it and neither
// said so anywhere, which makes "why did this score change?" unanswerable from
// the screen you are looking at.
//
// Read-only, and it keeps its last reading on a failed fetch: a blank stamp
// reads as "no bank", which is a different and much more alarming claim than
// "could not reach the server just now".
import React, { useCallback, useEffect, useState } from "react";
import { fetchAssemblyStatus, type AssemblyStatus } from "../api/cls";
import Glyph from "./Glyph";
import { TYPE, VERM } from "../ui/tokens";
import { useI18n } from "../i18n";

export default function BankStamp({ projectId, active, border }: {
  projectId: string | null;
  active: boolean;
  border: string;
}) {
  const { t } = useI18n();
  const [st, setSt] = useState<AssemblyStatus | null>(null);

  const load = useCallback(() => {
    fetchAssemblyStatus().then(setSt).catch(() => { /* keep the last stamp */ });
  }, []);

  // On activation, not on a timer: the only thing that changes this is an
  // assemble on the bank tab, and coming back here is how you get to see it.
  useEffect(() => {
    if (!active || !projectId) { return; }
    load();
  }, [active, projectId, load]);

  if (!st) return null;
  const c = st.counts ?? {};
  // The same marks the image lists use, at the same size. A stamp that
  // invented its own alphabet would be one more thing to learn.
  const tier = (t_: "normal" | "critical" | "negative", n: number, label: string) => (
    <span title={label} style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
      <Glyph tier={t_} size={12} />
      <b>{n.toLocaleString()}</b>
    </span>
  );

  return (
    <div
      style={{
        flex: "none", display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap",
        minWidth: 0, border, borderRadius: 8, padding: "5px 10px", fontSize: TYPE.base,
      }}
    >
      <span className="muted" style={{ flex: "none" }}>{t("stamp.bank")}</span>
      <b style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {st.labelset_name || "—"}
      </b>
      <span style={{ display: "inline-flex", alignItems: "center", gap: 10, flex: "none", fontVariantNumeric: "tabular-nums" }}>
        {tier("normal", c.normal ?? 0, t("develop.tier.normal"))}
        {tier("critical", c.critical ?? 0, t("develop.tier.critical"))}
        {tier("negative", c.negative ?? 0, t("develop.tier.negative"))}
      </span>
      <span className="muted" style={{ flex: "none", fontVariantNumeric: "tabular-nums" }}>
        {t("stamp.rows").replace("{n}", st.store_rows.toLocaleString())}
      </span>
      {/* The one thing that makes the numbers above a lie. */}
      {st.stale && (
        <span style={{ marginLeft: "auto", flex: "none", color: VERM, fontWeight: 600 }} title={t("bank.stale")}>
          ⚠ {t("stamp.stale")}
        </span>
      )}
    </div>
  );
}
