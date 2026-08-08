// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import React from "react";

type StatusToastProps = {
  toastMsg: string;
  toastCopied: boolean;
  inferStatus: string;
  onToastClick: () => void;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
};

export default React.memo(function StatusToast({
  toastMsg, toastCopied, inferStatus,
  onToastClick, onMouseEnter, onMouseLeave,
}: StatusToastProps) {
  return (
    <div
      className={`tabs-status${/fail|error/i.test(toastMsg) ? " toast-error" : ""}${!toastMsg && inferStatus ? " training-active" : ""}`}
      onClick={onToastClick}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      style={{ cursor: toastMsg ? "pointer" : undefined }}
      title={toastMsg ? "Click to copy" : ""}
    >
      {toastCopied ? "✓ Copied" : toastMsg || (inferStatus
        ? <><span className="train-spinner" />{inferStatus}</>
        : "")}
    </div>
  );
});
