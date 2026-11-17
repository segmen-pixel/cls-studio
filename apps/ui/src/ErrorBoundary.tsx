// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Cls-Studio Contributors
import React from "react";
import { TYPE } from "./ui/tokens";

type Props = { children: React.ReactNode };
type State = { error: Error | null };

export default class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 32, fontFamily: "monospace", color: "#e74c3c" }}>
          <h1>Something went wrong</h1>
          <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
            {this.state.error.message}
          </pre>
          <details style={{ marginTop: 16 }}>
            <summary>Stack trace</summary>
            <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: TYPE.xs }}>
              {this.state.error.stack}
            </pre>
          </details>
          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: 24,
              padding: "8px 24px",
              fontSize: TYPE.xl,
              cursor: "pointer",
            }}
          >
            Reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
