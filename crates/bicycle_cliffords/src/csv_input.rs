// Copyright contributors to the Bicycle Architecture Compiler project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::env;
use std::path::{Path, PathBuf};

pub fn resolve_csv_path(explicit: Option<&str>, fallbacks: &[&str]) -> Result<PathBuf, String> {
    if let Some(raw) = explicit.and_then(non_empty) {
        return resolve_explicit(raw)
            .ok_or_else(|| format_not_found("explicit path", raw, explicit_candidates(raw)));
    }

    let mut tried = Vec::new();
    for fallback in fallbacks.iter().copied().filter_map(non_empty) {
        let candidates = fallback_candidates(fallback);
        for candidate in &candidates {
            tried.push(candidate.display().to_string());
            if candidate.exists() {
                return Ok(normalize(candidate.clone()));
            }
        }
    }

    Err(format!(
        "failed to resolve CSV path: no fallback candidate exists.\nsearched:\n{}",
        tried.join("\n")
    ))
}

fn resolve_explicit(raw: &str) -> Option<PathBuf> {
    let candidates = explicit_candidates(raw);
    for candidate in candidates {
        if candidate.exists() {
            return Some(normalize(candidate));
        }
    }
    None
}

fn explicit_candidates(raw: &str) -> Vec<PathBuf> {
    let mut out = vec![PathBuf::from(raw)];
    out.push(workspace_root().join(raw));
    if let Ok(cwd) = env::current_dir() {
        out.push(cwd.join(raw));
    }
    dedup_paths(out)
}

fn fallback_candidates(raw: &str) -> Vec<PathBuf> {
    let mut out = vec![workspace_root().join(raw)];
    if let Ok(cwd) = env::current_dir() {
        out.push(cwd.join(raw));
    }
    dedup_paths(out)
}

fn dedup_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for path in paths {
        if out.iter().any(|seen: &PathBuf| seen == &path) {
            continue;
        }
        out.push(path);
    }
    out
}

fn normalize(path: PathBuf) -> PathBuf {
    path.canonicalize().unwrap_or(path)
}

fn workspace_root() -> PathBuf {
    let candidate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    candidate.canonicalize().unwrap_or(candidate)
}

fn non_empty(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

fn format_not_found(kind: &str, value: &str, candidates: Vec<PathBuf>) -> String {
    let mut lines = Vec::new();
    lines.push(format!("failed to resolve CSV {} '{}'", kind, value));
    lines.push("searched:".to_string());
    for candidate in candidates {
        lines.push(candidate.display().to_string());
    }
    lines.join("\n")
}
