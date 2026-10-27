#!/usr/bin/env node
// PostToolUse hook for the Edit and Write tools.
// Runs `ruff format` on the edited file when it is a .py file and the project
// venv has ruff. Never fails the tool call: any problem is silently skipped.

import { existsSync, readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join } from 'node:path';

try {
  const input = JSON.parse(readFileSync(0, 'utf8'));
  const file = input?.tool_input?.file_path;
  const root = process.env.CLAUDE_PROJECT_DIR || input?.cwd || process.cwd();
  const python = join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');

  if (typeof file === 'string' && /\.pyi?$/i.test(file) && existsSync(file) && existsSync(python)) {
    // --force-exclude: honour extend-exclude (vendored third_party code) even for an explicit path.
    spawnSync(python, ['-m', 'ruff', 'format', '--quiet', '--force-exclude', file], {
      cwd: root,
      stdio: 'ignore',
      timeout: 15000,
    });
  }
} catch {
  // Formatting is best effort.
}
process.exit(0);
