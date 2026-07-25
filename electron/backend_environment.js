const fs = require('fs');
const path = require('path');

function existingFile(candidate) {
  if (!candidate) return null;
  const resolved = path.resolve(candidate);
  try {
    return fs.statSync(resolved).isFile() ? resolved : null;
  } catch (_) {
    return null;
  }
}

function resolveBackendEnvironmentFile({
  env = process.env,
  isPackaged = false,
  userDataPath = '',
  executablePath = process.execPath,
  appImagePath = process.env.APPIMAGE || '',
  projectRoot = path.resolve(__dirname, '..'),
} = {}) {
  const explicit = String(env.SELENE_ENV_FILE || '').trim();
  if (explicit) {
    return existingFile(explicit) || explicit;
  }

  const candidates = [];
  if (isPackaged) {
    // User-owned desktop configuration is the stable location across app
    // upgrades. It is read only by the Python backend.
    if (userDataPath) candidates.push(path.join(userDataPath, '.env'));

    // Portable builds may keep their configuration beside the executable.
    if (appImagePath) {
      const appImageDirectory = path.dirname(path.resolve(appImagePath));
      candidates.push(path.join(appImageDirectory, '.env'));

      // Local release artifacts live in <project>/dist-electron. Let the
      // packaged app launched from that directory reuse <project>/.env.
      if (path.basename(appImageDirectory) === 'dist-electron') {
        candidates.push(path.join(appImageDirectory, '..', '.env'));
      }
    }
    if (executablePath) {
      const executableDirectory = path.dirname(path.resolve(executablePath));
      candidates.push(path.join(executableDirectory, '.env'));
      const artifactDirectory = path.dirname(executableDirectory);
      if (
        path.basename(artifactDirectory) === 'dist-electron'
        && path.basename(executableDirectory).endsWith('-unpacked')
      ) {
        candidates.push(path.join(artifactDirectory, '..', '.env'));
      }
    }
  } else {
    candidates.push(path.join(projectRoot, '.env'));
  }

  for (const candidate of candidates) {
    const found = existingFile(candidate);
    if (found) return found;
  }
  return null;
}

module.exports = {
  resolveBackendEnvironmentFile,
};
