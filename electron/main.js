const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

function createWindow(port) {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    },
    icon: path.join(__dirname, '../agent/static/favicon.png')
  });

  mainWindow.loadURL(`http://127.0.0.1:${port}`);
  mainWindow.setMenuBarVisibility(false);

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

function startPythonBackend() {
  const isWindows = process.platform === 'win32';
  const backendExecName = isWindows ? 'selene-backend.exe' : 'selene-backend';

  if (app.isPackaged) {
    // In production, the executable is inside resources/
    const backendPath = path.join(process.resourcesPath, backendExecName);
    console.log(`Starting backend executable from ${backendPath}...`);
    pythonProcess = spawn(backendPath, ['--no-browser']);
  } else {
    // Development must use source so frontend edits are never hidden by a stale dist build.
    const sourceBackend = path.join(__dirname, '../main.py');
    console.log(`Starting development backend from ${sourceBackend}...`);
    pythonProcess = spawn('python3', [sourceBackend, '--no-browser']);
  }

  pythonProcess.stdout.on('data', (data) => {
    const output = data.toString();
    console.log(`[Backend stdout] ${output}`);
    
    // Look for the ELECTRON_PORT emitted by the backend
    const match = output.match(/ELECTRON_PORT:(\d+)/);
    if (match && !mainWindow) {
      const port = match[1];
      createWindow(port);
    }
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Backend stderr] ${data.toString()}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Backend process exited with code ${code}`);
  });
}

app.on('ready', startPythonBackend);

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('quit', () => {
  // Ensure the python backend is killed when the electron app closes
  if (pythonProcess) {
    pythonProcess.kill();
  }
});

app.on('activate', function () {
  if (mainWindow === null && pythonProcess) {
    // If reactivated on macOS and window is closed, just wait for backend to emit port again? 
    // Usually we just recreate it if we stored the port.
    // For simplicity, we assume backend keeps running and we can just hit its default port if we tracked it.
  }
});
