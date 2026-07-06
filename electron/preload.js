const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Expose safe methods if needed
});
