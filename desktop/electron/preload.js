// 预加载脚本：只向页面暴露「界面缩放」这一组接口（设置 → 外观里用），其余能力不开放
const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('ariadneDesktop', {
  getZoom: () => ipcRenderer.invoke('zoom:get'),
  setZoom: (factor) => ipcRenderer.invoke('zoom:set', factor),
  /** Ctrl + = / − / 0 改了缩放时通知页面，设置页上的选中项跟着变；返回取消监听的函数 */
  onZoom: (cb) => {
    const listener = (_e, factor) => cb(factor)
    ipcRenderer.on('zoom:changed', listener)
    return () => ipcRenderer.removeListener('zoom:changed', listener)
  },
})
