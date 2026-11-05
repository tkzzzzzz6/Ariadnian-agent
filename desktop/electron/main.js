// Ariadne 桌面版 · Electron 主进程
//
// 启动流程：申请单实例锁 → 显示「正在启动」窗口 → 分配空闲端口 → 启动 Python 后端
// （打包版是 resources/server/ariadne-server，开发时用 uv 运行 desktop/server.py）
// → 等 /api/health 就绪 → 加载页面。关闭窗口时结束后端，不留后台进程。
const { app, BrowserWindow, Menu, ipcMain, shell } = require('electron')
const { spawn, spawnSync } = require('node:child_process')
const fs = require('node:fs')
const http = require('node:http')
const net = require('node:net')
const path = require('node:path')

const isWin = process.platform === 'win32'
const isMac = process.platform === 'darwin'
const DEV = !app.isPackaged
const REPO = path.resolve(__dirname, '..', '..')

// 文献库、论文、设置放在 %APPDATA%\Ariadne（macOS: ~/Library/Application Support/Ariadne），
// Electron 自己的缓存放到其中的 electron 子目录，互不干扰
const DATA_DIR = process.env.ARIA_DATA_DIR || path.join(app.getPath('appData'), 'Ariadne')
app.setPath('userData', path.join(DATA_DIR, 'electron'))
fs.mkdirSync(path.join(DATA_DIR, 'logs'), { recursive: true })
const LOG_FILE = path.join(DATA_DIR, 'logs', 'electron.log')
const logStream = fs.createWriteStream(LOG_FILE, { flags: 'a' })
const log = (...a) => {
  const line = `${new Date().toISOString()} ${a.join(' ')}\n`
  logStream.write(line)
  if (DEV) process.stdout.write(line)
}

let win = null
let server = null
let quitting = false

// 界面缩放：Chromium 自带的页面缩放（和浏览器里按 Ctrl + = 一样，弹出菜单、提示的位置都准确），
// 记在数据目录里，下次启动沿用。设置 → 外观 通过 preload.js 调整，Ctrl + = / − / 0 也可以
const ZOOM_FILE = path.join(DATA_DIR, 'electron', 'zoom.json')
const ZOOM_STEPS = [0.8, 0.9, 1, 1.1, 1.25, 1.5]
const clampZoom = (f) => (Number.isFinite(f) ? Math.min(1.5, Math.max(0.8, Math.round(f * 100) / 100)) : 1)
let zoom = 1
try {
  zoom = clampZoom(JSON.parse(fs.readFileSync(ZOOM_FILE, 'utf8')).factor)
} catch {
  /* 第一次启动没有记录 */
}
function setZoom(f) {
  zoom = clampZoom(f)
  if (win && !win.isDestroyed()) {
    win.webContents.setZoomFactor(zoom)
    win.webContents.send('zoom:changed', zoom)
  }
  try {
    fs.writeFileSync(ZOOM_FILE, JSON.stringify({ factor: zoom }))
  } catch (e) {
    log(`保存缩放失败：${e.message}`)
  }
  return zoom
}
ipcMain.handle('zoom:get', () => zoom)
ipcMain.handle('zoom:set', (_e, f) => setZoom(Number(f)))

if (!app.requestSingleInstanceLock()) {
  app.quit() // 已经有一个在运行：交给它把窗口调到前台
} else {
  app.on('second-instance', () => {
    if (!win) return
    if (win.isMinimized()) win.restore()
    win.show()
    win.focus()
  })
  app.whenReady().then(main)
}

function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer()
    s.unref()
    s.on('error', reject)
    s.listen(0, '127.0.0.1', () => {
      const { port } = s.address()
      s.close(() => resolve(port))
    })
  })
}

function healthy(port) {
  return new Promise((resolve) => {
    const req = http.get({ host: '127.0.0.1', port, path: '/api/health', timeout: 1000 }, (res) => {
      res.resume()
      resolve(res.statusCode === 200)
    })
    req.on('error', () => resolve(false))
    req.on('timeout', () => {
      req.destroy()
      resolve(false)
    })
  })
}

function startServer(port) {
  const env = { ...process.env, ARIA_DATA_DIR: DATA_DIR, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' }
  let child
  if (DEV) {
    // 开发模式：直接运行源码，改了后端代码重启窗口即可生效
    child = spawn('uv', ['run', 'python', '../desktop/server.py', '--port', String(port)], {
      cwd: path.join(REPO, 'backend'), env, windowsHide: true, shell: isWin,
    })
  } else {
    const exe = path.join(process.resourcesPath, 'server', isWin ? 'ariadne-server.exe' : 'ariadne-server')
    child = spawn(exe, ['--port', String(port)], { env, windowsHide: true })
  }
  child.stdout.on('data', (d) => logStream.write(d))
  child.stderr.on('data', (d) => logStream.write(d))
  child.on('exit', (code) => {
    log(`后端已退出，code=${code}`)
    server = null
    if (!quitting) showError(`后端意外退出（code ${code}）`)
  })
  return child
}

function stopServer() {
  if (!server) return
  const pid = server.pid
  // Windows 上开发模式是 uv → python 两层进程，要连同子进程一起结束
  if (isWin) spawnSync('taskkill', ['/pid', String(pid), '/T', '/F'], { windowsHide: true })
  else server.kill('SIGTERM')
  server = null
}

function showError(message) {
  if (!win || win.isDestroyed()) return
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c])
  const html = `<!doctype html><meta charset="utf-8"><body style="font-family:'Segoe UI','Microsoft YaHei',sans-serif;padding:40px;color:#242424">
<h2 style="color:#a4262c">启动失败</h2><p>${esc(message)}</p>
<p style="color:#616161">日志：<code>${esc(LOG_FILE)}</code><br>以及 <code>${esc(path.join(DATA_DIR, 'logs', 'server.log'))}</code></p></body>`
  win.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`)
}

function buildMenu() {
  if (!isMac) {
    Menu.setApplicationMenu(null) // Windows / Linux 不要菜单栏，界面更干净
    return
  }
  // macOS 需要「编辑」菜单，否则 ⌘C / ⌘V 不起作用
  Menu.setApplicationMenu(Menu.buildFromTemplate([{ role: 'appMenu' }, { role: 'editMenu' }, { role: 'viewMenu' }, { role: 'windowMenu' }]))
}

async function main() {
  log(`启动 Ariadne ${app.getVersion()}（${DEV ? '开发模式' : '打包版'}），数据目录 ${DATA_DIR}`)
  buildMenu()
  win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1000,
    minHeight: 640,
    title: 'Ariadne · 研究报告助手',
    icon: path.join(__dirname, 'build', 'icon.png'),
    backgroundColor: '#ffffff',
    show: false,
    webPreferences: { contextIsolation: true, sandbox: true, spellcheck: false, preload: path.join(__dirname, 'preload.js') },
  })
  // 每次加载页面（启动、刷新）都套用保存的缩放
  win.webContents.on('did-finish-load', () => win && win.webContents.setZoomFactor(zoom))
  // Ctrl + = / − / 0 缩放界面（Windows / Linux 没有菜单栏，这些快捷键要自己处理）
  win.webContents.on('before-input-event', (e, input) => {
    if (input.type !== 'keyDown' || !(input.control || input.meta) || input.alt) return
    const k = input.key
    if (!['=', '+', '-', '_', '0'].includes(k)) return
    e.preventDefault()
    if (k === '0') return setZoom(1)
    const down = k === '-' || k === '_'
    // 下一档：比当前大（小）的第一档；已经到头就不变
    const next = down ? [...ZOOM_STEPS].reverse().find((z) => z < zoom - 0.001) : ZOOM_STEPS.find((z) => z > zoom + 0.001)
    setZoom(next ?? zoom)
  })
  win.once('ready-to-show', () => win.show())
  win.on('closed', () => {
    win = null
  })

  // 论文链接、官网价格页等外部地址用系统浏览器打开，窗口里只显示 Ariadne 自己的页面
  const isLocal = (url) => url.startsWith('http://127.0.0.1:') || url.startsWith('file:') || url.startsWith('data:')
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/.test(url)) shell.openExternal(url)
    return { action: 'deny' }
  })
  win.webContents.on('will-navigate', (e, url) => {
    if (!isLocal(url)) {
      e.preventDefault()
      shell.openExternal(url)
    }
  })
  // 导出文件：Electron 默认会弹出「另存为」对话框，这里只记日志
  win.webContents.session.on('will-download', (_e, item) => log(`下载 ${item.getFilename()}`))

  await win.loadFile(path.join(__dirname, 'loading.html'))

  try {
    const port = await freePort()
    server = startServer(port)
    const deadline = Date.now() + 90_000
    while (!(await healthy(port))) {
      if (!server) return // 已经在 exit 回调里显示了错误
      if (Date.now() > deadline) throw new Error('后端 90 秒内没有就绪')
      await new Promise((r) => setTimeout(r, 200))
    }
    log(`已就绪：http://127.0.0.1:${port}`)
    if (win) await win.loadURL(`http://127.0.0.1:${port}/`)
  } catch (e) {
    log(`启动失败：${e.stack || e}`)
    showError(e.message || String(e))
  }
}

app.on('window-all-closed', () => app.quit())
app.on('before-quit', () => {
  quitting = true
  stopServer()
})
process.on('exit', stopServer)
