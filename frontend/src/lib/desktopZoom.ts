/**
 * 桌面版的界面缩放：通过 Electron 预加载脚本（desktop/electron/preload.js）调用 Chromium 自带的页面缩放。
 * 网页版没有这个接口（页面不能改浏览器的缩放），设置页会提示用浏览器的 Ctrl + = / −。
 */
import { useEffect, useState } from 'react'

interface DesktopBridge {
  getZoom(): Promise<number>
  setZoom(factor: number): Promise<number>
  onZoom(cb: (factor: number) => void): () => void
}

const bridge = (): DesktopBridge | undefined => (window as unknown as { ariadneDesktop?: DesktopBridge }).ariadneDesktop

export const DESKTOP_ZOOMS = [0.9, 1, 1.1, 1.25, 1.5]

export function useDesktopZoom(): { available: boolean; zoom: number; setZoom: (f: number) => void } {
  const b = bridge()
  const [zoom, setLocal] = useState(1)
  useEffect(() => {
    if (!b) return
    b.getZoom().then(setLocal).catch(() => undefined)
    return b.onZoom(setLocal)
  }, [b])
  return {
    available: !!b,
    zoom,
    setZoom: (f: number) => {
      if (!b) return
      setLocal(f)
      b.setZoom(f).then(setLocal).catch(() => undefined)
    },
  }
}
