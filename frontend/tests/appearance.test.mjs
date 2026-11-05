import { after, test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdirSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import ts from 'typescript'

// 外观设置的读取、校验、写入 <html> 和快捷键步进；用最小的 DOM / localStorage 替身跑真实模块
const cache = new URL('../node_modules/.cache/ariadne-tests/', import.meta.url)
mkdirSync(cache, { recursive: true })
const compiled = new URL(`appearance-${process.pid}.mjs`, cache)
const src = readFileSync(new URL('../src/lib/appearance.ts', import.meta.url), 'utf8').replace("import { useSyncExternalStore } from 'react'", 'const useSyncExternalStore = () => {}')
writeFileSync(compiled, ts.transpileModule(src, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText)
after(() => unlinkSync(compiled))

const store = new Map([['aria.appearance', JSON.stringify({ fontSize: 99, lineHeight: 'huge', fontFamily: 'serif', width: 'wide', pdfZoom: 2, indent: 'yes' })]])
globalThis.localStorage = { getItem: (k) => store.get(k) ?? null, setItem: (k, v) => store.set(k, v) }
const vars = new Map()
const classes = new Set()
globalThis.document = {
  documentElement: { style: { setProperty: (k, v) => vars.set(k, v) }, classList: { toggle: (c, on) => (on ? classes.add(c) : classes.delete(c)) } },
}
globalThis.window = { addEventListener() {} }

const m = await import(compiled.href)

test('读取时校验：越界或无效的值回到默认，合法的保留', () => {
  const a = m.getAppearance()
  assert.equal(a.fontSize, 20) // 99 夹到上限
  assert.equal(a.lineHeight, 'normal')
  assert.equal(a.fontFamily, 'serif')
  assert.equal(a.width, 'wide')
  assert.equal(a.pdfZoom, 'fit')
  assert.equal(a.indent, false)
})

test('写入 <html> 的变量和专注模式 class', () => {
  m.setAppearance({ fontSize: 17, lineHeight: 'loose', indent: true, focusMode: true })
  assert.equal(vars.get('--paper-size'), '17px')
  assert.equal(vars.get('--paper-line'), '2.2')
  assert.equal(vars.get('--paper-indent'), '2em')
  assert.match(vars.get('--paper-font'), /SimSun/)
  assert.ok(classes.has('focus-mode'))
  assert.equal(JSON.parse(store.get('aria.appearance')).fontSize, 17)
  m.setAppearance({ focusMode: false })
  assert.ok(!classes.has('focus-mode'))
})

test('快捷键步进：到头时返回 null，0 回到默认', () => {
  m.setAppearance({ fontSize: 19 })
  assert.equal(m.stepFontSize(1), 20)
  assert.equal(m.stepFontSize(1), null)
  assert.equal(m.getAppearance().fontSize, 20)
  assert.equal(m.stepFontSize(0), 16)
  m.setAppearance({ fontSize: 14 })
  assert.equal(m.stepFontSize(-1), null)
})

test('恢复默认', () => {
  m.resetAppearance()
  assert.deepEqual(m.getAppearance(), m.DEFAULT_APPEARANCE)
})
