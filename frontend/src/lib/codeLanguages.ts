export const CODE_LANGUAGES = [
  ['plaintext','纯文本'], ['python','Python'], ['javascript','JavaScript'], ['typescript','TypeScript'],
  ['c','C'], ['cpp','C++'], ['java','Java'], ['go','Go'], ['rust','Rust'], ['bash','Shell / Bash'],
  ['sql','SQL'], ['json','JSON'], ['yaml','YAML'], ['xml','HTML / XML'], ['css','CSS'], ['markdown','Markdown'],
] as const
const aliases: Record<string,string> = {text:'plaintext',txt:'plaintext',plain:'plaintext',py:'python',js:'javascript',jsx:'javascript',ts:'typescript',tsx:'typescript','c++':'cpp',cc:'cpp',cxx:'cpp',sh:'bash',shell:'bash',zsh:'bash',yml:'yaml',html:'xml',htm:'xml',md:'markdown'}
export function codeLanguage(value?:string|null): string|null {
  const name=(value||'').trim().toLowerCase()
  if(!/^[a-z0-9_+.#-]{1,40}$/.test(name))return null
  return aliases[name]||name
}
export function codeLanguageLabel(value?:string|null):string {
  const language=codeLanguage(value)
  return language ? CODE_LANGUAGES.find(([id])=>id===language)?.[1]||language : '自动识别'
}
export function knownCodeLanguage(value:string):string|null {
  const normalized=codeLanguage(value)
  return CODE_LANGUAGES.some(([id])=>id===normalized)?normalized:null
}
