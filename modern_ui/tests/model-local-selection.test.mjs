import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import test from 'node:test'
import ts from 'typescript'

// Execute the actual component handler; native picker/backend calls are simulated.
function handler(mode = 'krea2_fz', response = { ok: true }) {
  const source = readFileSync(new URL('../src/components/ModelDownloadDialog.vue', import.meta.url), 'utf8')
  const script = source.match(/<script setup[^>]*>([\s\S]*?)<\/script>/)[1]
  const ast = ts.createSourceFile('ModelDownloadDialog.ts', script, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
  const fn = ast.statements.find(node => ts.isFunctionDeclaration(node) && node.name?.text === 'selectLocal')
  assert.ok(fn)
  const code = ts.transpileModule(fn.getText(ast), { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
  const calls = []
  const invoke = name => async (...args) => { calls.push([name, ...args]); return response }
  const context = {
    window: { pywebview: { api: { select_krea2_component: invoke('krea2'), select_h3_component: invoke('h3'), select_zimage_fz_component: invoke('zimage'), select_moss_component: invoke('moss') } } },
    props: { mode, projectName: 'demo' }, busy: { value: false }, selecting: { value: false }, error: { value: '' },
    loadList: async () => { calls.push(['reload']) }, emit: (...args) => calls.push(['emit', ...args]),
  }
  vm.runInNewContext(code + '\nglobalThis.invoke = selectLocal', context)
  return { context, calls }
}

test('Krea 2 select/reset/disable use its own project component API', async () => {
  for (const action of ['select', 'default', 'disable']) {
    const { context, calls } = handler()
    await context.invoke({ key: action === 'disable' ? 'turbo' : 'te' }, action)
    assert.deepEqual(calls[0], ['krea2', 'demo', action === 'disable' ? 'turbo' : 'te', action])
    assert.ok(calls.some(call => call[0] === 'reload'))
    assert.ok(calls.some(call => call[0] === 'emit' && call[1] === 'changed'))
    assert.equal(context.selecting.value, false)
  }
})

test('H3 local selection keeps the original API', async () => {
  const { context, calls } = handler('h3_fz')
  await context.invoke({ key: 'dit' })
  assert.deepEqual(calls[0], ['h3', 'demo', 'dit', 'select'])
})

test('cancelled picker and invalid component do not emit a changed event', async () => {
  for (const response of [{ ok: true, cancelled: true }, { ok: false, error: 'wrong model' }]) {
    const { context, calls } = handler('krea2_fz', response)
    await context.invoke({ key: 'raw' })
    assert.equal(calls.length, 1)
    assert.equal(context.selecting.value, false)
    assert.equal(context.error.value, response.error || '')
  }
})

test('an active task prevents opening the component picker', async () => {
  const { context, calls } = handler()
  context.busy.value = true
  await context.invoke({ key: 'raw' })
  assert.equal(calls.length, 0)
})


test('MOSS and Z-Image route local component actions to their own APIs', async () => {
  for (const [mode, api] of [['moss_fz', 'moss'], ['zimage_fz', 'zimage']]) {
    for (const action of ['select', 'default']) {
      const { context, calls } = handler(mode)
      await context.invoke({ key: 'dit' }, action)
      assert.deepEqual(calls[0], [api, 'demo', 'dit', action])
      assert.ok(calls.some(call => call[0] === 'reload'))
    }
  }
})

test('MOSS cancelled picker preserves the current selection', async () => {
  const { context, calls } = handler('moss_fz', { ok: true, cancelled: true })
  await context.invoke({ key: 'te' })
  assert.equal(calls.length, 1)
  assert.equal(context.selecting.value, false)
  assert.equal(context.error.value, '')
})
