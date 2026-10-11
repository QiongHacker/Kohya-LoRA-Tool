import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import test from 'node:test'
import ts from 'typescript'

function actualFunction(file, name, context) {
  const script = readFileSync(new URL('../src/' + file, import.meta.url), 'utf8').match(/<script setup[^>]*>([\s\S]*?)<\/script>/)[1]
  const ast = ts.createSourceFile(file + '.ts', script, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
  const fn = ast.statements.find(node => ts.isFunctionDeclaration(node) && node.name?.text === name)
  assert.ok(fn, name)
  const code = ts.transpileModule(fn.getText(ast), { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
  vm.runInNewContext(code + '\nglobalThis.invoke = ' + name, context)
  return context.invoke
}
function workspace({ save = true, response = { ok: true }, navigate = false } = {}) {
  const calls = []
  const context = {
    krea2ComponentSaving: { value: false }, workspaceProject: { value: { name: 'demo', mode: 'krea2_fz' } }, preview: { value: false },
    window: { pywebview: { api: {
      select_krea2_encoder: async (...args) => { calls.push(['encoder', ...args]); return response },
      select_krea2_component: async (...args) => { calls.push(['file', ...args]); return response },
    } } },
    saveKohyaConfig: async patch => { calls.push(['save', patch]); if (navigate) context.workspaceProject.value = { name: 'other', mode: 'krea2_fz' }; return save },
    refreshGuideState: async () => calls.push(['refresh']), showToast: message => calls.push(['notice', message]), modelDialogOpen: { value: false },
  }
  const invoke = actualFunction('App.vue', 'runKrea2ComponentAction', context)
  return { invoke, calls, context }
}
test('workspace persists unsaved draft before changing encoder or file path', async () => {
  for (const [action, call] of [['krea2_encoder:te', ['encoder', 'demo', 'te']], ['krea2_component:raw:select', ['file', 'demo', 'raw', 'select']]]) {
    const { invoke, calls, context } = workspace()
    const patch = { params: { rank: 64 } }
    await invoke(action, patch)
    assert.deepEqual(calls.slice(0, 3), [['save', patch], call, ['refresh']])
    assert.equal(context.krea2ComponentSaving.value, false)
  }
})
test('failed draft save and project navigation abort component operations', async () => {
  for (const options of [{ save: false }, { navigate: true }]) {
    const { invoke, calls, context } = workspace(options)
    await invoke('krea2_encoder:te', { params: { rank: 64 } })
    assert.equal(calls.length, 1)
    assert.equal(context.krea2ComponentSaving.value, false)
  }
})
test('configure/download opens only after the draft is saved', async () => {
  const { invoke, calls, context } = workspace()
  await invoke('cmd_dl_krea2_models', { mode: 'krea2_fz' })
  assert.equal(calls[0][0], 'save')
  assert.equal(calls.length, 1)
  assert.equal(context.modelDialogOpen.value, true)
})
test('cancellation or backend failure never refreshes away current UI state', async () => {
  for (const response of [{ ok: true, cancelled: true }, { ok: false, error: 'cannot save' }]) {
    const { invoke, calls, context } = workspace({ response })
    await invoke('krea2_component:raw:select')
    assert.equal(calls.filter(call => call[0] === 'refresh').length, 0)
    assert.equal(context.krea2ComponentSaving.value, false)
    if (!response.ok) assert.deepEqual(calls[1], ['notice', 'cannot save'])
  }
})
test('busy and preview mode do not invoke component writes', async () => {
  const { invoke, calls, context } = workspace()
  context.krea2ComponentSaving.value = true
  await invoke('krea2_encoder:te')
  assert.equal(calls.length, 0)
  context.krea2ComponentSaving.value = false
  context.preview.value = true
  await invoke('krea2_encoder:te')
  assert.equal(calls.length, 1)
  assert.equal(calls[0][0], 'notice')
})
test('controlled encoder select stays at saved value until persistence succeeds', () => {
  const emitted = []
  const context = { props: { encoder: { choice: 'te' } }, emit: (...args) => emitted.push(args) }
  const select = actualFunction('components/Krea2EncoderSelect.vue', 'select', context)
  const event = { target: { value: 'te_fp8' } }
  select(event)
  assert.equal(event.target.value, 'te')
  assert.deepEqual(emitted, [['select', 'te_fp8']])
})
test('dialog encoder choice uses dedicated API, not file picker', async () => {
  const calls = []
  const context = { props: { projectName: 'demo' }, busy: { value: false }, selecting: { value: false }, error: { value: '' },
    window: { pywebview: { api: { select_krea2_encoder: async (...args) => { calls.push(args); return { ok: true } } } } },
    loadList: async () => calls.push(['reload']), emit: (...args) => calls.push(['emit', ...args]) }
  const invoke = actualFunction('components/ModelDownloadDialog.vue', 'selectEncoder', context)
  await invoke('auto')
  assert.deepEqual(calls[0], ['demo', 'auto'])
  assert.ok(calls.some(call => call[0] === 'reload'))
  assert.ok(calls.some(call => call[0] === 'emit' && call[1] === 'changed'))
  assert.equal(context.selecting.value, false)
})

test('failed dialog encoder save keeps previous list and does not emit changed', async () => {
  const calls = []
  const context = { props: { projectName: 'demo' }, busy: { value: false }, selecting: { value: false }, error: { value: '' },
    window: { pywebview: { api: { select_krea2_encoder: async () => ({ ok: false, error: 'cannot save' }) } } },
    loadList: async () => calls.push(['reload']), emit: (...args) => calls.push(['emit', ...args]) }
  await actualFunction('components/ModelDownloadDialog.vue', 'selectEncoder', context)('te_fp8')
  assert.equal(context.error.value,'cannot save')
  assert.equal(context.selecting.value,false)
  assert.equal(calls.length,0)
})
test('legacy version pending and save shortcuts cannot bypass component busy state', () => {
  const calls = []
  const context = { props: { componentBusy: true, desktop: true }, emit: (...args) => calls.push(args), makePatch: () => { throw Error('must not make patch while busy') } }
  for (const name of ['save','startTraining']) actualFunction('components/ModernEngineWorkspace.vue', name, context)()
  assert.equal(calls.length,0)
})
