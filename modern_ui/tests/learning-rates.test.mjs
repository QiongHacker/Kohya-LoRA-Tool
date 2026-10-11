import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import test from 'node:test'
import ts from 'typescript'

const components = ['ModernEngineWorkspace', 'ModernKohyaWorkspace', 'ModernQwenWorkspace']
function formatter(component, values, style) {
  const source = readFileSync(new URL(`../src/components/${component}.vue`, import.meta.url), 'utf8')
  const script = source.match(/<script setup[^>]*>([\s\S]*?)<\/script>/)[1]
  const ast = ts.createSourceFile(`${component}.ts`, script, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
  const fn = ast.statements.find(node => ts.isFunctionDeclaration(node) && node.name?.text === 'styledPresetValue')
  assert.ok(fn, `Missing actual styledPresetValue in ${component}`)
  const code = ts.transpileModule(fn.getText(ast), { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
  const context = { presetFor: () => values, draft: { style_preset: style, base_type: 'anima' },
    trainingDraft: { style_preset: style }, currentModeKey: { value: 'character' } }
  vm.runInNewContext(code + '\nglobalThis.format = styledPresetValue', context)
  return context.format
}

for (const component of components) {
  test(`${component}: zero learning rate stays a valid number in both style presets`, () => {
    for (const style of ['动漫', '写实']) {
      for (const zero of [0, '0', '0.00', '0e-4']) {
        const format = formatter(component, { te_lr: zero, unet_lr: zero }, style)
        for (const key of ['te_lr', 'unet_lr']) {
          const value = format(key)
          assert.equal(Number(value), 0, `${component} ${style}: ${value}`)
          assert.match(value, /^[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?$/i)
        }
      }
    }
  })
  test(`${component}: nonzero learning-rate scaling and compact exponent stay intact`, () => {
    for (const [style, expected] of [['动漫', 8.5e-5], ['写实', 1.15e-4], ['自定义', 1e-4]]) {
      const format = formatter(component, { unet_lr: '1e-4', rank: 32 }, style)
      assert.equal(Number(format('unet_lr')), expected)
      assert.equal(format('rank'), '32')
    }
  })
  test(`${component}: empty and nonnumeric preset values are not changed`, () => {
    const format = formatter(component, { unet_lr: 'auto', te_lr: '', rank: undefined }, '动漫')
    assert.equal(format('unet_lr'), 'auto')
    assert.equal(format('te_lr'), '')
    assert.equal(format('rank'), '')
  })
}