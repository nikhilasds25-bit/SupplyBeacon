const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const vm = require('node:vm')
const ts = require('typescript')

function mount(fetchSnapshot) {
  let value = null, poll, cleanup, cleared = false
  const exports = {}
  const source = fs.readFileSync(require.resolve('../src/hooks/useCurrentSnapshot.ts'), 'utf8')
  const compiled = ts.transpileModule(source, {compilerOptions: {module: ts.ModuleKind.CommonJS}}).outputText
  vm.runInNewContext(compiled, {
    exports,
    require(name) {
      if (name === 'react') return {
        useState: () => [value, next => {value = next}],
        useEffect: effect => {cleanup = effect()},
      }
      return {getCurrentSnapshot: fetchSnapshot}
    },
    window: {
      setInterval: callback => {poll = callback; return 1},
      clearInterval: () => {cleared = true},
      addEventListener: () => {},
      removeEventListener: () => {},
    },
  })
  exports.useCurrentSnapshot()
  return {value: () => value, poll: () => poll(), unmount: () => cleanup(), cleared: () => cleared}
}
const settle = () => new Promise(resolve => setImmediate(resolve))

test('polling replaces entity risks together with the event summary', async () => {
  const old = {summary: {operational_events: 1}, events: [{id: 1}], entities: [{risk_level: 'Medium'}]}
  const fresh = {summary: {operational_events: 0}, events: [], entities: [{risk_level: 'Low'}]}
  let snapshot = old
  const hook = mount(async () => snapshot)
  await settle()
  assert.equal(hook.value(), old)
  snapshot = fresh
  hook.poll()
  await settle()
  assert.equal(hook.value(), fresh)
  assert.equal(hook.value().entities[0].risk_level, 'Low')
  assert.equal(hook.value().summary.operational_events, 0)
  hook.unmount()
  assert.equal(hook.cleared(), true)
})

test('slow requests do not overlap or update an unmounted page', async () => {
  let resolve, calls = 0
  const hook = mount(() => {calls++; return new Promise(done => {resolve = done})})
  hook.poll()
  assert.equal(calls, 1)
  hook.unmount()
  resolve({summary: {operational_events: 1}})
  await settle()
  assert.equal(hook.value(), null)
})

test('poll failure reports unavailable data and retry loads a whole snapshot', async () => {
  let failed = false
  const fresh = {summary: {operational_events: 0}, entities: []}
  const hook = mount(async () => {if (failed) throw new Error('offline'); return fresh})
  await settle()
  failed = true
  hook.poll()
  await settle()
  assert.equal(hook.value().error, 'offline')
  failed = false
  hook.value().retry()
  await settle()
  assert.equal(hook.value(), fresh)
  hook.unmount()
})
