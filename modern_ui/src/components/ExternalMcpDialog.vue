<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import type { McpStatus } from '../bridge'

const props = defineProps<{ open: boolean; desktop: boolean }>()
const emit = defineEmits<{ close: []; changed: []; notify: [message: string] }>()
const status = ref<McpStatus | null>(null)
const enabled = ref(false)
const allowActions = ref(false)
const port = ref(19366)
const busy = ref(false)
const error = ref('')
const fallback = ref('')
const configInput = ref<HTMLTextAreaElement | null>(null)

async function refresh() {
  const api = window.pywebview?.api
  if (!props.desktop || !api) { error.value = '请在桌面版设置外部 MCP。'; return }
  busy.value = true
  try {
    const result = await api.get_mcp_status()
    status.value = result
    enabled.value = result.enabled
    allowActions.value = result.allow_actions
    port.value = result.port
    error.value = result.error || ''
  } catch { error.value = '无法读取 MCP 设置。' }
  finally { busy.value = false }
}

async function save() {
  const api = window.pywebview?.api
  if (!api || busy.value) return
  busy.value = true
  error.value = ''
  fallback.value = ''
  try {
    const result = await api.configure_mcp(enabled.value, allowActions.value, port.value)
    status.value = result
    if (!result.ok) error.value = result.error || '保存 MCP 设置失败。'
    else { emit('changed'); emit('notify', result.running ? '外部 MCP 已开启' : '外部 MCP 已关闭') }
  } catch { error.value = '保存 MCP 设置失败。' }
  finally { busy.value = false }
}

async function copy(format: 'codex' | 'json') {
  const api = window.pywebview?.api
  if (!api || busy.value) return
  busy.value = true
  error.value = ''
  try {
    const result = await api.get_mcp_client_config(format)
    if (!result.ok || !result.text) { error.value = result.error || '无法生成连接配置。'; return }
    try {
      await navigator.clipboard.writeText(result.text)
      fallback.value = ''
      emit('notify', format === 'codex' ? '已复制 Codex 配置，粘贴到自己的 config.toml 后重新连接 MCP' : '已复制 MCP 连接配置')
    } catch {
      fallback.value = result.text
      await nextTick()
      configInput.value?.focus()
      configInput.value?.select()
    }
  } catch { error.value = '无法生成连接配置。' }
  finally { busy.value = false }
}

async function rotate() {
  const api = window.pywebview?.api
  if (!api || busy.value) return
  busy.value = true
  error.value = ''
  fallback.value = ''
  try {
    const result = await api.rotate_mcp_token()
    if (!result.ok) error.value = result.error || '重置连接密钥失败。'
    else { status.value = result; emit('notify', '连接密钥已重置，请重新复制客户端配置') }
  } catch { error.value = '重置连接密钥失败。' }
  finally { busy.value = false }
}

watch(() => props.open, open => {
  fallback.value = ''
  if (open) void refresh()
})
</script>

<template>
  <div v-if="open" class="mcp-backdrop" @click.self="emit('close')">
    <section class="mcp-dialog" role="dialog" aria-modal="true" aria-labelledby="mcp-title">
      <header><div><h2 id="mcp-title">外部 Agent · MCP</h2><p>连接 Codex 等 Agent，管理项目、分析报错和调用训练任务。</p></div><button class="close" type="button" aria-label="关闭" @click="emit('close')">×</button></header>
      <label class="option"><input v-model="enabled" type="checkbox" :disabled="busy || !desktop"><span><strong>开启本机 MCP 接口</strong><small>只接受本机连接；软件关闭后接口停止。</small></span></label>
      <label class="option"><input v-model="allowActions" type="checkbox" :disabled="busy || !enabled || !desktop"><span><strong>允许修改配置、下载和控制训练</strong><small>关闭时只允许查询。任务与软件界面共用执行流程。</small></span></label>
      <label class="port">连接端口<input v-model.number="port" type="number" min="1024" max="65535" step="1" :disabled="busy || !desktop"></label>
      <div class="status"><span :class="{ running: status?.running }">{{ status?.running ? '运行中' : '未开启' }}</span><code>{{ status?.url || 'http://127.0.0.1:19366/mcp' }}</code></div>
      <p class="hint">连接后，外部 Agent 可读取本机路径、项目参数与脱敏日志。软件的 API Key 和文字服务配置不通过 MCP 提供。</p>
      <div class="actions"><button type="button" :disabled="busy || !desktop" @click="save">保存设置</button><button type="button" :disabled="busy || !status?.running" @click="copy('codex')">复制 Codex 配置</button><button type="button" :disabled="busy || !status?.running" @click="copy('json')">复制通用配置</button></div>
      <p class="hint">将 Codex 配置粘贴到自己的 config.toml 后重新连接 MCP。连接配置含本机访问密钥，请勿公开分享。</p>
      <textarea v-if="fallback" ref="configInput" :value="fallback" readonly aria-label="MCP 连接配置，请按 Ctrl+C 复制"></textarea>
      <p v-if="fallback" class="hint">已选中配置，请按 Ctrl+C 复制。</p>
      <p v-if="error" class="error" role="alert">{{ error }}</p>
      <footer><button type="button" :disabled="busy || !desktop" @click="rotate">重置连接密钥</button><button type="button" @click="emit('close')">完成</button></footer>
    </section>
  </div>
</template>

<style scoped>
.mcp-backdrop { position:fixed; inset:0; z-index:34; display:grid; place-items:center; padding:20px; background:rgb(10 12 16 / 55%); }
.mcp-dialog { width:min(100%,620px); max-height:calc(100dvh - 40px); overflow:auto; padding:24px; border:1px solid var(--border); border-radius:12px; color:var(--text); background:var(--card); box-shadow:0 18px 60px rgb(0 0 0 / 30%); }
header { display:flex; justify-content:space-between; gap:16px; margin-bottom:18px; } h2 { margin:0 0 8px; font-size:20px; font-weight:550; } header p,.hint { margin:8px 0; color:var(--hint); font-size:12px; line-height:1.7; }
button { min-height:34px; padding:6px 12px; border:1px solid var(--border); border-radius:6px; background:transparent; color:var(--text); font:inherit; font-size:12px; cursor:pointer; } button:hover:not(:disabled) { border-color:var(--accent); } button:disabled { opacity:.45; cursor:default; } .close { border:0; font-size:23px; padding:0 5px; align-self:flex-start; }
.option { display:flex; gap:12px; align-items:flex-start; padding:14px 0; border-top:1px solid var(--border); } .option input { margin-top:4px; accent-color:var(--accent); } .option span { display:grid; gap:6px; } strong { font-size:13px; font-weight:500; } small { color:var(--hint); font-size:12px; line-height:1.5; }
.port { display:flex; align-items:center; gap:18px; margin:14px 0; font-size:12px; } .port input { width:110px; padding:8px; border:1px solid var(--border); border-radius:5px; color:var(--text); background:var(--bg); }
.status { display:flex; gap:12px; flex-wrap:wrap; padding:12px; border:1px solid var(--border); border-radius:6px; font-size:12px; } .status span { color:var(--hint); } .status .running { color:#79b995; } code { overflow-wrap:anywhere; }
.actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:18px; } .actions button:first-child { background:var(--accent); color:var(--bg); } textarea { box-sizing:border-box; width:100%; min-height:100px; margin-top:10px; padding:10px; color:var(--text); background:var(--bg); border:1px solid var(--border); border-radius:6px; font-family:monospace; font-size:11px; resize:vertical; } .error { color:#d59a9a; font-size:12px; white-space:pre-wrap; }
footer { display:flex; justify-content:space-between; gap:8px; margin-top:20px; padding-top:16px; border-top:1px solid var(--border); }
</style>
