<script setup lang="ts">
import { computed } from 'vue'
import type { Krea2EncoderChoice, ModelDownloadItem, ModelDownloadList } from '../bridge'
import Krea2EncoderSelect from './Krea2EncoderSelect.vue'
const props = defineProps<{ setup?: ModelDownloadList | null; disabled?: boolean; versionPending?: boolean }>()
const emit = defineEmits<{
  configure: []
  selectEncoder: [choice: Krea2EncoderChoice]
  component: [key: string, action: 'select' | 'default']
}>()
const required = computed(() => props.setup?.items?.filter(item => item.required) || [])
const extras = computed(() => props.setup?.items?.filter(item => !item.required) || [])
const missing = computed(() => required.value.filter(item => !item.present).length)
function filename(item: ModelDownloadItem) { return item.path?.split(/[\\/]/).pop() || item.filename }
function status(item: ModelDownloadItem) {
  return item.validation_error ? '文件无效' : item.disabled ? '已停用' : item.present ? (item.required ? '当前使用' : '已准备 · 备用') : (item.required ? '缺少文件' : '未准备 · 可选')
}
</script>

<template>
  <section class="components-panel" aria-label="模型与配套组件">
    <header class="components-heading">
      <div><span>01</span><h2>模型与配套组件</h2><small v-if="setup?.ok" :class="{ warning: missing }">{{ missing ? `还需准备 ${missing} 项` : '训练组件就绪' }}</small></div>
      <button type="button" :disabled="disabled" @click="emit('configure')">配置与下载</button>
    </header>
    <template v-if="setup?.ok && setup.encoder">
      <Krea2EncoderSelect :encoder="setup.encoder" :disabled="disabled || versionPending" @select="emit('selectEncoder', $event)" />
      <p v-if="versionPending" class="component-hint">引擎版本尚未保存；请保存修改，或点击“配置与下载”先保存再配置。下方显示已保存版本的组件。</p>
      <div class="component-list">
        <article v-for="item in required" :key="item.key" class="component-row">
          <div class="component-copy"><strong>{{ item.key === 'raw' ? 'RAW 训练底模' : item.key === 'vae' ? 'VAE' : '文本编码器 · ' + setup.encoder.label }}</strong><span :title="item.path">{{ filename(item) }}</span><small :title="item.path">{{ item.manual ? '指定本地文件' : '默认文件' }} · {{ item.path }}</small><small v-if="item.validation_error" class="warning">{{ item.validation_error }}</small></div>
          <span class="component-status" :class="{ warning: !item.present }">{{ status(item) }}</span>
          <div class="component-actions"><button type="button" :disabled="disabled || versionPending" @click="emit('component', item.key, 'select')">指定本地文件</button><button v-if="item.manual" class="text-button" type="button" :disabled="disabled || versionPending" @click="emit('component', item.key, 'default')">恢复默认文件</button></div>
        </article>
      </div>
      <details class="component-extras">
        <summary>备用编码器与 Turbo（可选）</summary>
        <p class="component-hint">备用编码器不会参与本次训练，无需两个都下载。Turbo 仅供启用采样时使用，详细设置在“配置与下载”。</p>
        <article v-for="item in extras" :key="item.key" class="component-row">
          <div class="component-copy"><strong>{{ item.label }}</strong><span :title="item.path">{{ filename(item) }}</span></div><span class="component-status">{{ item.key === 'turbo' && item.present ? '已准备 · 预览可选' : status(item) }}</span>
          <button type="button" :disabled="disabled || versionPending" @click="emit('component', item.key, 'select')">指定本地文件</button>
        </article>
      </details>
    </template>
    <p v-else class="component-hint">{{ setup?.error || '模型管理在桌面版读取本机组件；可通过“配置与下载”选择已有文件。' }}</p>
  </section>
</template>

<style scoped>
.components-panel { padding:13px; border:1px solid var(--outline-subtle); border-radius:7px; background:var(--card); min-width:0; }
.components-heading { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px; }
.components-heading>div { display:flex; align-items:center; gap:9px; flex-wrap:wrap; }
.components-heading h2 { margin:0; color:var(--tone-cbd0d7); font-weight:400; font-size:13px; }
.components-heading span { color:var(--tone-8993a0); font-size:10px; }
.components-heading small,.component-status { color:var(--tone-aab8ad); font-size:10px; }
button { min-height:29px; padding:0 9px; border:1px solid var(--border); border-radius:5px; background:transparent; color:var(--tone-b3bac4); font-size:10px; white-space:nowrap; cursor:pointer; }
button:hover:not(:disabled) { border-color:var(--tone-505660); background:var(--tone-292c34); }
button:disabled { opacity:.5; cursor:wait; }
.component-list { margin-top:8px; }
.component-row { display:flex; align-items:center; gap:14px; padding:10px 0; border-bottom:1px solid var(--outline-subtle); min-width:0; }
.component-row:last-child { border-bottom:0; padding-bottom:3px; }
.component-copy { display:grid; flex:1; min-width:0; gap:4px; }
.component-copy strong { color:var(--tone-c5cad1); font-size:11px; font-weight:400; }
.component-copy span { color:var(--tone-a7adb7); font-size:11px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.component-copy small { color:var(--hint); font-size:9px; overflow-wrap:anywhere; }
.component-copy .warning,.warning { color:var(--tone-c69da1); }
.component-actions { display:flex; flex-direction:column; align-items:flex-end; gap:6px; }
.text-button { min-height:20px; padding:0; border:0; font-size:9px; text-decoration:underline; text-underline-offset:3px; }
.component-extras { margin-top:11px; color:var(--hint); font-size:10px; }
.component-extras summary { cursor:pointer; }
.component-hint { color:var(--hint); font-size:10px; line-height:1.5; margin:8px 0; }
@media(max-width:700px) { .component-row { flex-wrap:wrap; gap:8px; }.component-copy { flex-basis:70%; }.component-actions { flex-direction:row; align-items:center; } }
</style>
