<script setup lang="ts">
import { onUnmounted, ref, watch } from 'vue'
import type { H3DatasetSummary } from '../bridge'
import DatasetImageBrowser from './DatasetImageBrowser.vue'
const props = defineProps<{ directory: string; trigger: string; desktop: boolean; method: string; syncTrigger: boolean; replace: boolean }>()
const emit = defineEmits<{ action: [value: string]; 'update:syncTrigger': [value: boolean]; 'update:replace': [value: boolean] }>()
const summary = ref<H3DatasetSummary | null>(null)
const busy = ref(false)
let request = 0
let timer: ReturnType<typeof setTimeout> | undefined
async function inspect() {
  const token = ++request
  if (!props.directory || !props.desktop || !window.pywebview?.api) { summary.value = null; busy.value = false; return }
  busy.value = true
  try {
    const result = await window.pywebview.api.inspect_h3_dataset(props.directory, props.trigger)
    if (token === request) summary.value = result
  } catch { if (token === request) summary.value = { ok: false, error: '无法检查 H3 数据，请重试。' } }
  finally { if (token === request) busy.value = false }
}
watch(() => [props.directory, props.trigger], () => {
  request++; summary.value = null
  if (timer) clearTimeout(timer)
  timer = setTimeout(() => { void inspect() }, 250)
}, { immediate: true })
onUnmounted(() => { request++; if (timer) clearTimeout(timer) })
</script>

<template>
  <section class="h3-data" aria-label="H3 数据准备">
    <label v-if="method === 'wd14'" class="h3-check"><input type="checkbox" :checked="replace" @change="emit('update:replace', ($event.target as HTMLInputElement).checked)" />重新生成已有图片标签（原文本备份）</label>
    <label class="h3-check"><input type="checkbox" :checked="syncTrigger" @change="emit('update:syncTrigger', ($event.target as HTMLInputElement).checked)" />同步触发词（保留描述正文，修改前备份）</label>
    <div class="h3-actions"><button type="button" @click="emit('action', 'preprocess')">准备标签与检查媒体</button><button v-if="!summary?.ok || summary.videos" type="button" @click="emit('action', 'video_caption')">视频 AI 描述</button><button type="button" :disabled="busy" @click="inspect()">{{ busy ? '检查中…' : '刷新检查' }}</button></div>
    <template v-if="summary?.ok">
      <p>图片 {{ summary.images }} · 视频 {{ summary.videos }} · 音频 {{ summary.audio }} · 已有描述 {{ summary.captioned }}</p>
      <p v-if="summary.missing_captions?.length" class="warning">缺少描述 {{ summary.missing_captions.length }} 个：{{ summary.missing_captions.slice(0, 5).join('、') }}{{ summary.missing_captions.length > 5 ? '…' : '' }}</p>
      <p v-if="summary.missing_trigger?.length" class="warning">{{ summary.missing_trigger.length }} 个文本未包含项目触发词。{{ syncTrigger ? '准备数据时补入。' : '当前未开启同步。' }}</p>
      <p v-if="!summary.total">目录中没有受支持的训练媒体。</p>
      <p v-if="summary.audio">音频使用同名 .txt 描述，需手动准备。</p>
    </template>
    <p v-else-if="summary?.error" class="warning">{{ summary.error }}</p>
    <p>图片标签默认只补缺失；视频描述跳过已有文本。递归读取子目录，不裁切、移动或转码训练媒体。</p>
    <DatasetImageBrowser :directory="directory" :desktop="desktop" recursive raw-media />
  </section>
</template>

<style scoped>
.h3-data{display:grid;gap:9px;padding:12px;border:1px solid var(--border);border-radius:6px;font-size:12px;background:var(--bg)}.h3-check{display:flex;align-items:flex-start;gap:7px;line-height:1.6}.h3-actions{display:flex;flex-wrap:wrap;gap:8px}.h3-data p{margin:0;color:var(--hint);line-height:1.6;overflow-wrap:anywhere}.h3-data .warning{color:var(--tone-d4b06a)}.h3-data button{font:inherit;color:var(--text);background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:7px 10px;cursor:pointer}.h3-data button:disabled{opacity:.5;cursor:default}.h3-data button:focus-visible,.h3-data input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
</style>
