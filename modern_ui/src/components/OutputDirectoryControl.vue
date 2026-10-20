<script setup lang="ts">
import { ref, watch } from 'vue'
import type { ProjectConfig } from '../bridge'
const props = defineProps<{ projectName: string; config?: ProjectConfig | null; desktop: boolean }>()
const emit = defineEmits<{ action: [action: string] }>()
const directory = ref('')
let request = 0
watch(() => [props.projectName, props.config] as const, async () => {
  const id = ++request
  directory.value = String(props.config?.output_dir || '')
  const api = window.pywebview?.api
  if (!props.desktop || !api) return
  try {
    const result = await api.run_action('output_dir_info', props.projectName)
    if (id === request) directory.value = result.ok ? result.path || '' : result.error || '目录读取失败'
  } catch { if (id === request) directory.value ||= '目录读取失败' }
}, { immediate: true })
</script>

<template>
  <div class="output-directory">
    <div class="output-directory-copy"><strong>输出文件目录</strong><span :title="directory">{{ directory || '默认项目输出目录' }}</span></div>
    <div class="output-directory-actions">
      <button type="button" :disabled="!desktop" title="选择本项目的输出文件夹，已有文件保留在原处。" @click="emit('action', 'choose_output_dir')">手动指定</button>
      <button v-if="config?.output_dir" type="button" :disabled="!desktop" @click="emit('action', 'reset_output_dir')">恢复默认</button>
    </div>
    <small>后续训练保存到此处；已有文件保留在原目录。</small>
  </div>
</template>

<style scoped>
.output-directory{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;padding:12px 14px;border:1px solid var(--border);border-radius:7px;background:var(--tone-252830);margin:10px 0}
.output-directory-copy{flex:1;min-width:180px;display:flex;flex-direction:column;gap:5px}
.output-directory-copy strong{font-size:12px;font-weight:500;color:var(--text)}
.output-directory-copy span{font-size:11px;color:var(--muted);overflow-wrap:anywhere}
.output-directory-actions{display:flex;flex-wrap:wrap;gap:6px}
.output-directory button{border:1px solid var(--border);border-radius:5px;min-height:32px;padding:5px 10px;background:var(--panel);color:var(--text);font:inherit;font-size:12px;cursor:pointer}
.output-directory button:hover{border-color:var(--accent)}
.output-directory button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.output-directory button:disabled{opacity:.5;cursor:default}
.output-directory small{flex-basis:100%;color:var(--muted);font-size:10px}
</style>
