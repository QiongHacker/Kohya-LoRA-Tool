<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'

type Key = 'pixai_character' | 'pixai_copyright' | 'pixai_style' | 'pixai_threshold'
const props = withDefaults(defineProps<{
  options: { pixai_character: boolean; pixai_copyright: boolean; pixai_style: boolean; pixai_threshold: string | number }
  desktop?: boolean; disabled?: boolean; download?: boolean
}>(), { download: true })
const emit = defineEmits<{ change: [key: Key, value: boolean | string] }>()
const ready = ref(false)
const starting = ref(false)
const busy = ref(false)
const taskId = ref('')
const message = ref('')
const progress = ref<number | null>(null)
let disposed = false
let timer: ReturnType<typeof setTimeout> | undefined
async function load() {
  if (!props.download || !props.desktop || !window.pywebview?.api) return
  try {
    const result = await window.pywebview.api.get_prompt_reverse_model('pixai')
    if (disposed) return
    ready.value = Boolean(result.ready)
    if (!result.ok) message.value = result.error || '模型状态读取失败。'
    const current = await window.pywebview.api.get_prompt_reverse()
    if (disposed) return
    if (current.task?.kind === 'prompt_reverse_model' && current.task.model_key === 'pixai') {
      taskId.value = current.task.id
      void poll()
    }
  } catch { if (!disposed) message.value = 'PixAI 接口尚未就绪，请重启更新后的软件。' }
}
async function poll() {
  if (disposed || !taskId.value || !window.pywebview?.api) return
  try {
    const result = await window.pywebview.api.get_prompt_reverse(taskId.value)
    if (disposed) return
    busy.value = result.task?.status === 'running'
    progress.value = result.task?.progress ?? null
    message.value = busy.value ? result.task?.detail || '正在下载并校验…' : result.task?.message || '下载任务已结束。'
    if (!busy.value) {
      const state = await window.pywebview.api.get_prompt_reverse_model('pixai')
      if (!disposed) ready.value = Boolean(state.ready)
      return
    }
  } catch { if (!disposed) message.value = '进度读取失败，重新打开页面可继续查看。' }
  if (!disposed) timer = setTimeout(() => { void poll() }, 1000)
}
async function downloadModel() {
  if (!props.desktop || !window.pywebview?.api || busy.value || starting.value) return
  starting.value = true
  try {
    const result = await window.pywebview.api.download_prompt_reverse_model('pixai')
    if (disposed) return
    if (!result.ok || !result.task_id) { message.value = result.error || '模型下载未启动。'; return }
    taskId.value = result.task_id
    busy.value = true; message.value = '正在下载 PixAI…'
    void poll()
  } catch { if (!disposed) message.value = '模型下载未启动，请重试。' }
  finally { starting.value = false }
}
async function stop() {
  if (!taskId.value || !window.pywebview?.api) return
  try {
    const result = await window.pywebview.api.cancel_task(taskId.value)
    if (!disposed) message.value = result.ok ? '正在停止，已下载文件保留。' : result.error || '停止失败。'
  } catch { if (!disposed) message.value = '停止请求未完成，请在运行任务中重试。' }
}
onMounted(() => { void load() })
onUnmounted(() => { disposed = true; if (timer) clearTimeout(timer) })
</script>

<template>
  <section class="pixai-options" aria-label="PixAI 标签设置">
    <label class="threshold">内容标签阈值<input :value="options.pixai_threshold" type="number" min="0.01" max="0.99" step="0.01" :disabled="disabled || busy" @input="emit('change', 'pixai_threshold', ($event.target as HTMLInputElement).value)" /></label>
    <div class="categories">
      <label><input type="checkbox" :checked="options.pixai_character" :disabled="disabled || busy" @change="emit('change', 'pixai_character', ($event.target as HTMLInputElement).checked)" />角色名称</label>
      <label><input type="checkbox" :checked="options.pixai_copyright" :disabled="disabled || busy" @change="emit('change', 'pixai_copyright', ($event.target as HTMLInputElement).checked)" />作品名称</label>
      <label><input type="checkbox" :checked="options.pixai_style" :disabled="disabled || busy" @change="emit('change', 'pixai_style', ($event.target as HTMLInputElement).checked)" />画风标签</label>
    </div>
    <p>默认仅输出画面内容。内容阈值 0.17；可选类别使用官方阈值。</p>
    <template v-if="download">
      <div class="model"><span>PixAI v1.0 · {{ ready ? '已下载' : '模型约 1.95 GB' }}</span><button type="button" :disabled="!desktop || starting || busy || disabled" @click="downloadModel">{{ busy ? '下载中…' : starting ? '正在启动…' : ready ? '检查 / 修复' : '下载模型' }}</button><button v-if="busy" type="button" @click="stop">停止下载</button></div>
      <progress v-if="busy && progress !== null" :value="progress" max="1"></progress>
      <p>首次使用准备独立推理环境。更换打标模型或类别后，请勾选“重新处理已存在的图片”。</p>
      <p v-if="message" role="status">{{ message }}</p>
    </template>
  </section>
</template>

<style scoped>
.pixai-options{display:grid;gap:9px;padding:12px;margin:8px 0;border:1px solid var(--border);border-radius:6px;background:var(--bg);font-size:12px;color:var(--text)}.threshold{display:grid;gap:6px}.threshold input{box-sizing:border-box;width:100%;padding:8px;border:1px solid var(--border);border-radius:4px;background:var(--card);color:var(--text);font:inherit}.categories,.model{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.categories label{display:flex;align-items:center;gap:5px}.categories input{accent-color:var(--accent)}p{margin:0;font-size:12px;line-height:1.65;color:var(--hint)}button{padding:7px 10px;border:1px solid var(--border);border-radius:5px;background:var(--card);color:var(--text);font:inherit;cursor:pointer}button:disabled{opacity:.5;cursor:default}progress{width:100%;accent-color:var(--accent)}
</style>
