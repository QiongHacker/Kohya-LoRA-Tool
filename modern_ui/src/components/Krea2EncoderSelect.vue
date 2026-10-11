<script setup lang="ts">
import type { Krea2EncoderChoice, Krea2EncoderSelection } from '../bridge'
const props = defineProps<{ encoder: Krea2EncoderSelection; disabled?: boolean }>()
const emit = defineEmits<{ select: [choice: Krea2EncoderChoice] }>()
function select(event: Event) {
  const element = event.target as HTMLSelectElement
  const choice = element.value as Krea2EncoderChoice
  // The visible value changes only after persistence succeeds, including errors.
  element.value = props.encoder.choice
  if (choice !== props.encoder.choice) emit('select', choice)
}
</script>

<template>
  <div class="encoder-setting">
    <div class="encoder-control">
      <label><span>文本编码器类型</span>
        <select v-if="encoder.supports_fp8" :value="encoder.choice" :disabled="disabled" aria-label="文本编码器类型" @change="select">
          <option value="auto">自动推荐（优先已就绪的 FP8 scaled）</option>
          <option value="te_fp8">FP8 scaled（较省内存，8GB 显卡推荐）</option>
          <option value="te">BF16（完整精度，内存需求较高）</option>
        </select>
        <strong v-else>BF16 · 此引擎版本不支持 FP8 scaled</strong>
      </label>
      <div class="encoder-result" :class="{ missing: !encoder.ready }">
        <span>{{ encoder.ready ? '当前使用' : '待准备' }}</span><strong>{{ encoder.label }}</strong>
      </div>
    </div>
    <p>{{ encoder.reason }}{{ encoder.choice === 'auto' && encoder.supports_fp8 ? '要固定使用一种，请明确选择 FP8 scaled 或 BF16。' : '下载或指定备用文件不会切换类型。' }}</p>
  </div>
</template>

<style scoped>
.encoder-setting { padding: 10px 12px; border: 1px solid var(--tone-3d414a); border-radius: 6px; background: var(--tone-22252c); }
.encoder-control { display:flex; align-items:flex-end; gap:16px; flex-wrap:wrap; }
.encoder-control label { display:grid; gap:6px; flex:1; min-width:220px; color:var(--hint); font-size:11px; }
.encoder-control select { width:100%; min-height:32px; padding:0 9px; border:1px solid var(--tone-3d414a); border-radius:5px; color:var(--tone-c6cbd3); background:var(--tone-272a32); font-size:11px; }
.encoder-control select:disabled { opacity:.55; cursor:wait; }
.encoder-result { display:flex; gap:8px; align-items:center; padding-bottom:8px; font-size:11px; color:var(--tone-aab8ad); }
.encoder-result.missing { color:var(--tone-c69da1); }
.encoder-control strong { font-size:11px; font-weight:400; }
p { margin:8px 0 0; font-size:10px; color:var(--hint); line-height:1.5; }
</style>
