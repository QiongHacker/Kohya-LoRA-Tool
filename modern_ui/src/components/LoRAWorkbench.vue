<script setup lang="ts">
import { computed, nextTick, onUnmounted, reactive, ref, watch } from 'vue'
import type { LoRAInfo, LoRAPreview, LoRAWorkbenchReport, LoRAWeightAnalysis, LoRAProfileConfig, LoRAProfilePicture, LoRAProfileWindow, LoRAPreset, LoRAExploreConfig, LoRAExploreVariant, LoRAProfileBaseline } from '../bridge'
import UiIcon from './UiIcon.vue'
import { ANIMA_GUIDE_SOURCE, ANIMA_MOD_GUIDANCE_SOURCE, ANIMA_GUIDE_GROUPS, ANIMA_GROUPING, animaLayerHint, supportsAnimaGuide } from '../animaLayerGuide'

const props = defineProps<{ open: boolean; desktop: boolean; projectName: string }>()
const emit = defineEmits<{ close: []; notify: [message: string]; environment: [] }>()
const dialog = ref<HTMLElement | null>(null)
let previousFocus: HTMLElement | null = null
const source = ref('')
const family = ref('auto')
const families = ref<Record<string, string>>({ krea2: 'Krea 2', anima: 'Anima' })
const ready = ref(false)
const reason = ref('')
const recent = ref<Array<{ name: string; path: string }>>([])
const info = ref<LoRAInfo | null>(null)
const blocks = reactive<Record<string, number>>({})
const tab = ref<'adjust' | 'compress'>('adjust')
const showLayers = ref(false)
const workspaceMode = ref<'repair' | 'explore'>('repair')
const inspectorOpen = ref(true)
const focusedPicture = ref<number | null>(null)
const exploration = ref<LoRAWorkbenchReport | null>(null)
const exploreConfig = reactive<LoRAExploreConfig>({ intensity: 0.2, structure: 0, mutations: 3, mutation_seed: 42, state: {}, frozen: [], last_pick: [] })
const explorePictures = ref<Array<LoRAExploreVariant & { url: string }>>([])
const exploreBaseline = ref<{ url: string; label: string } | null>(null)
const pickedVariant = ref('')
const profileBaseline = ref<LoRAProfileBaseline | null>(null)
const undoPreview = ref<LoRAPreview | null>(null)
const undoExplore = ref<{ last_pick: string[]; picked: string; state: LoRAExploreConfig['state'] } | null>(null)
function sameBlocks(values?: Record<string, number>) { return !!values && Object.keys(blocks).every(id => blocks[id] === values[id]) }
const explorationOutdated = computed(() => !!exploration.value && (exploration.value.info.fingerprint !== info.value?.fingerprint ||
  Object.keys(preview).some(k => String(preview[k as keyof LoRAPreview]) !== String(exploration.value?.recipe?.[k as keyof LoRAPreview])) ||
  (!sameBlocks(exploration.value.blocks) && !explorePictures.value.some(v => sameBlocks(v.blocks)))))
const profileBaselineOutdated = computed(() => !!profileBaseline.value && (!sameBlocks(profileBaseline.value.blocks) || Object.keys(preview).some(k => String(preview[k as keyof LoRAPreview]) !== String(profileBaseline.value?.recipe[k as keyof LoRAPreview]))))
const activeExploreBlocks = computed(() => blockIds.value.filter(id => !exploreConfig.frozen.includes(id)))
const visiblePictures = computed(() => selectedPictures.value.filter((_, index) => focusedPicture.value == null || focusedPicture.value === index))
const currentExploreImage = computed(() => {
  const chosen = explorePictures.value.find(v => sameBlocks(v.blocks))
  return chosen ? { url: chosen.url, label: `当前基准 · ${chosen.label}` } : exploreBaseline.value
})
function toggleFrozen(ids: string[], frozen: boolean) {
  if (locked.value) return
  const values = new Set(exploreConfig.frozen)
  ids.forEach(id => frozen ? values.add(id) : values.delete(id))
  exploreConfig.frozen = [...values]
}
function canPick(variant: LoRAExploreVariant) {
  return !locked.value && !explorationOutdated.value && exploreConfig.frozen.every(id => variant.blocks[id] === blocks[id])
}
function pickVariant(variant: LoRAExploreVariant) {
  if (!canPick(variant)) return
  remember(); Object.assign(blocks, variant.blocks)
  exploreConfig.last_pick = [...variant.changed]; exploreConfig.state = { ...variant.state }; pickedVariant.value = variant.name
  message.value = `${variant.label} 已选为基准，可继续探索或进入分层调整。`
}
async function rollExploration() {
  if (!previewConfigured.value || !activeExploreBlocks.value.length || preview.strength === 0) return
  exploreConfig.mutation_seed = Math.floor(Math.random() * 2 ** 32)
  await start('explore')
}
async function readExploration(report: LoRAWorkbenchReport, id: string, restore = false) {
  if (!report.exploration || !report.images) return
  const loaded = new Map<string, string>()
  for (const image of report.images) {
    const result = await window.pywebview?.api.get_lora_workbench_image(id, image.name)
    if (!result?.ok || !result.data_url) { fail(result?.error || '变体图片读取失败。'); return }
    loaded.set(image.name, result.data_url)
  }
  if (!loaded.get('explore_0000.png') || report.exploration.variants.length !== 4 || report.exploration.variants.some(v => !loaded.has(v.name))) { fail('变体结果不完整，请重新生成。'); return }
  exploration.value = report; profileBaseline.value = null
  exploreBaseline.value = { url: loaded.get('explore_0000.png')!, label: '本轮基准' }
  explorePictures.value = report.exploration.variants.map(v => ({ ...v, url: loaded.get(v.name)! }))
  pickedVariant.value = ''
  if (restore) {
    Object.assign(exploreConfig, report.exploration.config)
    if (!sameBlocks(report.blocks)) {
      const chosen = explorePictures.value.find(v => sameBlocks(v.blocks))
      if (chosen) { pickedVariant.value = chosen.name; exploreConfig.state = { ...chosen.state }; exploreConfig.last_pick = [...chosen.changed] }
    }
    undoBlocks.value = null; undoExplore.value = null
  }
}

const flowStep = ref(0)
const selectedWindowId = ref('')
type WorkbenchPanel = 'presets' | 'suggestions' | 'layer-guide'
const panel = ref<WorkbenchPanel | null>(null)
const panelTitle = computed(() => panel.value === 'presets' ? '调整预设' : panel.value === 'layer-guide' ? 'Anima 层级参考' : '官方分析建议')
const applyingReport = ref(false)
const mainScroll = ref<HTMLElement | null>(null)
const stepLabels = ['选择 LoRA', '配置分析', '调整效果', '对照与导出']
let panelOpener: HTMLElement | null = null
const rank = ref(8)
const busy = ref(false)
const starting = ref(false)
const taskId = ref('')
const message = ref('')
const error = ref('')
const progress = ref<number | null>(null)
const images = ref<Array<{ label: string; url: string }>>([])
const previewReport = ref<LoRAWorkbenchReport | null>(null)
const output = ref('')
const exportReport = ref<LoRAWorkbenchReport | null>(null)
const adjustedOutput = computed(() => exportReport.value?.output || '')
const exportOutdated = computed(() => !!exportReport.value && (exportReport.value.info.fingerprint !== info.value?.fingerprint || Object.keys(blocks).some(k => blocks[k] !== exportReport.value?.blocks?.[k])))
const previewFamily = ref('')
const compression = ref<LoRAWorkbenchReport['compression']>()
const profiler = ref<LoRAWorkbenchReport | null>(null)
const profileTaskId = ref('')
const infoTaskId = ref('')
const profileConfig = reactive<LoRAProfileConfig>({ class_prompt: '', references: [] })
const presets = ref<LoRAPreset[]>([])
const presetName = ref('')
const presetId = ref('')
const undoBlocks = ref<Record<string, number> | null>(null)
const fullProfileImage = ref<{ url: string; label: string } | null>(null)
const loadingImage = ref(false)
const profileLoads = new Map<string, string>()
const canvasImages = reactive<Record<string, string>>({})
function pictureSource(picture: LoRAProfilePicture) { return canvasImages[`${profileTaskId.value}/${picture.name}`] || picture.thumbnail }
async function loadCanvasImages() {
  const id = profileTaskId.value, selected = selectedWindow.value?.id
  if (!id || !props.open || flowStep.value !== 2 || workspaceMode.value !== 'repair') return
  for (const { picture } of selectedPictures.value) {
    const key = `${id}/${picture.name}`
    if (canvasImages[key]) continue
    const cached = profileLoads.get(key)
    if (cached) { canvasImages[key] = cached; continue }
    try {
      const result = await window.pywebview?.api.get_lora_workbench_image(id, picture.name)
      if (id !== profileTaskId.value || !props.open) return
      if (result?.ok && result.data_url) { canvasImages[key] = result.data_url; profileLoads.set(key, result.data_url) }
      else { fail(result?.error || '完整对照图读取失败，显示缩略图。'); return }
    } catch { fail('完整对照图读取失败，显示缩略图。'); return }
    if (selectedWindow.value?.id !== selected) return
  }
}
const analysis = ref<LoRAWeightAnalysis | null>(null)
const analysisEnergy = computed(() => Object.values(analysis.value?.blocks || {}).reduce((sum, b) => sum + b.norm ** 2, 0))
function energy(id: string) { return analysisEnergy.value ? 100 * (analysis.value?.blocks[id]?.norm || 0) ** 2 / analysisEnergy.value : 0 }
const preview = reactive<LoRAPreview>({ dit: '', te: '', vae: '', prompt: '', negative: '', seed: 42, width: 512, height: 512, steps: 25, cfg: 4.5, strength: 1, precision: 'auto', swap: 0 })
let timer: ReturnType<typeof setTimeout> | undefined
let loadRevision = 0
let appliedTask = ''
const locked = computed(() => busy.value || starting.value || applyingReport.value)
const modified = computed(() => Object.values(blocks).filter(v => v !== 1).length)
const current = computed(() => !!info.value && info.value.source === source.value && info.value.family === family.value)
const outdated = computed(() => !!previewReport.value && (previewReport.value.info.fingerprint !== info.value?.fingerprint || Object.keys(blocks).some(k => blocks[k] !== previewReport.value?.blocks?.[k]) || Object.keys(preview).some(k => String(preview[k as keyof LoRAPreview]) !== String(previewReport.value?.recipe?.[k as keyof LoRAPreview]))))
const maxRank = computed(() => Math.max(...(info.value?.ranks || [1])))
const size = (bytes: number) => `${(bytes / 1024 / 1024).toFixed(1)} MB`
const blockIds = computed(() => (info.value?.groups || []).flatMap(g => g.blocks.map(b => b.id)).sort())
const selectedPreset = computed(() => presets.value.find(p => p.id === presetId.value))
function matchesPreset(p: LoRAPreset) { return p.family === family.value && JSON.stringify([...p.block_ids].sort()) === JSON.stringify(blockIds.value) }
const profileCount = computed(() => comparisonWindows.value.length ? (2 + 2 * comparisonWindows.value.length) * (profileConfig.class_prompt.trim() ? 2 : 1) : null)
const profileOutdated = computed(() => !!profiler.value && (profileLayoutOutdated.value || profiler.value.info.fingerprint !== info.value?.fingerprint ||
  Object.keys(preview).some(k => k !== 'exact_attention' && String(preview[k as keyof LoRAPreview]) !== String(profiler.value?.recipe?.[k as keyof LoRAPreview])) ||
  profileConfig.class_prompt.trim() !== profiler.value.profile?.class_prompt || JSON.stringify(profileConfig.references) !== JSON.stringify(profiler.value.profile?.references)))
const progressPercent = computed(() => progress.value == null ? null : Math.round(Math.max(0, Math.min(1, progress.value)) * 100))
const pct = (value: number | null) => value == null ? '未测量' : `${(100 * value).toFixed(0)}%`
type EditWindow = { id: string; label: string; blocks: string[]; note: string; tag: string; result: LoRAProfileWindow | null; community?: boolean; range?: string; summary?: string }
const animaGuideSupported = computed(() => current.value && supportsAnimaGuide(info.value))
const communityWindows = computed<EditWindow[]>(() => {
  if (!animaGuideSupported.value) return []
  const available = new Set(blockIds.value)
  const windows: EditWindow[] = ANIMA_GUIDE_GROUPS.map(group => ({
    id: group.id, label: group.label, blocks: group.layers.map(layer => layer.id).filter(id => available.has(id)),
    note: '尚未生成对照', summary: group.summary, tag: '', result: null, community: true, range: group.range,
  })).filter(group => group.blocks.length > 0)
  const mapped = new Set(windows.flatMap(group => group.blocks))
  const extra = blockIds.value.filter(id => !mapped.has(id))
  if (extra.length && info.value?.profile_grouping === ANIMA_GROUPING) {
    for (const group of info.value.profile_windows || []) {
      const ids = group.blocks.filter(id => extra.includes(id))
      if (ids.length) windows.push({ ...group, blocks: ids, note: '尚未生成对照',
        summary: '不属于 L00–L27，需单独检查。', tag: '', result: null, community: true })
    }
  } else if (extra.length) windows.push({ id: 'community_other', label: '其他组件', blocks: extra,
    note: '尚未生成对照', summary: '不属于 L00–L27，需单独检查。', tag: '', result: null, community: true })
  return windows
})
const comparisonWindows = computed(() => {
  if (!current.value) return []
  if (animaGuideSupported.value) return communityWindows.value
  return info.value?.profile_windows || info.value?.groups.map((group, index) => ({ id: `group_${index}`, label: group.label, blocks: group.blocks.map(block => block.id) })) || []
})
function sameWindowBlocks(left: string[], right: string[]) {
  return left.length === right.length && new Set(left).size === left.length && new Set(right).size === right.length && left.every(id => right.includes(id))
}
const profileLayoutOutdated = computed(() => {
  const recorded = profiler.value?.profile?.windows
  return !!recorded && (recorded.length !== comparisonWindows.value.length || comparisonWindows.value.some(group =>
    !recorded.some(previous => previous.id === group.id && sameWindowBlocks(previous.blocks, group.blocks))))
})
const groupComparisonsReady = computed(() => !!profiler.value?.profile && !profileOutdated.value)
function guideBlockIds(id: string) { return communityWindows.value.find(group => group.id === id)?.blocks || [] }
function selectGuideGroup(id: string) {
  if (locked.value || !animaGuideSupported.value || !guideBlockIds(id).length) return
  selectedWindowId.value = id
  workspaceMode.value = 'repair'; flowStep.value = 2; tab.value = 'adjust'
  showLayers.value = true; inspectorOpen.value = true; panel.value = null
}
function blockGuideHint(id: string) { return animaGuideSupported.value ? animaLayerHint(id) : '' }
const editWindows = computed<EditWindow[]>(() => {
  const results = profiler.value?.profile?.windows || []
  return comparisonWindows.value.map(group => {
    const result = results.find(previous => previous.id === group.id && sameWindowBlocks(previous.blocks, group.blocks)) || null
    return { ...group, result, note: result?.text || '尚未生成对照', tag: result?.tag || '' }
  })
})
const selectedWindow = computed(() => editWindows.value.find(w => w.id === selectedWindowId.value) || editWindows.value[0] || null)
const selectedBlocks = computed(() => info.value?.groups.flatMap(g => g.blocks).filter(b => selectedWindow.value?.blocks.includes(b.id)) || [])
function windowStrength(ids: string[]) {
  const value = blocks[ids[0] || ''] ?? 1
  return ids.every(id => blocks[id] === value) ? value : null
}
const selectedStrength = computed(() => selectedWindow.value ? windowStrength(selectedWindow.value.blocks) : 1)
const selectedPictures = computed(() => {
  const result = selectedWindow.value?.result, report = profiler.value?.profile
  return result && report ? [
    { label: '原版 LoRA', picture: report.baseline },
    { label: '仅保留本组', picture: result.alone },
    { label: '关闭本组', picture: result.removed },
  ] : []
})
const canAdoptProfile = computed(() => current.value && !!profiler.value?.recipe && profiler.value.info.fingerprint === info.value?.fingerprint && ready.value && !profileLayoutOutdated.value && !locked.value)
async function useProfileBaseline(picture: LoRAProfilePicture) {
  const api = window.pywebview?.api, fingerprint = info.value?.fingerprint, id = profileTaskId.value
  if (!api || !fingerprint || !id || !canAdoptProfile.value) return
  starting.value = true; error.value = ''
  try {
    const result = await api.use_lora_workbench_profile_baseline(id, picture.name, fingerprint, { ...exploreConfig, frozen: [...exploreConfig.frozen] })
    if (!result.ok || !result.baseline || !result.explore || !result.data_url) { fail(result.error || '探索基准设置失败。'); return }
    if (!props.open || info.value?.fingerprint !== fingerprint) return
    remember(); Object.assign(blocks, result.baseline.blocks); Object.assign(preview, result.baseline.recipe)
    Object.assign(exploreConfig, result.explore)
    if (profiler.value?.profile) Object.assign(profileConfig, { class_prompt: profiler.value.profile.class_prompt, references: [...profiler.value.profile.references] })
    profileBaseline.value = result.baseline
    exploration.value = null; explorePictures.value = []; pickedVariant.value = ''
    exploreBaseline.value = { url: result.data_url, label: `已选基准 · ${result.baseline.label}` }
    workspaceMode.value = 'explore'; flowStep.value = 2; panel.value = null
    taskId.value = id; appliedTask = id
    message.value = '基准已设置，点击“生成四个变体”开始探索。'
  } catch { fail('探索基准设置失败，请重试。') }
  finally { starting.value = false }
}
const missingModels = computed(() => ['dit', 'te', 'vae'].filter(key => !String(preview[key as keyof LoRAPreview]).trim()).length)
const previewConfigured = computed(() => current.value && info.value?.can_preview && !missingModels.value && !!preview.prompt.trim())
const canProfile = computed(() => previewConfigured.value && preview.strength !== 0)
const primaryLabel = computed(() => {
  if (tab.value === 'compress') return '压缩并另存'
  if (flowStep.value === 0) return current.value ? '配置分析 →' : '读取 LoRA →'
  if (flowStep.value === 1) return groupComparisonsReady.value ? '重新生成分组对照 →' : '生成分组对照 →'
  if (flowStep.value === 2 && workspaceMode.value === 'explore') return exploration.value ? '继续探索' : '生成四个变体'
  if (flowStep.value === 2 && info.value?.can_preview && !groupComparisonsReady.value && !modified.value) return '生成分组对照 →'
  if (flowStep.value === 2) return previewConfigured.value ? '生成调整版对照 →' : '对照与导出 →'
  return adjustedOutput.value ? '再次另存调整版' : '另存调整版'
})
const primaryDisabled = computed(() => {
  if (locked.value || !ready.value) return true
  if (tab.value === 'compress') return !current.value || !info.value?.can_compress || rank.value < 1 || rank.value >= maxRank.value
  if (flowStep.value === 0) return !source.value.trim()
  if (flowStep.value === 1) return !canProfile.value
  if (flowStep.value === 2 && workspaceMode.value === 'explore') return !previewConfigured.value || preview.strength === 0 || !activeExploreBlocks.value.length
  return !current.value
})
const nextHint = computed(() => {
  if (locked.value) return message.value || '正在处理…'
  if (tab.value === 'compress') return compression.value ? '压缩完成，已保存新文件。' : '压缩原始 LoRA，当前分层倍率不参与。'
  if (flowStep.value === 0) return current.value ? '文件已读取，可以配置分析。' : '先选择一个 LoRA 文件。'
  if (flowStep.value === 1 && !info.value?.can_preview) return '此文件暂不支持完整出图分析，可跳过并手动调整。'
  if (flowStep.value === 1) return missingModels.value ? `还需选择 ${missingModels.value} 个预览模型文件。` : !preview.prompt.trim() ? '填写验证提示词后开始分析。' : preview.strength === 0 ? '分析时 LoRA 使用强度不能为 0。' : `将生成 ${profileCount.value ?? '多'} 张分组对照图。`
  if (flowStep.value === 2 && workspaceMode.value === 'explore') return !previewConfigured.value ? '先配置底模与验证提示词。' : !activeExploreBlocks.value.length ? '所有层均已冻结，请先解冻。' : '每轮生成 1 张基准与 4 张变体，选择后继续探索。'
  if (flowStep.value === 2 && info.value?.can_preview && !groupComparisonsReady.value && !modified.value) return previewConfigured.value ? `按 ${comparisonWindows.value.length} 组生成 ${profileCount.value} 张对照图。` : '先配置底模与验证提示词，再生成分组对照。'
  if (flowStep.value === 2) return `已调整 ${modified.value} 层${previewConfigured.value ? '，生成对照检查效果。' : '，可继续另存或返回配置预览。'}`
  return adjustedOutput.value ? (exportOutdated.value ? '倍率已改变，请重新另存调整版。' : '调整版已另存，原始 LoRA 保留。') : outdated.value ? '倍率或采样设置已改变，请重新生成对照。' : '确认效果后另存为新的 LoRA。'
})
function goStep(value: number) {
  if (locked.value || (value !== 0 && !current.value)) return
  tab.value = 'adjust'; panel.value = null; flowStep.value = value
}
async function primaryAction() {
  if (primaryDisabled.value) return
  if (tab.value === 'compress') { await start('compress'); return }
  if (flowStep.value === 0) { if (current.value) goStep(1); else await start('inspect') }
  else if (flowStep.value === 1) await start('profile')
  else if (flowStep.value === 2) { if (workspaceMode.value === 'explore') await rollExploration(); else if (info.value?.can_preview && !groupComparisonsReady.value && !modified.value) await generateGroupComparison(); else if (previewConfigured.value) await start('preview'); else goStep(3) }
  else await start('export')
}
async function generateGroupComparison() {
  if (locked.value || !current.value || !info.value?.can_preview) return
  if (!previewConfigured.value) { goStep(1); message.value = '配置底模与验证提示词后，生成分组对照。'; return }
  if (preview.strength === 0) { fail('生成分组对照时，LoRA 使用强度不能为 0。'); return }
  await start('profile')
}
function setSelectedStrength(raw: string, keepUndo = true) {
  if (locked.value || !selectedWindow.value || !raw.trim()) return
  const value = Number(raw)
  if (!Number.isFinite(value) || value < -3 || value > 3) { fail('倍率应为 -3 到 3。'); return }
  if (keepUndo) remember()
  selectedWindow.value.blocks.forEach(id => { blocks[id] = value })
}
function setBlockStrength(id: string, raw: string) {
  if (locked.value || !raw.trim()) return
  const value = Number(raw)
  if (!Number.isFinite(value) || value < -3 || value > 3) { fail('倍率应为 -3 到 3。'); return }
  remember(); blocks[id] = value
}
function openPanel(value: WorkbenchPanel) { panelOpener = document.activeElement as HTMLElement | null; panel.value = value }
function showLarge(url: string, label: string) { panelOpener = document.activeElement as HTMLElement | null; fullProfileImage.value = { url, label } }
const suggestions = computed(() => profileLayoutOutdated.value ? [] : profiler.value?.profile?.windows.filter(w => w.blocks.some(id => profiler.value?.profile?.suggested_blocks[id] !== 1)) || [])
function invalidate() { profileBaseline.value = null; undoPreview.value = null; workspaceMode.value = 'repair'; exploration.value = null; explorePictures.value = []; exploreBaseline.value = null; exploreConfig.frozen = []; exploreConfig.last_pick = []; exploreConfig.state = {}; pickedVariant.value = ''; undoExplore.value = null; flowStep.value = 0; selectedWindowId.value = ''; panel.value = null; profiler.value = null; profileTaskId.value = ''; infoTaskId.value = ''; undoBlocks.value = null; fullProfileImage.value = null; profileLoads.clear(); Object.keys(canvasImages).forEach(key => { delete canvasImages[key] }); analysis.value = null; info.value = null; images.value = []; previewReport.value = null; output.value = ''; exportReport.value = null; compression.value = undefined }
function stopPolling() { if (timer) clearTimeout(timer); timer = undefined }
function fail(text: string) { error.value = text }
function reset() { remember(); Object.keys(blocks).forEach(k => { blocks[k] = 1 }); exploreConfig.last_pick = []; exploreConfig.state = {}; pickedVariant.value = '' }
function remember() { undoBlocks.value = { ...blocks }; undoPreview.value = { ...preview }; undoExplore.value = { last_pick: [...exploreConfig.last_pick], picked: pickedVariant.value, state: { ...exploreConfig.state } } }
function rememberSlider(event: KeyboardEvent) { if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown'].includes(event.key)) remember() }
function undo() { if (undoBlocks.value) { Object.assign(blocks, undoBlocks.value); if (undoPreview.value) Object.assign(preview, undoPreview.value); undoPreview.value = null; if (undoExplore.value) { exploreConfig.last_pick = [...undoExplore.value.last_pick]; pickedVariant.value = undoExplore.value.picked; exploreConfig.state = { ...undoExplore.value.state } }; undoBlocks.value = null; undoExplore.value = null } }
function applySuggestions() {
  if (locked.value || profileOutdated.value || !profiler.value?.profile || !current.value) return
  remember(); Object.assign(blocks, profiler.value.profile.suggested_blocks)
  message.value = '已应用建议，请生成对照确认效果。'; panel.value = null
}
function applyPreset() {
  const p = selectedPreset.value
  if (!p || !current.value || locked.value || !matchesPreset(p)) { fail('预设与当前模型或层级结构不匹配。'); return }
  remember(); Object.assign(blocks, p.blocks); panel.value = null
  message.value = p.fingerprint === info.value?.fingerprint ? '已应用预设。' : '已应用其他 LoRA 的结构预设，请生成对照确认效果。'
}
async function loadPresets() {
  try {
    const result = await window.pywebview?.api.get_lora_workbench_presets()
    if (result?.ok) presets.value = result.presets || []
    else if (result) fail(result.error || '预设读取失败。')
  } catch { fail('预设读取失败。') }
}
async function savePreset() {
  if (locked.value || !current.value || !infoTaskId.value) return
  starting.value = true; error.value = ''
  try {
    const result = await window.pywebview?.api.save_lora_workbench_preset(presetName.value, infoTaskId.value, { ...blocks })
    if (result?.ok) { presets.value = result.presets || []; presetId.value = result.id || ''; presetName.value = ''; message.value = '预设已保存。' }
    else fail(result?.error || '预设保存失败。')
  } catch { fail('预设保存失败。') }
  finally { starting.value = false }
}
async function deletePreset() {
  if (locked.value || !presetId.value) return
  starting.value = true; error.value = ''
  try {
    const result = await window.pywebview?.api.delete_lora_workbench_preset(presetId.value)
    if (result?.ok) { presets.value = result.presets || []; presetId.value = ''; message.value = '预设已删除。' }
    else fail(result?.error || '预设删除失败。')
  } catch { fail('预设删除失败。') }
  finally { starting.value = false }
}
async function chooseReference() {
  if (locked.value || family.value !== 'krea2' || profileConfig.references.length >= 3) return
  try {
    const result = await window.pywebview?.api.choose_path('image', '', 'lora_workbench_reference')
    if (result?.ok && result.path && !result.cancelled && !profileConfig.references.includes(result.path)) profileConfig.references.push(result.path)
    else if (result && !result.ok) fail(result.error || '参考照片选择失败。')
  } catch { fail('参考照片选择失败。') }
}
async function showProfileImage(picture: LoRAProfilePicture, label: string) {
  if (loadingImage.value || !profileTaskId.value) return
  const id = profileTaskId.value
  const cached = profileLoads.get(`${id}/${picture.name}`)
  if (cached) { showLarge(cached, label); return }
  loadingImage.value = true
  try {
    const result = await window.pywebview?.api.get_lora_workbench_image(id, picture.name)
    if (id !== profileTaskId.value) return
    if (result?.ok && result.data_url) { profileLoads.set(`${id}/${picture.name}`, result.data_url); showLarge(result.data_url, label) }
    else fail(result?.error || '分析图片读取失败。')
  } catch { fail('分析图片读取失败。') }
  finally { loadingImage.value = false }
}
function keys(event: KeyboardEvent) {
  if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); if (fullProfileImage.value) fullProfileImage.value = null; else if (panel.value) panel.value = null; else emit('close'); return }
  if (event.key !== 'Tab' || !dialog.value) return
  const focusRoot = fullProfileImage.value ? dialog.value.querySelector<HTMLElement>('.lwb-large-image') || dialog.value : panel.value ? dialog.value.querySelector<HTMLElement>('.lwb-sidepanel') || dialog.value : dialog.value
  const nodes = [...focusRoot.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled),textarea:not(:disabled),select:not(:disabled),summary,a[href]')].filter(n => n.getClientRects().length)
  const first = nodes[0], last = nodes[nodes.length - 1]
  if (!first || !last) { event.preventDefault(); dialog.value.focus(); return }
  if (!focusRoot.contains(document.activeElement)) { event.preventDefault(); (event.shiftKey ? last : first).focus(); return }
  if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.value)) { event.preventDefault(); last.focus() }
  else if (!event.shiftKey && (document.activeElement === last || document.activeElement === dialog.value)) { event.preventDefault(); first.focus() }
}
async function choose(key: 'source' | 'dit' | 'te' | 'vae') {
  if (!props.desktop || !window.pywebview?.api || locked.value) return
  try {
    const result = await window.pywebview.api.choose_path('model', key === 'source' ? source.value : preview[key], `lora_workbench_${key}`)
    if (!result.ok) { fail(result.error || '文件选择失败。'); return }
    if (result.cancelled || !result.path) return
    if (key === 'source') { source.value = result.path; invalidate(); void start('inspect') }
    else { preview[key] = result.path; if (key === 'dit' && family.value === 'sdxl') preview.te = preview.vae = result.path }
  } catch { fail('文件选择失败，请重试。') }
}
async function defaults() {
  if (!window.pywebview?.api || family.value === 'auto') return
  const requestedFamily = family.value
  try {
    const result = await window.pywebview.api.get_lora_workbench(props.projectName, requestedFamily)
    if (family.value !== requestedFamily) return
    if (result.ok) { Object.assign(preview, result.defaults); message.value = '已读取本机模型配置。' }
    else fail(result.error || '模型配置读取失败。')
  } catch { fail('模型配置读取失败，请重试。') }
}
async function applyReport(report: LoRAWorkbenchReport, id: string, restore = false) {
  if (report.info) {
    infoTaskId.value = id
    info.value = report.info
    if (restore || report.action === 'inspect') {
      source.value = report.info.source
      family.value = report.info.family
      const savedBlocks = { ...blocks }
      Object.keys(blocks).forEach(k => { delete blocks[k] })
      for (const group of report.info.groups) for (const block of group.blocks) blocks[block.id] = restore ? (report.blocks?.[block.id] ?? savedBlocks[block.id] ?? 1) : 1
      rank.value = Math.max(1, Math.min(8, Math.floor(maxRank.value / 2)))
      if (report.action === 'inspect') { profiler.value = null; fullProfileImage.value = null; profileLoads.clear(); Object.keys(canvasImages).forEach(key => { delete canvasImages[key] }); undoBlocks.value = null; analysis.value = null; images.value = []; previewReport.value = null; if (previewFamily.value !== report.info.family || !preview.dit) await defaults(); previewFamily.value = report.info.family }
    }
  }
  if (report.exploration && report.recipe) {
    if (restore) { Object.assign(preview, report.recipe); previewFamily.value = report.info.family }
    await readExploration(report, id, restore); workspaceMode.value = 'explore'; flowStep.value = 2
  } else if (report.images && report.recipe) {
    previewReport.value = report
    if (restore) { previewFamily.value = report.info.family; Object.assign(preview, report.recipe); if (report.blocks) Object.assign(blocks, report.blocks) }
    images.value = []
    for (const item of report.images) {
      const result = await window.pywebview?.api.get_lora_workbench_image(id, item.name)
      if (result?.ok && result.data_url) images.value.push({ label: item.label, url: result.data_url })
      else fail(result?.error || '对照图读取失败。')
    }
  }
  if (report.profile) {
    profiler.value = report; profileTaskId.value = id; fullProfileImage.value = null; profileLoads.clear(); Object.keys(canvasImages).forEach(key => { delete canvasImages[key] })
    if (restore) { Object.assign(preview, report.recipe); Object.assign(profileConfig, { class_prompt: report.profile.class_prompt, references: report.profile.references }); previewFamily.value = report.info.family }
  }
  if (report.analysis) { analysis.value = report.analysis; showLayers.value = true; flowStep.value = 2 }
  if (report.output) { output.value = report.output; compression.value = report.compression; if (report.action === 'export') exportReport.value = report }
  if (report.action === 'inspect') flowStep.value = 1
  else if (report.action === 'profile') flowStep.value = 2
  else if (report.action === 'preview' || report.action === 'export') flowStep.value = 3
  else if (report.action === 'compress') tab.value = 'compress'
}
async function poll(restore = false) {
  stopPolling()
  const api = window.pywebview?.api, id = taskId.value
  if (!api || !props.open || !id) return
  try {
    const result = await api.get_lora_workbench_result(id)
    if (!props.open || id !== taskId.value) return
    if (!result.ok) { fail(result.error || '记录读取失败。'); busy.value = false; return }
    busy.value = result.task?.status === 'running'
    message.value = result.task?.message || ''
    progress.value = result.task?.progress ?? null
    if (result.task?.status === 'failed') fail(result.task.message)
    if (result.report && !result.report.ok) fail(result.report.error || '工作台处理失败。')
    if (result.report?.ok && appliedTask !== id && (!result.task || result.task.status === 'completed')) {
      appliedTask = id
      applyingReport.value = true
      try { await applyReport(result.report, id, restore) } finally { applyingReport.value = false }
    }
    if (busy.value) timer = setTimeout(() => { void poll(restore) }, 850)
  } catch { fail('工作台进度读取失败，可关闭后重新打开继续查看。'); busy.value = false }
}
async function start(action: 'inspect' | 'analyze' | 'profile' | 'preview' | 'explore' | 'export' | 'compress') {
  if (locked.value || !ready.value) return
  if (!props.desktop || !window.pywebview?.api) { fail('请在桌面程序中使用 LoRA 工作台。'); return }
  if (action !== 'inspect' && !current.value) { fail('请先读取 LoRA。'); return }
  starting.value = true; error.value = ''; output.value = ''; compression.value = undefined
  try {
    const result = await window.pywebview.api.start_lora_workbench(action, { source: source.value, family: family.value, fingerprint: info.value?.fingerprint, blocks: { ...blocks }, rank: rank.value, explore: { ...exploreConfig, frozen: [...exploreConfig.frozen], last_pick: [...exploreConfig.last_pick] }, preview: { ...preview }, profile: { class_prompt: profileConfig.class_prompt, references: [...profileConfig.references] } })
    if (!result.ok) { fail(result.error || '任务未能启动。'); return }
    if (result.cancelled) { message.value = '已取消另存。'; return }
    taskId.value = result.task_id || ''; busy.value = true; message.value = '正在启动…'; progress.value = null
    await poll()
  } catch { fail('任务未能启动，请查看运行日志。') }
  finally { starting.value = false }
}
async function cancel() {
  if (!window.pywebview?.api || !taskId.value) return
  const result = await window.pywebview.api.cancel_task(taskId.value)
  if (!result.ok) fail(result.error || '停止请求失败。')
  else message.value = '正在停止…'
}
async function load() {
  const revision = ++loadRevision
  if (!props.desktop || !window.pywebview?.api) { reason.value = '请在桌面程序中使用；浏览器预览不会操作本机文件。'; return }
  try {
    const result = await window.pywebview.api.get_lora_workbench(props.projectName)
    if (revision !== loadRevision || !props.open) return
    if (!result.ok) { fail(result.error || '工作台无法打开。'); return }
    ready.value = result.ready; reason.value = result.reason; families.value = result.families; recent.value = result.recent
    if (!source.value && result.saved.source) {
      source.value = result.saved.source; family.value = result.saved.family
      if (result.saved.preview) { Object.assign(preview, result.saved.preview); previewFamily.value = result.saved.family }
      if (result.saved.blocks) Object.assign(blocks, result.saved.blocks)
      if (result.saved.profile) Object.assign(profileConfig, result.saved.profile)
    }
    const restore = !info.value
    taskId.value = (result.task?.status === 'running' ? result.task.id : result.saved.last_task || result.task?.id) || taskId.value
    busy.value = result.task?.status === 'running'
    if (taskId.value) await poll(restore)
    if (!profiler.value && current.value && result.saved.last_profile_task) {
      const cached = await window.pywebview.api.get_lora_workbench_result(result.saved.last_profile_task)
      if (revision !== loadRevision || !props.open) return
      if (cached.report?.ok && cached.report.profile && cached.report.info.fingerprint === info.value?.fingerprint) {
        profiler.value = cached.report; profileTaskId.value = result.saved.last_profile_task
      }
    }
    if (!exploration.value && current.value && result.saved.last_explore_task && result.saved.last_task !== result.saved.profile_explore_baseline?.task_id) {
      const cached = await window.pywebview.api.get_lora_workbench_result(result.saved.last_explore_task)
      if (revision !== loadRevision || !props.open) return
      if (cached.report?.ok && cached.report.exploration && cached.report.info.fingerprint === info.value?.fingerprint) await readExploration(cached.report, result.saved.last_explore_task, true)
    }
    const baseline = result.saved.profile_explore_baseline
    if (baseline && !exploration.value && !profileBaseline.value && result.saved.last_task === baseline.task_id && baseline.fingerprint === info.value?.fingerprint) {
      const image = await window.pywebview.api.get_lora_workbench_image(baseline.task_id, baseline.name)
      if (revision !== loadRevision || !props.open) return
      if (image.ok && image.data_url) {
        profileBaseline.value = baseline
        exploreBaseline.value = { url: image.data_url, label: `已选基准 · ${baseline.label}` }
        Object.assign(blocks, result.saved.blocks || baseline.blocks)
        if (result.saved.preview) Object.assign(preview, result.saved.preview)
        if (result.saved.explore) Object.assign(exploreConfig, result.saved.explore)
        workspaceMode.value = 'explore'; flowStep.value = 2
      } else fail(image.error || '所选探索基准图无法恢复。')
    }
    await loadPresets()
  } catch { fail('工作台初始化失败，请重试。') }
}
watch(() => props.open, async open => {
  if (!open) { panel.value = null; fullProfileImage.value = null; ++loadRevision; stopPolling(); await nextTick(); previousFocus?.focus(); return }
  previousFocus = document.activeElement as HTMLElement | null
  await nextTick(); dialog.value?.focus(); void load()
}, { immediate: true })
watch(family, value => { if (value !== 'krea2') { profileConfig.references = []; preview.swap = 0 } })
watch(editWindows, windows => {
  if (!windows.some(w => w.id === selectedWindowId.value)) selectedWindowId.value = windows[0]?.id || ''
}, { immediate: true })
watch([flowStep, tab], async () => { panel.value = null; await nextTick(); if (mainScroll.value) mainScroll.value.scrollTop = 0 })
watch(() => selectedWindow.value?.id, async () => { focusedPicture.value = null; await nextTick(); if (mainScroll.value) mainScroll.value.scrollTop = 0 })
watch([panel, fullProfileImage], async ([drawer, image]) => {
  await nextTick()
  if (drawer || image) dialog.value?.querySelector<HTMLElement>(image ? '.lwb-large-image button' : '.lwb-sidepanel button')?.focus()
  else if (props.open) panelOpener?.focus()
})
watch([selectedWindowId, profileTaskId, flowStep, workspaceMode, () => props.open], () => { void loadCanvasImages() })
onUnmounted(stopPolling)
</script>

<template>
  <div v-if="open" class="lwb-backdrop">
    <section ref="dialog" class="lwb-dialog" role="dialog" aria-modal="true" aria-labelledby="lwb-title" tabindex="-1" @keydown="keys">
      <header class="lwb-header">
        <div class="lwb-heading"><h2 id="lwb-title">LoRA 工作台</h2><div v-if="current && info" class="lwb-header-file"><strong :title="info.source">{{ info.name }}</strong><span>{{ info.family_label }} · Rank {{ info.ranks.join(' / ') }}</span><button :disabled="locked" @click="goStep(0)">更换</button></div><span v-else class="lwb-header-subtitle">Krea 2 / Anima</span></div>
        <div class="lwb-actions">
          <button v-if="current && family === 'anima'" class="lwb-subtle" @click="openPanel('layer-guide')">层级参考</button>
          <button v-if="tab === 'adjust'" class="lwb-subtle" :disabled="locked || !current" @click="tab = 'compress'"><UiIcon name="settings" />权重压缩</button>
          <button v-else class="lwb-subtle" :disabled="locked" @click="tab = 'adjust'"><UiIcon name="back" />返回分层调整</button>
          <button class="lwb-icon-button" aria-label="关闭工作台" @click="emit('close')"><UiIcon name="close" /></button>
        </div>
      </header>
      <nav v-if="tab === 'adjust'" class="lwb-steps" aria-label="工作台操作步骤">
        <button v-for="(label, index) in stepLabels" :key="label" :class="{ active: flowStep === index, passed: flowStep > index }" :aria-current="flowStep === index ? 'step' : undefined" :disabled="locked || (index > 0 && !current)" @click="goStep(index)">
          <span class="lwb-step-number">{{ index + 1 }}</span><span>{{ label }}</span>
        </button>
      </nav>
      <div v-if="reason && !ready" class="lwb-notice lwb-environment" role="status"><span>{{ reason }}</span><button v-if="desktop" :disabled="locked" @click="emit('environment')">准备 Fizgig 环境</button></div>
      <div v-if="error" class="lwb-error" role="alert"><span>{{ error }}</span><button class="lwb-icon-button" aria-label="关闭错误提示" @click="error = ''"><UiIcon name="close" /></button></div>
      <div v-if="current && info?.warnings.length && flowStep !== 0 && tab === 'adjust'" class="lwb-notice lwb-file-warning"><p v-for="warning in info.warnings" :key="warning">{{ warning }}</p></div>
      <main class="lwb-body" :aria-busy="locked">
        <div v-if="tab === 'compress'" ref="mainScroll" class="lwb-stage lwb-scrolling">
          <div class="lwb-stage-heading"><div><h3>权重压缩</h3><p>降低原始 LoRA 的 Rank，保存为新文件。</p></div></div>
          <section class="lwb-section lwb-narrow">
            <h4>设置目标 Rank</h4>
            <fieldset :disabled="locked"><label>目标 Rank<input v-model.number="rank" type="number" min="1" :max="Math.max(1, maxRank - 1)" /></label></fieldset>
            <p class="lwb-help">当前 Rank {{ info?.ranks.join(' / ') }}；分层调整倍率不参与压缩。</p>
            <p v-if="info && !info.can_compress" class="lwb-notice">此文件的权重结构暂不支持压缩。</p>
            <div v-if="compression" class="lwb-result"><strong>压缩完成 · Rank {{ compression.rank }}</strong><p>保留权重能量 {{ pct(compression.mean_energy) }}，请出图确认效果。</p><code>{{ output }}</code></div>
          </section>
        </div>
        <div v-else-if="flowStep === 0" ref="mainScroll" class="lwb-stage lwb-scrolling">
          <div class="lwb-stage-heading"><div><h3>选择要调整的 LoRA</h3><p>支持 Krea 2、Anima 的 .safetensors LoRA。</p></div></div>
          <section class="lwb-section lwb-narrow">
            <fieldset :disabled="locked">
              <label>LoRA 文件<div class="lwb-input-action"><input v-model="source" placeholder="选择本机 LoRA 文件" @change="invalidate" /><button :disabled="!desktop" @click="choose('source')"><UiIcon name="folder" />选择文件</button></div></label>
              <div class="lwb-form-grid">
                <label>模型架构<select v-model="family" @change="invalidate"><option value="auto">自动识别</option><option v-for="(label, key) in families" :key="key" :value="key">{{ label }}</option></select></label>
                <label v-if="recent.length">当前项目的训练产物<select v-model="source" @change="invalidate"><option value="">选择训练检查点…</option><option v-for="item in recent" :key="item.path" :value="item.path">{{ item.name }}</option></select></label>
              </div>
            </fieldset>
            <div v-if="current && info" class="lwb-file-info">
              <div><span>文件大小</span><strong>{{ size(info.bytes) }}</strong></div><div><span>模型架构</span><strong>{{ info.family_label }}</strong></div><div><span>已映射权重</span><strong>{{ info.mapped_pairs }} / {{ info.pairs }}</strong></div><div><span>可调整层级</span><strong>{{ blockIds.length }} 层</strong></div>
            </div>
            <div v-if="info?.warnings.length" class="lwb-notice"><p v-for="warning in info.warnings" :key="warning">{{ warning }}</p></div>
          </section>
          <p class="lwb-route-note">读取文件后，可分析各组效果，也可直接手动调整并导出。</p>
        </div>
        <div v-else-if="flowStep === 1" ref="mainScroll" class="lwb-stage lwb-scrolling">
          <div class="lwb-stage-heading"><div><h3>配置分析场景</h3><p>固定模型与采样参数，对照各组对画面的影响。</p></div></div>
          <fieldset :disabled="locked" class="lwb-config-grid">
            <section class="lwb-section">
              <div class="lwb-section-heading"><h4>预览模型</h4><button class="lwb-small" :disabled="!ready" @click="defaults">使用已有模型配置</button></div>
              <label>底模<div class="lwb-input-action"><input v-model="preview.dit" placeholder="与 LoRA 匹配的底模" /><button :disabled="!desktop" @click="choose('dit')">选择</button></div></label>
              <label>文本编码器<div class="lwb-input-action"><input v-model="preview.te" placeholder="文本编码器文件" /><button :disabled="!desktop" @click="choose('te')">选择</button></div></label>
              <label>VAE<div class="lwb-input-action"><input v-model="preview.vae" placeholder="VAE 文件" /><button :disabled="!desktop" @click="choose('vae')">选择</button></div></label>
              <p class="lwb-help">使用本机模型文件，无需启动 ComfyUI。</p>
              <div class="lwb-rule"></div>
              <div class="lwb-form-grid lwb-three">
                <label>宽度<input v-model.number="preview.width" type="number" min="256" max="1536" step="64" /></label>
                <label>高度<input v-model.number="preview.height" type="number" min="256" max="1536" step="64" /></label>
                <label>随机种子<input v-model.number="preview.seed" type="number" min="0" /></label>
              </div>
              <details class="lwb-detail"><summary>采样与显存设置 <span>{{ preview.steps }} 步 · CFG {{ preview.cfg }}</span></summary>
                <div class="lwb-form-grid">
                  <label>采样步数<input v-model.number="preview.steps" type="number" min="1" max="100" /></label><label>CFG<input v-model.number="preview.cfg" type="number" min="0" max="20" step="0.5" /></label>
                  <label>LoRA 使用强度<input v-model.number="preview.strength" type="number" min="-3" max="3" step="0.1" /></label><label>底模精度<select v-model="preview.precision"><option value="auto">自动</option><option value="bf16">BF16</option><option value="int8">INT8</option><option value="nf4">NF4</option></select></label>
                  <label v-if="family === 'krea2'">Block Swap<input v-model.number="preview.swap" type="number" min="0" max="100" /></label>
                </div><p class="lwb-help">使用强度仅影响预览，不写入导出文件。</p>
              </details>
            </section>
            <section class="lwb-section">
              <h4>验证提示词</h4>
              <label class="lwb-visually-hidden" for="lwb-prompt">验证提示词</label><textarea id="lwb-prompt" v-model="preview.prompt" class="lwb-prompt" placeholder="填写场景、人物及 LoRA 触发词…" rows="5"></textarea>
              <p class="lwb-help">各组共用这段提示词，先用常用场景分析。</p>
              <details class="lwb-detail"><summary>负向提示词</summary><textarea v-model="preview.negative" aria-label="负向提示词" placeholder="可选" rows="3"></textarea></details>
              <details class="lwb-detail"><summary>扩展分析（可选）</summary>
                <label>无触发词的场景<textarea v-model="profileConfig.class_prompt" placeholder="去掉角色触发词，保留场景描述" rows="3"></textarea></label>
                <p class="lwb-help">追加同场景对照，生成张数加倍。</p>
                <template v-if="family === 'krea2'">
                  <div class="lwb-section-heading"><h4>人物参考照片</h4><button class="lwb-small" :disabled="profileConfig.references.length >= 3 || !desktop" @click="chooseReference">添加照片</button></div>
                  <div v-for="(path, index) in profileConfig.references" :key="path" class="lwb-reference"><span :title="path">{{ path.split(/[\\/]/).pop() }}</span><button class="lwb-icon-button" :aria-label="`移除参考照片 ${index + 1}`" @click="profileConfig.references.splice(index, 1)"><UiIcon name="close" /></button></div>
                  <p class="lwb-help">最多 3 张。人脸分析需已安装 ArcFace 依赖及 buffalo_l 模型。</p>
                </template>
              </details>
              <div class="lwb-analysis-summary"><strong>{{ profileCount ?? '多' }} 张对照图</strong><span>固定种子 · 原版 / 停用 / 分组保留 / 分组关闭</span></div>
            </section>
          </fieldset>
        </div>
        <div v-else-if="flowStep === 2" class="lwb-stage lwb-adjust-stage">
          <div class="lwb-workspace-toolbar">
            <div class="lwb-mode-switch" aria-label="编辑模式">
              <button :class="{ active: workspaceMode === 'repair' }" :aria-pressed="workspaceMode === 'repair'" :disabled="locked" @click="workspaceMode = 'repair'">分层调整</button>
              <button :class="{ active: workspaceMode === 'explore' }" :aria-pressed="workspaceMode === 'explore'" :disabled="locked || !info?.can_preview" @click="workspaceMode = 'explore'">变体探索</button>
            </div>
            <div class="lwb-actions lwb-toolbar">
              <button v-if="workspaceMode === 'repair'" :disabled="locked || !profiler?.profile" @click="openPanel('suggestions')">分析建议<span v-if="suggestions.length" class="lwb-count">{{ suggestions.length }}</span></button>
              <button :disabled="locked" @click="openPanel('presets')">预设</button><button :disabled="locked || !undoBlocks" @click="undo">撤销</button><button :disabled="locked || !modified" @click="reset">恢复原版</button>
              <button :aria-pressed="inspectorOpen" @click="inspectorOpen = !inspectorOpen">{{ inspectorOpen ? '收起参数' : '展开参数' }}</button>
            </div>
          </div>
          <div :class="['lwb-studio-layout', { 'is-explore': workspaceMode === 'explore', 'inspector-hidden': !inspectorOpen }]">
            <aside v-if="workspaceMode === 'repair'" class="lwb-group-rail" aria-label="层级分组">
              <div class="lwb-rail-title"><span>层级分组</span><span>{{ editWindows.length }} 组</span></div>
              <div v-if="animaGuideSupported" class="lwb-grouping-control"><span>社区分组</span><button class="lwb-guide-link" @click="openPanel('layer-guide')">查看依据</button></div>
              <div class="lwb-group-list"><button v-for="group in editWindows" :key="group.id" :class="['lwb-group-button', { selected: selectedWindow?.id === group.id }]" :aria-pressed="selectedWindow?.id === group.id" :disabled="locked" @click="selectedWindowId = group.id">
                <span class="lwb-group-top"><strong>{{ group.label }}</strong><span class="lwb-group-value" :class="{ changed: group.blocks.some(id => blocks[id] !== 1) }">{{ windowStrength(group.blocks) == null ? '混合' : `${windowStrength(group.blocks)}×` }}</span></span>
                <span class="lwb-group-note">{{ group.community ? group.range || '独立组件' : profileOutdated ? '上次结果' : group.note }}</span>
              </button></div>
              <div class="lwb-rail-footer"><span>{{ modified }} 层已调整</span><button :disabled="locked" @click="goStep(1)">分析设置</button></div>
            </aside>
            <section class="lwb-viewer" :aria-label="workspaceMode === 'repair' ? '分组图片对照' : '变体探索画布'">
              <template v-if="workspaceMode === 'repair'">
                <div class="lwb-canvas-toolbar"><strong>{{ selectedWindow?.label || '分组对照' }}</strong>
                  <div v-if="selectedPictures.length" class="lwb-view-controls"><button :class="{ active: focusedPicture == null }" @click="focusedPicture = null">三图</button><button v-for="(picture, index) in selectedPictures" :key="picture.label" :class="{ active: focusedPicture === index }" @click="focusedPicture = index">{{ picture.label }}</button></div><button v-if="selectedPictures.length" class="lwb-small" :disabled="locked || !ready" @click="generateGroupComparison">重新生成 · {{ profileCount }} 张</button>
                </div>
                <div v-if="profileOutdated" class="lwb-canvas-status">{{ profileLayoutOutdated ? '分组已更新，请重新生成分组对照。' : '采样条件已改变，显示上次结果。' }}<button :disabled="locked" @click="generateGroupComparison">重新生成</button></div>
                <div v-if="visiblePictures.length" :class="['lwb-canvas-pictures', { single: focusedPicture != null }]">
                  <figure v-for="picture in visiblePictures" :key="picture.label"><button class="lwb-picture-button" :disabled="loadingImage" :aria-label="`放大${picture.label}`" @click="showProfileImage(picture.picture, picture.label)"><img :src="pictureSource(picture.picture)" :alt="picture.label" /></button><figcaption class="lwb-profile-caption"><span>{{ picture.label }}</span><button :disabled="!canAdoptProfile" title="采用此图的完整层级倍率和采样设置，进入变体探索" @click="useProfileBaseline(picture.picture)">以此为基准探索</button></figcaption></figure>
                </div>
                <div v-else class="lwb-empty lwb-studio-empty"><UiIcon name="image" /><strong>{{ selectedWindow?.community ? `${selectedWindow.range || ''} ${selectedWindow.label}` : '尚无分组对照' }}</strong><p>{{ selectedWindow?.summary || '固定提示词与种子，查看各组的实际影响。' }}<br />原版 / 仅保留本组 / 关闭本组</p><button :disabled="locked || !ready || !info?.can_preview" @click="generateGroupComparison">生成分组对照<span v-if="profileCount"> · {{ profileCount }} 张</span></button></div>
                <div class="lwb-canvas-footnote">分组对照使用原始 LoRA；修改倍率后生成调整版对照。</div>
              </template>
              <template v-else>
                <div class="lwb-canvas-toolbar"><strong>LoRA 变体探索</strong><span>{{ explorePictures.length ? '选择一个变体，继续迭代' : '同一场景 · 四个候选' }}</span></div>
                <div v-if="explorationOutdated || profileBaselineOutdated" class="lwb-canvas-status">参数已改变，请生成新一轮基准与变体。</div>
                <div v-if="currentExploreImage" class="lwb-explorer-canvas">
                  <figure class="lwb-explorer-baseline"><button class="lwb-picture-button" :aria-label="`放大${currentExploreImage.label}`" @click="showLarge(currentExploreImage.url, currentExploreImage.label)"><img :src="currentExploreImage.url" :alt="currentExploreImage.label" /></button><figcaption>{{ currentExploreImage.label }}<span>种子 {{ exploration?.recipe?.seed ?? profileBaseline?.recipe.seed ?? preview.seed }}</span></figcaption></figure>
                  <div v-if="explorePictures.length" class="lwb-variant-grid">
                    <figure v-for="variant in explorePictures" :key="variant.name" :class="{ picked: pickedVariant === variant.name && sameBlocks(variant.blocks) }">
                      <button class="lwb-picture-button" :aria-label="`放大${variant.label}`" @click="showLarge(variant.url, variant.label)"><img :src="variant.url" :alt="variant.label" /></button>
                      <figcaption><span>{{ variant.label }}<small>{{ variant.changed.length }} 层变化</small></span><button :disabled="!canPick(variant) || sameBlocks(variant.blocks)" :title="!canPick(variant) && !locked ? '参数已改变或该候选会改变冻结层，请重新生成' : ''" @click="pickVariant(variant)">{{ sameBlocks(variant.blocks) ? '当前基准' : '选为基准' }}</button></figcaption>
                    </figure>
                  </div>
                  <div v-else class="lwb-empty lwb-await-variants"><UiIcon name="image" /><strong>分析图已设为探索基准</strong><p>{{ profileBaseline?.label }} · 已带入完整层级配置<br />点击“生成四个变体”继续。</p></div>
                </div>
                <div v-else class="lwb-empty lwb-studio-empty"><UiIcon name="image" /><strong>看图选择新的调整方向</strong><p>官方 Explorer 生成四个变体，选中后继续探索。</p><button :disabled="locked" @click="goStep(1)">{{ previewConfigured ? '查看生成设置' : '配置底模与提示词' }}</button></div>
                <div class="lwb-canvas-footnote">候选仅调整层级权重。选为基准后，点击“继续探索”生成下一轮。</div>
              </template>
            </section>
            <aside v-if="inspectorOpen" class="lwb-inspector lwb-scrolling" aria-label="调整参数">
              <template v-if="workspaceMode === 'repair' && selectedWindow">
                <div class="lwb-inspector-title"><h4>{{ selectedWindow.label }}</h4><span>{{ selectedWindow.blocks.length }} 层</span></div>
                <p class="lwb-help">{{ selectedWindow.community ? `${selectedWindow.range || '独立组件'} · ${selectedWindow.summary}` : profileOutdated ? '上次分析结果' : selectedWindow.note }}</p>
                <p v-if="selectedWindow.community && selectedWindow.result" class="lwb-help">{{ profileOutdated ? '上次结果：' : '' }}{{ selectedWindow.note }}</p>
                <section class="lwb-inspector-section"><div class="lwb-section-heading"><h4>本组倍率</h4><span>{{ selectedStrength == null ? '混合' : `${selectedStrength}×` }}</span></div>
                  <input type="range" min="-3" max="3" step="0.05" :value="selectedStrength ?? 1" :disabled="locked" aria-label="本组倍率" @pointerdown="remember" @keydown="rememberSlider" @input="setSelectedStrength(($event.target as HTMLInputElement).value, false)" />
                  <div class="lwb-range-labels"><span>−3</span><input type="number" min="-3" max="3" step="0.05" :value="selectedStrength ?? ''" :placeholder="selectedStrength == null ? '混合' : ''" :disabled="locked" aria-label="本组倍率数值" @change="setSelectedStrength(($event.target as HTMLInputElement).value)" /><span>+3</span></div>
                  <div class="lwb-actions lwb-quick-values"><button :disabled="locked" @click="setSelectedStrength('0.85')">0.85</button><button :disabled="locked" @click="setSelectedStrength('0.7')">0.7</button><button :disabled="locked" @click="setSelectedStrength('0')">关闭</button><button :disabled="locked" @click="setSelectedStrength('1')">恢复</button></div>
                  <p v-if="selectedStrength == null" class="lwb-help">设置倍率会统一本组所有层。</p>
                  <label class="lwb-checkbox"><input type="checkbox" :checked="selectedWindow.blocks.every(id => exploreConfig.frozen.includes(id))" :indeterminate="selectedWindow.blocks.some(id => exploreConfig.frozen.includes(id)) && !selectedWindow.blocks.every(id => exploreConfig.frozen.includes(id))" :disabled="locked" @change="toggleFrozen(selectedWindow.blocks, ($event.target as HTMLInputElement).checked)" />变体探索时冻结本组</label>
                </section>
                <section v-if="selectedWindow.result" class="lwb-inspector-section"><h4>画面变化</h4><div class="lwb-stat-list"><div><span>仅保留本组</span><strong>{{ pct(selectedWindow.result.alone_change) }}</strong></div><div><span>关闭本组</span><strong>{{ pct(selectedWindow.result.removed_change) }}</strong></div><div v-if="selectedWindow.result.likeness_share != null"><span>人脸相似度贡献</span><strong>{{ pct(selectedWindow.result.likeness_share) }}</strong></div><div v-if="selectedWindow.result.bleed_share != null"><span>特征泄漏</span><strong>{{ pct(selectedWindow.result.bleed_share) }}</strong></div></div><p class="lwb-help">变化量相对原版，不代表画质。</p></section>
                <details class="lwb-detail" :open="showLayers" @toggle="showLayers = ($event.target as HTMLDetailsElement).open"><summary>逐层调整<span>{{ selectedBlocks.length }} 层</span></summary><button class="lwb-small" :disabled="locked" @click="start('analyze')">{{ analysis ? '重新分析权重' : '分析权重规模' }}</button>
                  <div v-for="block in selectedBlocks" :key="block.id" class="lwb-layer-control"><div><strong>{{ block.label }}</strong><input type="number" min="-3" max="3" step="0.05" :value="blocks[block.id]" :disabled="locked" :aria-label="`${block.label} 倍率数值`" @change="setBlockStrength(block.id, ($event.target as HTMLInputElement).value)" /></div><input type="range" min="-3" max="3" step="0.05" :value="blocks[block.id]" :disabled="locked" :aria-label="`${block.label} 倍率`" @pointerdown="remember" @keydown="rememberSlider" @input="blocks[block.id] = Number(($event.target as HTMLInputElement).value)" /><p v-if="blockGuideHint(block.id)" class="lwb-help lwb-layer-hint">社区观察：{{ blockGuideHint(block.id) }}</p><p class="lwb-help">{{ analysis ? `权重能量占比 ${energy(block.id).toFixed(1)}%` : `${block.modules} 个权重模块` }}</p><label class="lwb-checkbox"><input type="checkbox" :disabled="locked" :checked="exploreConfig.frozen.includes(block.id)" @change="toggleFrozen([block.id], ($event.target as HTMLInputElement).checked)" />探索时冻结</label></div><p v-if="analysis" class="lwb-help">权重规模不代表视觉贡献。</p>
                </details>
                <details v-if="profiler?.profile" class="lwb-detail"><summary>底模基线</summary><div class="lwb-baseline-pictures"><figure><button class="lwb-picture-button" :disabled="loadingImage" @click="showProfileImage(profiler.profile.off, '停用 LoRA')"><img :src="profiler.profile.off.thumbnail" alt="停用 LoRA" /></button><figcaption>停用 LoRA</figcaption></figure><figure><button class="lwb-picture-button" :disabled="loadingImage" @click="showProfileImage(profiler.profile.baseline, '原版 LoRA')"><img :src="profiler.profile.baseline.thumbnail" alt="原版 LoRA" /></button><figcaption>原版 LoRA</figcaption></figure></div></details>
                <details v-if="selectedWindow.result?.class_alone" class="lwb-detail"><summary>无触发词场景</summary><button class="lwb-picture-button" :disabled="loadingImage" @click="showProfileImage(selectedWindow.result.class_alone, '无触发词场景')"><img :src="selectedWindow.result.class_alone.thumbnail" alt="无触发词场景" /></button></details>
              </template>
              <template v-else-if="workspaceMode === 'explore'">
                <div class="lwb-inspector-title"><h4>探索参数</h4><span>官方 Explorer</span></div>
                <section class="lwb-inspector-section">
                  <label>权重扰动幅度 <span class="lwb-control-value">{{ Math.round(exploreConfig.intensity * 100) }}%</span><input v-model.number="exploreConfig.intensity" :disabled="locked" type="range" min="0" max="1" step="0.05" /></label>
                  <label>层级结构扰动 <span class="lwb-control-value">{{ Math.round(exploreConfig.structure * 100) }}%</span><input v-model.number="exploreConfig.structure" :disabled="locked" type="range" min="0" max="1" step="0.05" /></label>
                  <p class="lwb-help">加强前两个候选的一层反转或极值变化，不保证改变构图。权重扰动幅度需超过 30%。</p>
                  <label>每个候选变化层数<input v-model.number="exploreConfig.mutations" :disabled="locked" type="number" min="1" :max="Math.max(1, activeExploreBlocks.length)" /></label>
                  <details class="lwb-detail"><summary>随机参数</summary><label>变体随机种子<input v-model.number="exploreConfig.mutation_seed" :disabled="locked" type="number" min="0" max="4294967295" /></label><button class="lwb-small" :disabled="locked || !previewConfigured || !activeExploreBlocks.length || preview.strength === 0" @click="start('explore')">按此种子生成</button><p class="lwb-help">“继续探索”会更换变体种子，采样种子保持不变。</p></details>
                </section>
                <section class="lwb-inspector-section"><div class="lwb-section-heading"><h4>冻结层级</h4><span>{{ exploreConfig.frozen.length }} / {{ blockIds.length }}</span></div>
                  <label v-for="group in editWindows" :key="group.id" class="lwb-checkbox"><input type="checkbox" :checked="group.blocks.every(id => exploreConfig.frozen.includes(id))" :indeterminate="group.blocks.some(id => exploreConfig.frozen.includes(id)) && !group.blocks.every(id => exploreConfig.frozen.includes(id))" :disabled="locked" @change="toggleFrozen(group.blocks, ($event.target as HTMLInputElement).checked)" />{{ group.label }}<span v-if="group.range" class="lwb-freeze-range">{{ group.range }}</span></label><p class="lwb-help">{{ animaGuideSupported ? '按社区分组冻结，保留当前倍率。' : '冻结组保留当前倍率，不参与随机变化。' }}</p>
                </section>
                <section class="lwb-inspector-section"><h4>生成设置</h4><p class="lwb-help">{{ preview.width }} × {{ preview.height }} · {{ preview.steps }} 步<br />CFG {{ preview.cfg }} · 种子 {{ preview.seed }}</p><div class="lwb-actions lwb-quick-values"><button :disabled="locked" @click="goStep(1)">修改设置</button><button :disabled="locked" @click="workspaceMode = 'repair'">逐层调整</button><button :disabled="locked" @click="goStep(3)">对照与导出</button></div></section>
              </template>
            </aside>
          </div>
        </div>
        <div v-else ref="mainScroll" class="lwb-stage lwb-scrolling">
          <div class="lwb-stage-heading"><div><h3>原版 / 调整版对照</h3><p>同一底模、提示词、种子与采样设置。</p></div><button :disabled="locked || !ready || !previewConfigured" @click="start('preview')"><UiIcon name="refresh" />{{ images.length ? '重新生成对照' : '生成对照' }}</button></div>
          <div v-if="outdated" class="lwb-notice">倍率或采样设置已改变，下方图片为上次结果。</div>
          <div v-if="images.length" class="lwb-ab-grid"><figure v-for="image in images" :key="image.label"><button class="lwb-picture-button" :aria-label="`放大${image.label}`" @click="showLarge(image.url, image.label)"><img :src="image.url" :alt="image.label" /></button><figcaption>{{ image.label }}</figcaption></figure></div>
          <div v-else class="lwb-empty lwb-ab-empty"><UiIcon name="image" /><strong>{{ previewConfigured ? '准备生成原版与调整版' : '尚未配置图片预览' }}</strong><p>{{ previewConfigured ? '点击生成对照，检查当前调整效果。' : '可以先配置预览，也可直接另存调整版。' }}</p><button v-if="!previewConfigured" :disabled="locked" @click="goStep(1)">配置预览</button></div>
          <div class="lwb-export-summary"><div><strong>已调整 {{ modified }} 层</strong><p>{{ preview.width }} × {{ preview.height }} · {{ preview.steps }} 步 · CFG {{ preview.cfg }} · 种子 {{ preview.seed }}</p></div><button :disabled="locked" @click="goStep(2)"><UiIcon name="back" />继续调整</button></div>
          <div v-if="adjustedOutput" class="lwb-result"><strong>{{ exportOutdated ? '上次导出' : '调整版已保存' }}</strong><p v-if="exportOutdated">倍率已改变，需重新另存。</p><code>{{ adjustedOutput }}</code></div>
        </div>
      </main>
      <footer class="lwb-footer">
        <div class="lwb-footer-status" aria-live="polite"><span>{{ nextHint }}</span><div v-if="locked" class="lwb-progress" role="progressbar" aria-label="工作台任务进度" :aria-valuenow="progressPercent ?? undefined" aria-valuemin="0" aria-valuemax="100" :class="{ indeterminate: progress == null }"><i :style="{ width: progressPercent == null ? '35%' : `${progressPercent}%` }"></i></div></div>
        <div class="lwb-actions">
          <button v-if="busy" @click="cancel">停止任务</button>
          <template v-else><button v-if="tab === 'adjust' && flowStep > 0" :disabled="locked" @click="goStep(flowStep - 1)">上一步</button><button v-if="tab === 'adjust' && flowStep === 1" class="lwb-subtle" :disabled="locked || !current" @click="goStep(2)">跳过分析</button></template>
          <button class="lwb-primary" :disabled="primaryDisabled" @click="primaryAction">{{ locked ? '处理中…' : primaryLabel }}</button>
        </div>
      </footer>
      <div v-if="panel" class="lwb-panel-backdrop" @click.self="panel = null">
        <aside class="lwb-sidepanel" role="dialog" aria-modal="true" :aria-label="panelTitle">
          <header><h3>{{ panelTitle }}</h3><button class="lwb-icon-button" aria-label="关闭侧栏" @click="panel = null"><UiIcon name="close" /></button></header>
          <div class="lwb-panel-content lwb-scrolling">
            <p v-if="starting || message" class="lwb-panel-status" aria-live="polite">{{ starting ? '正在处理…' : message }}</p>
            <div v-if="error" class="lwb-notice" role="alert">{{ error }}</div>
            <template v-if="panel === 'presets'">
              <section><h4>应用已保存的预设</h4><label>选择预设<select v-model="presetId" :disabled="locked"><option value="">选择一个预设…</option><option v-for="preset in presets" :key="preset.id" :value="preset.id">{{ preset.name }}{{ matchesPreset(preset) ? '' : '（结构不匹配）' }}</option></select></label>
                <p v-if="!presets.length" class="lwb-help">暂无预设，可保存当前倍率。</p><p v-else-if="selectedPreset" class="lwb-help">{{ selectedPreset.source_name }}<br />{{ selectedPreset.created_at }}</p>
                <div v-if="selectedPreset && !matchesPreset(selectedPreset)" class="lwb-notice">此预设与当前模型或层级结构不匹配。</div><div v-else-if="selectedPreset && selectedPreset.fingerprint !== info?.fingerprint" class="lwb-notice">来自其他 LoRA，应用后请生成对照确认效果。</div>
                <div class="lwb-actions"><button :disabled="locked || !selectedPreset || !matchesPreset(selectedPreset)" @click="applyPreset">应用预设</button><button :disabled="locked || !selectedPreset" @click="deletePreset">删除</button></div>
              </section><div class="lwb-rule"></div><section><h4>保存当前倍率</h4><label>预设名称<input v-model="presetName" :disabled="locked" placeholder="例如：画风减弱方案" maxlength="80" /></label><button :disabled="locked || !presetName.trim() || !current" @click="savePreset">保存预设</button></section>
            </template>
            <template v-else-if="panel === 'layer-guide'">
              <p class="lwb-help lwb-guide-intro">标准 Anima 28 层 · L00–L27。以下来自 {{ ANIMA_GUIDE_SOURCE.author }} 的合并观察，各层影响可能交叠。</p>
              <div v-if="!animaGuideSupported" class="lwb-notice">尚未确认当前文件为标准 28 层映射，仅供查阅。重新读取 LoRA 后再检查。</div>
              <p v-else class="lwb-help">按经验分组生成实测对照；仅调整当前 LoRA 已有的层。</p>
              <section v-for="group in ANIMA_GUIDE_GROUPS" :key="group.id" class="lwb-guide-group">
                <div class="lwb-guide-heading"><div><h4>{{ group.label }}</h4><span>{{ group.range }}</span></div><button class="lwb-small" :disabled="locked || !guideBlockIds(group.id).length" @click="selectGuideGroup(group.id)">调整此组</button></div>
                <div v-for="layer in group.layers" :key="layer.id" class="lwb-guide-layer" :class="{ 'is-absent': animaGuideSupported && !blockIds.includes(layer.id) }"><span>{{ layer.label }}</span><span>{{ layer.hint }}<small v-if="animaGuideSupported && !blockIds.includes(layer.id)"> · 文件未包含</small></span></div>
              </section>
              <details class="lwb-detail"><summary>使用方法与来源</summary>
                <p class="lwb-help">先生成分组对照，再根据图片调整倍率；确认后保存预设或另存 LoRA。</p>
                <p class="lwb-help">Modulation Guidance 的另一组实验发现，L00–L07 也会影响整体布局和色调，L27 存在补偿作用。层的影响取决于修改方式。</p>
                <p class="lwb-help">这些经验用于已有 LoRA 的权重调整，尚未验证为训练层选择配方。</p>
                <div class="lwb-guide-sources"><a :href="ANIMA_GUIDE_SOURCE.url" target="_blank" rel="noopener noreferrer">{{ ANIMA_GUIDE_SOURCE.author }} · {{ ANIMA_GUIDE_SOURCE.title }} ↗</a><a :href="ANIMA_MOD_GUIDANCE_SOURCE.url" target="_blank" rel="noopener noreferrer">{{ ANIMA_MOD_GUIDANCE_SOURCE.title }} ↗</a><span>来源核对：{{ ANIMA_GUIDE_SOURCE.checkedAt }}</span></div>
              </details>
            </template>
            <template v-else-if="panel === 'suggestions' && profiler?.profile">
              <div v-if="profileOutdated" class="lwb-notice">{{ profileLayoutOutdated ? '分组已更新，请重新生成分组对照后应用建议。' : '采样条件已改变，请重新生成对照后应用建议。' }}</div><p class="lwb-help">建议针对本次采样场景，应用后需生成对照确认。</p>
              <div v-for="group in suggestions" :key="group.id" class="lwb-suggestion"><strong>{{ group.label }}</strong><p>{{ group.text }}</p><span>建议倍率 {{ [...new Set(group.blocks.map(id => profiler?.profile?.suggested_blocks[id]))].join(' / ') }}</span></div>
              <div v-if="!suggestions.length && !profileOutdated" class="lwb-empty"><strong>本次建议保留原始倍率</strong><p>仍可根据分组对照手动调整。</p></div>
              <div v-for="warning in profiler.profile.warnings" :key="warning" class="lwb-notice">{{ warning }}</div><p class="lwb-help">应用将替换当前所有层级倍率，可撤销。</p>
              <button class="lwb-primary" :disabled="locked || profileOutdated || !suggestions.length" @click="applySuggestions">应用建议</button>
            </template>
          </div>
        </aside>
      </div>
      <div v-if="fullProfileImage" class="lwb-image-backdrop" @click.self="fullProfileImage = null"><section class="lwb-large-image" role="dialog" aria-modal="true" :aria-label="fullProfileImage.label"><header><strong>{{ fullProfileImage.label }}</strong><button class="lwb-icon-button" aria-label="关闭大图" @click="fullProfileImage = null"><UiIcon name="close" /></button></header><img :src="fullProfileImage.url" :alt="fullProfileImage.label" /></section></div>
    </section>
  </div>
</template>
<style scoped>
.lwb-backdrop { position: fixed; inset: 0; z-index: 120; display: grid; place-items: center; padding: 20px; background: rgb(0 0 0 / .56); }
.lwb-dialog { position: relative; display: flex; flex-direction: column; width: min(1320px, 100%); height: min(900px, calc(100dvh - 40px)); min-height: 0; overflow: hidden; border: 1px solid var(--border); border-radius: 16px; background: var(--bg); color: var(--text); box-shadow: 0 24px 80px rgb(0 0 0 / .3); font-size: 13px; outline: none; }
.lwb-dialog button { display: inline-flex; align-items: center; justify-content: center; gap: 7px; min-height: 36px; padding: 7px 12px; border: 1px solid var(--border); border-radius: 7px; background: var(--card); color: var(--text); cursor: pointer; font: inherit; line-height: 1.4; white-space: nowrap; }
.lwb-dialog button:hover:not(:disabled) { background: var(--card-hover); border-color: var(--accent); }
.lwb-dialog button:disabled { cursor: default; opacity: .42; }
.lwb-dialog input, .lwb-dialog textarea, .lwb-dialog select { width: 100%; min-width: 0; border: 1px solid var(--border); border-radius: 7px; background: var(--bg); color: var(--text); padding: 10px 11px; font: inherit; line-height: 1.5; }
.lwb-dialog input::placeholder, .lwb-dialog textarea::placeholder { color: var(--sub); opacity: .8; }
.lwb-dialog input:disabled, .lwb-dialog textarea:disabled, .lwb-dialog select:disabled { opacity: .5; }
.lwb-dialog textarea { resize: vertical; min-height: 80px; }
.lwb-dialog input[type="range"] { padding: 0; height: 26px; border: 0; accent-color: var(--accent); cursor: pointer; }
.lwb-dialog input[type="number"] { font-variant-numeric: tabular-nums; }
.lwb-dialog :is(button, input, textarea, select, summary, a):focus-visible { outline: 2px solid var(--accent-hover); outline-offset: 3px; }
.lwb-dialog :is(h2, h3, h4, p, figure) { margin: 0; }
.lwb-dialog h3 { font-size: 20px; line-height: 1.5; font-weight: 600; }
.lwb-dialog h4 { font-size: 15px; line-height: 1.5; font-weight: 600; }
.lwb-dialog fieldset { min-width: 0; padding: 0; margin: 0; border: 0; }
.lwb-dialog label { display: flex; flex-direction: column; gap: 7px; line-height: 1.5; }
.lwb-dialog .ui-icon { width: 17px; height: 17px; flex: 0 0 auto; }
.lwb-dialog .lwb-primary { border-color: var(--accent); background: var(--accent); color: var(--text); font-weight: 600; }
.lwb-dialog .lwb-primary:hover:not(:disabled) { background: var(--accent-hover); }
.lwb-dialog .lwb-subtle { background: transparent; }
.lwb-dialog .lwb-icon-button { width: 34px; min-height: 34px; flex: 0 0 auto; padding: 7px; background: transparent; }
.lwb-header { display: flex; flex: 0 0 auto; justify-content: space-between; align-items: center; gap: 20px; padding: 21px 26px 17px; }
.lwb-heading h2 { font-size: 24px; line-height: 1.4; font-weight: 650; }
.lwb-heading p { margin-top: 3px; color: var(--sub); font-size: 12px; }
.lwb-actions { display: flex; align-items: center; gap: 8px; }
.lwb-file-strip { display: flex; flex: 0 0 auto; align-items: center; gap: 13px; margin: 0 26px 14px; color: var(--sub); font-size: 12px; }
.lwb-file-strip strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 50%; color: var(--text); font-weight: 500; }
.lwb-file-strip button { padding: 2px 4px; min-height: 24px; border: 0; background: transparent; color: var(--sub); margin-left: auto; }
.lwb-steps { display: grid; flex: 0 0 auto; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; padding: 0 26px 16px; border-bottom: 1px solid var(--border); }
.lwb-steps button { justify-content: flex-start; gap: 10px; padding: 10px 12px; min-height: 46px; border-color: transparent; background: transparent; color: var(--sub); }
.lwb-steps button.active { background: var(--card); border-color: var(--border); color: var(--text); }
.lwb-step-number { display: grid; place-items: center; width: 25px; height: 25px; flex: 0 0 auto; border: 1px solid var(--border); border-radius: 50%; font-size: 12px; }
.lwb-steps .active .lwb-step-number { background: var(--accent); border-color: var(--accent); color: var(--text); }
.lwb-steps .passed .lwb-step-number { color: var(--text); border-color: var(--accent); }
.lwb-body { display: flex; flex-direction: column; flex: 1 1 auto; min-height: 0; overflow: hidden; }
.lwb-stage { flex: 1 1 auto; min-height: 0; padding: 24px 26px; }
.lwb-scrolling { overflow: auto; overscroll-behavior: contain; scrollbar-width: thin; scrollbar-color: var(--accent) transparent; }
.lwb-scrolling::-webkit-scrollbar, .lwb-group-list::-webkit-scrollbar { width: 7px; height: 7px; }
.lwb-scrolling::-webkit-scrollbar-thumb, .lwb-group-list::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 8px; }
.lwb-stage-heading { display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-bottom: 20px; }
.lwb-stage-heading p { margin-top: 4px; color: var(--sub); line-height: 1.5; }
.lwb-section { padding: 22px; background: var(--card); border: 1px solid var(--border); border-radius: 10px; }
.lwb-section > h4 { margin-bottom: 16px; }
.lwb-section > label + label { margin-top: 16px; }
.lwb-narrow { max-width: 900px; }
.lwb-input-action { display: flex; gap: 8px; }
.lwb-input-action > input { flex: 1; }
.lwb-input-action > button { flex: 0 0 auto; min-width: 60px; }
.lwb-form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 17px; }
.lwb-three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.lwb-help { color: var(--sub); font-size: 12px; line-height: 1.65; margin-top: 9px !important; }
.lwb-section-heading { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-bottom: 16px; }
.lwb-section-heading > span { color: var(--sub); font-size: 12px; }
.lwb-dialog .lwb-small { min-height: 30px; padding: 5px 9px; font-size: 12px; }
.lwb-rule { height: 1px; background: var(--border); margin: 22px 0; }
.lwb-detail { margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--border); }
.lwb-detail summary { cursor: pointer; color: var(--text); line-height: 1.6; }
.lwb-detail summary span { float: right; color: var(--sub); font-size: 12px; }
.lwb-detail[open] summary { margin-bottom: 16px; }
.lwb-detail > textarea { margin-top: 8px; }
.lwb-config-grid { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: 18px; align-items: start; }
.lwb-prompt { min-height: 165px !important; }
.lwb-file-info { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 18px; margin-top: 24px; padding-top: 21px; border-top: 1px solid var(--border); }
.lwb-file-info div { display: flex; flex-direction: column; gap: 8px; }
.lwb-file-info span { color: var(--sub); font-size: 12px; }
.lwb-file-info strong { font-size: 15px; font-weight: 500; }
.lwb-route-note { margin-top: 18px !important; color: var(--sub); line-height: 1.6; }
.lwb-notice { margin: 12px 0; padding: 11px 13px; border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 5px; background: var(--card); color: var(--sub); font-size: 12px; line-height: 1.6; overflow-wrap: anywhere; }
.lwb-notice p + p { margin-top: 6px; }
.lwb-notice button { margin-left: 9px; min-height: 26px; padding: 3px 7px; font-size: 12px; }
.lwb-environment { display: flex; flex: 0 0 auto; align-items: center; justify-content: space-between; gap: 12px; margin: 12px 26px 0; }
.lwb-file-warning { flex: 0 0 auto; margin: 12px 26px 0; }
.lwb-error { display: flex; flex: 0 0 auto; align-items: center; justify-content: space-between; gap: 14px; margin: 12px 26px 0; padding: 9px 12px; border: 1px solid #bc7171; border-radius: 7px; background: var(--card); line-height: 1.5; overflow-wrap: anywhere; }
.lwb-analysis-summary { display: flex; flex-direction: column; gap: 6px; margin-top: 22px; padding: 16px; background: var(--bg); border-radius: 7px; }
.lwb-analysis-summary strong { font-size: 14px; font-weight: 500; }
.lwb-analysis-summary span { color: var(--sub); font-size: 12px; line-height: 1.5; }
.lwb-reference { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-top: 7px; }
.lwb-reference span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--sub); }
.lwb-adjust-stage { display: flex; flex-direction: column; }
.lwb-adjust-stage > .lwb-stage-heading { flex: 0 0 auto; }
.lwb-toolbar { flex-wrap: wrap; justify-content: flex-end; }
.lwb-toolbar button { min-height: 32px; padding: 5px 10px; font-size: 12px; }
.lwb-count { display: inline-grid; place-items: center; min-width: 18px; height: 18px; padding: 0 3px; border-radius: 4px; background: var(--bg); font-variant-numeric: tabular-nums; }
.lwb-edit-layout { display: grid; grid-template-columns: 236px minmax(0, 1fr); flex: 1 1 auto; min-height: 0; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: var(--card); }
.lwb-group-rail { display: flex; flex-direction: column; min-height: 0; border-right: 1px solid var(--border); background: var(--bg); }
.lwb-rail-title { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 15px 16px 10px; color: var(--sub); font-size: 12px; flex: 0 0 auto; }
.lwb-group-list { flex: 1 1 auto; min-height: 0; overflow: auto; overscroll-behavior: contain; padding: 5px 8px 12px; scrollbar-width: thin; scrollbar-color: var(--accent) transparent; }
.lwb-dialog .lwb-group-button { display: flex; align-items: stretch; flex-direction: column; gap: 8px; width: 100%; margin-bottom: 5px; padding: 12px 11px; text-align: left; background: transparent; border-color: transparent; white-space: normal; }
.lwb-dialog .lwb-group-button.selected { border-color: var(--accent); background: var(--card); }
.lwb-group-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.lwb-group-top strong { font-size: 13px; font-weight: 600; }
.lwb-group-value { padding: 2px 5px; color: var(--sub); font-size: 11px; line-height: 1.2; font-variant-numeric: tabular-nums; }
.lwb-group-value.changed { color: var(--text); border: 1px solid var(--accent); border-radius: 4px; }
.lwb-group-note { color: var(--sub); font-size: 11px; line-height: 1.5; }
.lwb-rail-footer { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex: 0 0 auto; padding: 11px 12px; border-top: 1px solid var(--border); color: var(--sub); font-size: 12px; }
.lwb-rail-footer button { padding: 3px 7px; min-height: 26px; font-size: 11px; background: transparent; }
.lwb-group-detail { display: flex; flex-direction: column; min-height: 0; min-width: 0; overflow: hidden; padding: 21px 22px 22px; }
.lwb-group-detail > .lwb-section-heading { flex: 0 0 auto; }
.lwb-analysis-canvas { flex: 1 1 auto; min-height: 0; padding-right: 6px; padding-bottom: 10px; }
.lwb-chip { flex: 0 0 auto; padding: 4px 8px; border: 1px solid var(--border); border-radius: 5px; }
.lwb-analysis-pictures { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.lwb-dialog .lwb-picture-button { display: flex; padding: 0; width: 100%; background: var(--bg); overflow: hidden; border-radius: 7px; }
.lwb-picture-button img { display: block; width: 100%; height: 100%; object-fit: contain; }
.lwb-analysis-pictures .lwb-picture-button { aspect-ratio: 1; max-height: 230px; }
.lwb-dialog figcaption { padding-top: 9px; text-align: center; color: var(--sub); font-size: 12px; line-height: 1.5; }
.lwb-baseline-pictures { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.lwb-baseline-pictures .lwb-picture-button { aspect-ratio: 1; max-height: 260px; }
.lwb-metrics { display: flex; flex-wrap: wrap; gap: 16px 28px; margin-top: 20px; }
.lwb-metrics div { display: flex; flex-direction: column; gap: 7px; }
.lwb-metrics span { color: var(--sub); font-size: 11px; }
.lwb-metrics strong { font-size: 16px; font-weight: 500; font-variant-numeric: tabular-nums; }
.lwb-strength-section { flex: 0 0 auto; margin-top: 14px; padding-top: 15px; border-top: 1px solid var(--border); }
.lwb-strength-section > .lwb-section-heading { margin-bottom: 10px; }
.lwb-strength-control { display: flex; align-items: center; gap: 12px; }
.lwb-strength-control > span { color: var(--sub); font-size: 11px; font-variant-numeric: tabular-nums; }
.lwb-strength-control input[type="range"] { flex: 1; }
.lwb-strength-control input[type="number"] { width: 88px; flex: 0 0 auto; text-align: center; }
.lwb-quick-values { gap: 7px; margin-top: 14px; flex-wrap: wrap; }
.lwb-quick-values button { min-height: 30px; padding: 5px 9px; font-size: 12px; }
.lwb-block-row { display: grid; grid-template-columns: minmax(110px, 1fr) minmax(80px, 1.3fr) 80px; align-items: center; gap: 16px; padding: 12px 0; border-top: 1px solid var(--border); }
.lwb-block-row strong { display: block; font-size: 12px; font-weight: 500; }
.lwb-block-row span { display: block; margin-top: 5px; color: var(--sub); font-size: 11px; }
.lwb-class-picture { max-width: 360px; }
.lwb-class-picture img { max-height: 340px; }
.lwb-empty { display: flex; align-items: center; justify-content: center; flex-direction: column; gap: 12px; padding: 36px 20px; border: 1px dashed var(--border); border-radius: 8px; text-align: center; }
.lwb-empty > .ui-icon { width: 28px; height: 28px; color: var(--sub); }
.lwb-empty strong { font-size: 14px; font-weight: 500; }
.lwb-empty p { color: var(--sub); font-size: 12px; line-height: 1.6; }
.lwb-analysis-empty { min-height: 220px; }
.lwb-ab-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
.lwb-ab-grid .lwb-picture-button { height: clamp(220px, 40vh, 420px); }
.lwb-ab-empty { min-height: 285px; }
.lwb-export-summary { display: flex; align-items: center; justify-content: space-between; gap: 18px; padding: 20px 0; margin-top: 15px; border-top: 1px solid var(--border); }
.lwb-export-summary strong { font-size: 14px; font-weight: 500; }
.lwb-export-summary p { margin-top: 7px; color: var(--sub); font-size: 12px; }
.lwb-result { display: flex; flex-direction: column; gap: 10px; padding: 16px; background: var(--card); border: 1px solid var(--accent); border-radius: 7px; margin-top: 18px; }
.lwb-result strong { font-weight: 600; }
.lwb-result p { color: var(--sub); font-size: 12px; line-height: 1.6; }
.lwb-result code { color: var(--sub); font-size: 12px; overflow-wrap: anywhere; user-select: text; }
.lwb-footer { display: flex; flex: 0 0 auto; align-items: center; justify-content: space-between; gap: 20px; padding: 16px 26px; min-height: 76px; border-top: 1px solid var(--border); }
.lwb-footer-status { flex: 1 1 auto; min-width: 0; color: var(--sub); font-size: 12px; line-height: 1.5; }
.lwb-footer > .lwb-actions { flex: 0 0 auto; }
.lwb-footer .lwb-primary { min-width: 150px; min-height: 40px; }
.lwb-progress { max-width: 420px; height: 4px; border-radius: 3px; overflow: hidden; background: var(--border); margin-top: 8px; }
.lwb-progress i { display: block; height: 100%; background: var(--accent-hover); border-radius: inherit; }
.lwb-progress.indeterminate i { animation: lwb-progress 1.4s ease-in-out infinite alternate; }
.lwb-panel-backdrop { position: absolute; z-index: 2; inset: 0; background: rgb(0 0 0 / .4); }
.lwb-sidepanel { position: absolute; inset: 0 0 0 auto; display: flex; flex-direction: column; width: min(410px, 95%); background: var(--bg); border-left: 1px solid var(--border); box-shadow: -12px 0 40px rgb(0 0 0 / .25); }
.lwb-sidepanel > header, .lwb-large-image > header { display: flex; flex: 0 0 auto; align-items: center; justify-content: space-between; gap: 18px; padding: 20px 22px; border-bottom: 1px solid var(--border); }
.lwb-sidepanel h3 { font-size: 18px; }
.lwb-panel-content { flex: 1; min-height: 0; padding: 24px 22px; }
.lwb-panel-status { margin-bottom: 18px !important; color: var(--sub); font-size: 12px; line-height: 1.6; }
.lwb-panel-content section > h4 { margin-bottom: 18px; }
.lwb-panel-content section > button, .lwb-panel-content section > .lwb-actions { margin-top: 16px; }
.lwb-suggestion { padding: 16px 0; border-bottom: 1px solid var(--border); }
.lwb-suggestion strong { font-weight: 600; }
.lwb-suggestion p { margin-top: 8px; color: var(--sub); line-height: 1.6; }
.lwb-suggestion > span { display: block; margin-top: 8px; font-size: 12px; }
.lwb-panel-content > .lwb-primary { width: 100%; margin-top: 16px; }
.lwb-image-backdrop { position: absolute; inset: 0; z-index: 3; display: grid; place-items: center; padding: 18px; background: rgb(0 0 0 / .72); }
.lwb-large-image { display: flex; flex-direction: column; width: 100%; max-height: 100%; overflow: hidden; border: 1px solid var(--border); border-radius: 10px; background: var(--bg); }
.lwb-large-image > header { padding: 12px 16px; }
.lwb-large-image > header strong { font-weight: 500; }
.lwb-large-image > img { display: block; width: 100%; min-height: 0; object-fit: contain; flex: 1 1 auto; }
.lwb-visually-hidden { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; border: 0; }
@keyframes lwb-progress { from { transform: translateX(0); } to { transform: translateX(180%); } }
@media (max-width: 1050px) {
  .lwb-edit-layout { grid-template-columns: 200px minmax(0, 1fr); }
  .lwb-stage, .lwb-header { padding-left: 20px; padding-right: 20px; }
  .lwb-steps { padding-left: 20px; padding-right: 20px; }
  .lwb-group-detail { padding: 18px; }
  .lwb-section { padding: 18px; }
  .lwb-config-grid { gap: 14px; }
  .lwb-section-heading { gap: 8px; }
  .lwb-section-heading .lwb-small { font-size: 11px; padding: 5px 7px; }
  .lwb-strength-control { gap: 9px; }
}
@media (max-width: 800px) {
  .lwb-backdrop { padding: 10px; }
  .lwb-dialog { height: calc(100dvh - 20px); border-radius: 10px; }
  .lwb-header { padding: 15px 16px; }
  .lwb-heading h2 { font-size: 21px; }
  .lwb-header .lwb-subtle { font-size: 11px; padding: 5px 7px; }
  .lwb-file-strip { margin: 0 16px 12px; gap: 8px; }
  .lwb-file-strip strong { max-width: 42%; }
  .lwb-file-strip > span:last-of-type { display: none; }
  .lwb-steps { padding: 0 12px 12px; gap: 2px; }
  .lwb-steps button { gap: 6px; font-size: 11px; padding: 7px 5px; }
  .lwb-step-number { width: 22px; height: 22px; }
  .lwb-stage { padding: 18px 16px; }
  .lwb-config-grid { grid-template-columns: 1fr; }
  .lwb-stage-heading { align-items: flex-start; margin-bottom: 16px; }
  .lwb-adjust-stage > .lwb-stage-heading { flex-wrap: wrap; gap: 12px; }
  .lwb-toolbar { justify-content: flex-start; gap: 5px; }
  .lwb-edit-layout { grid-template-columns: 154px minmax(0, 1fr); }
  .lwb-group-detail { padding: 14px; }
  .lwb-group-button { padding: 10px 7px !important; }
  .lwb-group-top { flex-wrap: wrap; gap: 5px; }
  .lwb-group-note { font-size: 10px; }
  .lwb-analysis-pictures { gap: 7px; }
  .lwb-analysis-pictures .lwb-picture-button { max-height: 160px; }
  .lwb-metrics { gap: 12px; }
  .lwb-strength-control > span { display: none; }
  .lwb-block-row { grid-template-columns: minmax(70px, 1fr) 65px; gap: 8px; }
  .lwb-block-row input[type="range"] { grid-column: 1; grid-row: 2; }
  .lwb-block-row input[type="number"] { grid-column: 2; grid-row: 1 / 3; }
  .lwb-rail-footer { flex-direction: column; gap: 6px; }
  .lwb-footer { padding: 12px 16px; gap: 12px; }
  .lwb-footer .lwb-actions { gap: 6px; }
  .lwb-footer .lwb-primary { min-width: 115px; }
  .lwb-footer-status { font-size: 11px; }
  .lwb-footer button { padding: 7px 9px; font-size: 12px; }
  .lwb-file-info { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-height: 760px) {
  .lwb-header { padding-top: 13px; padding-bottom: 11px; }
  .lwb-heading h2 { font-size: 21px; }
  .lwb-file-strip { margin-bottom: 9px; }
  .lwb-steps { padding-bottom: 11px; }
  .lwb-steps button { min-height: 39px; padding-top: 7px; padding-bottom: 7px; }
  .lwb-stage { padding-top: 17px; padding-bottom: 17px; }
  .lwb-stage-heading { margin-bottom: 14px; }
  .lwb-stage-heading h3 { font-size: 18px; }
  .lwb-footer { min-height: 65px; padding-top: 12px; padding-bottom: 12px; }
  .lwb-analysis-pictures .lwb-picture-button { max-height: 180px; }
}
/* Canvas-first studio: controls scroll independently; images never use a clipped strip. */
.lwb-backdrop { padding: 10px; }
.lwb-dialog { width: min(1900px, 100%); height: calc(100dvh - 20px); border-radius: 11px; }
.lwb-header { gap: 14px; min-height: 55px; padding: 9px 16px; }
.lwb-heading { display: flex; align-items: center; gap: 20px; min-width: 0; }
.lwb-heading h2 { font-size: 18px; white-space: nowrap; }
.lwb-header-file { display: flex; align-items: center; gap: 12px; min-width: 0; font-size: 12px; }
.lwb-header-file strong { max-width: 360px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; font-weight: 500; }
.lwb-header-file span, .lwb-header-subtitle { color: var(--sub); white-space: nowrap; font-size: 12px; }
.lwb-header-file button { border: 0; background: transparent; min-height: 25px; padding: 3px 5px; color: var(--sub); }
.lwb-header > .lwb-actions { flex: 0 0 auto; }
.lwb-header .lwb-subtle { min-height: 31px; font-size: 12px; }
.lwb-steps { display: flex; gap: 15px; padding: 0 16px; min-height: 39px; }
.lwb-steps button { position: relative; min-height: 39px; border: 0; border-radius: 0; padding: 5px 8px; gap: 7px; font-size: 12px; }
.lwb-steps button.active { background: transparent; }
.lwb-steps button.active::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px; background: var(--accent-hover); }
.lwb-step-number { width: 20px; height: 20px; font-size: 10px; }
.lwb-stage { padding: 18px 20px; }
.lwb-adjust-stage { padding: 10px 12px 12px; }
.lwb-workspace-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex: 0 0 auto; margin-bottom: 9px; }
.lwb-mode-switch { display: flex; align-items: center; gap: 3px; padding: 3px; background: var(--card); border: 1px solid var(--border); border-radius: 7px; }
.lwb-mode-switch button { min-height: 29px; padding: 4px 12px; background: transparent; border-color: transparent; font-size: 12px; }
.lwb-mode-switch button.active { background: var(--accent); border-color: var(--accent); }
.lwb-toolbar { flex-wrap: nowrap; gap: 6px; }
.lwb-toolbar button { min-height: 29px; padding: 4px 9px; font-size: 12px; }
.lwb-studio-layout { display: grid; grid-template-columns: 180px minmax(0, 1fr) 282px; flex: 1 1 0; min-height: 0; overflow: hidden; border: 1px solid var(--border); border-radius: 8px; }
.lwb-studio-layout.inspector-hidden { grid-template-columns: 180px minmax(0, 1fr); }
.lwb-studio-layout.is-explore { grid-template-columns: minmax(0, 1fr) 282px; }
.lwb-studio-layout.is-explore.inspector-hidden { grid-template-columns: minmax(0, 1fr); }
.lwb-group-rail { min-width: 0; }
.lwb-rail-title { padding: 13px 11px 8px; }
.lwb-group-list { padding: 4px 5px 8px; }
.lwb-dialog .lwb-group-button { padding: 10px 7px; gap: 6px; }
.lwb-group-top strong { font-size: 12px; }
.lwb-group-note { font-size: 11px; }
.lwb-rail-footer { padding: 8px; gap: 4px; font-size: 11px; flex-direction: column; align-items: stretch; }
.lwb-rail-footer button { min-height: 25px; }
.lwb-viewer { display: flex; flex-direction: column; min-width: 0; min-height: 0; padding: 12px; overflow: hidden; background: color-mix(in srgb, var(--bg) 88%, black); }
.lwb-canvas-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 8px; flex: 0 0 auto; min-height: 28px; margin-bottom: 10px; }
.lwb-canvas-toolbar strong { font-size: 13px; font-weight: 500; }
.lwb-canvas-toolbar > span { color: var(--sub); font-size: 11px; }
.lwb-view-controls { display: flex; gap: 3px; }
.lwb-view-controls button { min-height: 26px; font-size: 11px; padding: 3px 7px; border-color: transparent; background: transparent; color: var(--sub); }
.lwb-view-controls button.active { color: var(--text); background: var(--card); border-color: var(--border); }
.lwb-canvas-pictures { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; flex: 1 1 0; min-height: 0; }
.lwb-canvas-pictures.single { grid-template-columns: minmax(0, 1fr); }
.lwb-canvas-pictures > figure, .lwb-explorer-canvas figure { display: flex; flex-direction: column; min-width: 0; min-height: 0; overflow: hidden; }
.lwb-canvas-pictures .lwb-picture-button, .lwb-explorer-canvas .lwb-picture-button { flex: 1 1 0; min-height: 0; height: 0; max-height: none; aspect-ratio: auto; border: 1px solid var(--border); border-radius: 6px; }
.lwb-canvas-pictures .lwb-picture-button img, .lwb-explorer-canvas .lwb-picture-button img { min-width: 0; min-height: 0; width: 100%; height: 100%; object-fit: contain; }
.lwb-canvas-pictures figcaption { flex: 0 0 auto; padding: 9px 0 3px; }
.lwb-canvas-pictures .lwb-profile-caption { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.lwb-profile-caption button { min-height: 27px; padding: 4px 9px; font-size: 11px; }
.lwb-await-variants { min-height: 0; }
.lwb-canvas-footnote { flex: 0 0 auto; margin-top: 9px; color: var(--sub); font-size: 11px; line-height: 1.5; }
.lwb-canvas-status { flex: 0 0 auto; display: flex; align-items: center; gap: 8px; padding: 6px 9px; margin-bottom: 9px; background: var(--card); color: var(--sub); font-size: 11px; border-radius: 4px; }
.lwb-canvas-status button { font-size: 11px; min-height: 24px; padding: 2px 7px; }
.lwb-studio-empty { flex: 1 1 0; min-height: 0; padding: 20px; }
.lwb-inspector { border-left: 1px solid var(--border); background: var(--card); padding: 16px 15px; min-height: 0; min-width: 0; }
.lwb-inspector-title { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.lwb-inspector-title > span { color: var(--sub); font-size: 11px; }
.lwb-inspector-section { margin-top: 17px; padding-top: 17px; border-top: 1px solid var(--border); }
.lwb-inspector-section > label + label { margin-top: 16px; }
.lwb-inspector .lwb-section-heading { margin-bottom: 8px; }
.lwb-inspector h4 { font-size: 13px; }
.lwb-range-labels { display: flex; align-items: center; justify-content: space-between; gap: 10px; color: var(--sub); font-size: 11px; }
.lwb-range-labels input { width: 90px; text-align: center; padding: 6px 8px; }
.lwb-quick-values { flex-wrap: wrap; gap: 5px; margin-top: 10px; }
.lwb-quick-values button { min-height: 26px; font-size: 11px; padding: 4px 8px; }
.lwb-dialog .lwb-checkbox { display: flex; flex-direction: row; align-items: center; gap: 8px; font-size: 12px; margin-top: 13px; cursor: pointer; }
.lwb-checkbox input { width: 15px; height: 15px; accent-color: var(--accent); flex: 0 0 auto; }
.lwb-stat-list { margin-top: 10px; }
.lwb-stat-list > div { display: flex; justify-content: space-between; gap: 8px; padding: 5px 0; font-size: 12px; }
.lwb-stat-list span { color: var(--sub); }
.lwb-stat-list strong { font-weight: 500; font-variant-numeric: tabular-nums; }
.lwb-inspector .lwb-detail { margin-top: 17px; padding-top: 15px; font-size: 12px; }
.lwb-layer-control { padding: 14px 0; border-bottom: 1px solid var(--border); }
.lwb-layer-control > div { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-size: 12px; }
.lwb-layer-control strong { font-weight: 500; }
.lwb-layer-control input[type="number"] { width: 80px; padding: 5px 7px; }
.lwb-layer-control input[type="range"] { margin-top: 7px; }
.lwb-layer-control p { margin-top: 3px !important; font-size: 11px; }
.lwb-inspector .lwb-baseline-pictures .lwb-picture-button { max-height: 145px; }
.lwb-inspector-section > label { position: relative; font-size: 12px; }
.lwb-control-value { position: absolute; right: 0; color: var(--sub); font-size: 11px; }
.lwb-explorer-canvas { display: grid; grid-template-columns: minmax(0, .95fr) minmax(0, 2fr); gap: 12px; flex: 1 1 0; min-height: 0; }
.lwb-explorer-baseline figcaption { display: flex; flex-direction: column; gap: 4px; padding: 8px 0; }
.lwb-explorer-baseline figcaption span { font-size: 10px; }
.lwb-variant-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); grid-template-rows: repeat(2, minmax(0, 1fr)); gap: 12px; min-width: 0; min-height: 0; }
.lwb-variant-grid > figure { padding: 5px; background: var(--card); border: 1px solid var(--border); border-radius: 7px; }
.lwb-variant-grid > figure.picked { border-color: var(--accent-hover); }
.lwb-variant-grid figcaption { display: flex; align-items: center; justify-content: space-between; flex: 0 0 auto; gap: 5px; padding: 6px 2px 1px; text-align: left; }
.lwb-variant-grid figcaption > span { display: flex; flex-direction: column; gap: 1px; color: var(--text); font-size: 11px; }
.lwb-variant-grid figcaption small { color: var(--sub); font-size: 10px; }
.lwb-variant-grid figcaption button { min-height: 27px; padding: 4px 7px; font-size: 11px; }
.lwb-footer { min-height: 51px; padding: 8px 16px; gap: 12px; }
.lwb-footer .lwb-primary { min-height: 33px; min-width: 130px; }
.lwb-footer .lwb-actions > button { min-height: 31px; font-size: 12px; }
.lwb-file-warning { margin: 6px 12px 0; max-height: 60px; overflow: auto; padding: 5px 9px; }
@media (max-width: 1200px) {
  .lwb-studio-layout { grid-template-columns: 152px minmax(0, 1fr) 244px; }
  .lwb-studio-layout.inspector-hidden { grid-template-columns: 152px minmax(0, 1fr); }
  .lwb-studio-layout.is-explore { grid-template-columns: minmax(0, 1fr) 244px; }
  .lwb-inspector { padding: 13px 12px; }
  .lwb-header-file strong { max-width: 210px; }
  .lwb-canvas-pictures { gap: 6px; }
  .lwb-viewer { padding: 9px; }
  .lwb-header-file { gap: 7px; }
}
@media (max-width: 950px) {
  .lwb-header-file > span { display: none; }
  .lwb-studio-layout { grid-template-columns: 132px minmax(0, 1fr) 225px; }
  .lwb-studio-layout.inspector-hidden { grid-template-columns: 132px minmax(0, 1fr); }
  .lwb-studio-layout.is-explore { grid-template-columns: minmax(0, 1fr) 225px; }
  .lwb-group-top { flex-wrap: wrap; gap: 4px; }
  .lwb-canvas-toolbar { flex-wrap: wrap; }
  .lwb-toolbar button { font-size: 11px; padding: 4px 7px; }
  .lwb-heading { gap: 12px; }
  .lwb-steps { gap: 7px; }
  .lwb-mode-switch button { padding: 4px 8px; }
  .lwb-footer-status { font-size: 11px; }
}

.lwb-grouping-control { display: flex; align-items: center; justify-content: space-between; gap: 8px; flex: 0 0 auto; padding: 0 10px 9px; color: var(--sub); font-size: 11px; }
.lwb-dialog .lwb-guide-link { padding: 4px 1px; min-height: 25px; font-size: 10px; color: var(--sub); border: 0; background: transparent; }
.lwb-layer-hint { color: var(--text); line-height: 1.6; }
.lwb-freeze-range { color: var(--sub); font-size: 10px; margin-left: auto; }
.lwb-guide-intro { margin-top: 0 !important; }
.lwb-guide-group { padding: 16px 0; border-bottom: 1px solid var(--border); }
.lwb-guide-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 9px; }
.lwb-guide-heading h4 { font-size: 13px; }
.lwb-guide-heading span { display: block; margin-top: 3px; color: var(--sub); font-size: 11px; }
.lwb-guide-layer { display: grid; grid-template-columns: 35px minmax(0, 1fr); gap: 8px; padding: 5px 0; font-size: 11px; line-height: 1.6; }
.lwb-guide-layer > span:first-child { color: var(--sub); font-variant-numeric: tabular-nums; }
.lwb-guide-layer.is-absent { color: var(--sub); }
.lwb-guide-layer small { font-size: inherit; }
.lwb-guide-sources { display: flex; flex-direction: column; align-items: flex-start; gap: 9px; margin-top: 14px; font-size: 11px; line-height: 1.5; overflow-wrap: anywhere; }
.lwb-guide-sources a { color: var(--text); text-underline-offset: 3px; }
.lwb-guide-sources span { color: var(--sub); }
@media (prefers-reduced-motion: reduce) { .lwb-progress.indeterminate i { animation: none; } }
</style>
