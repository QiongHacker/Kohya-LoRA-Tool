import type { LoRAInfo } from './bridge'
import guide from '../../kohya_core/anima_layer_guide.json'

// The worker reads this same catalog; IDs and ranges must match real ablations.
export const ANIMA_GUIDE_SOURCE = guide.source
export const ANIMA_MOD_GUIDANCE_SOURCE = guide.modulationSource
export const ANIMA_GROUPING = guide.grouping
export const ANIMA_LAYER_GUIDE = guide.layers.map((hint, index) => ({
  id: `block_${index}`, label: `L${String(index).padStart(2, '0')}`, index, hint,
}))
export const ANIMA_GUIDE_GROUPS = guide.groups.map(group => ({
  ...group,
  range: `L${String(group.start).padStart(2, '0')}–L${String(group.end).padStart(2, '0')}`,
  layers: ANIMA_LAYER_GUIDE.filter(layer => layer.index >= group.start && layer.index <= group.end),
}))

export function animaLayerHint(id: string): string {
  return ANIMA_LAYER_GUIDE.find(layer => layer.id === id)?.hint || ''
}

export function supportsAnimaGuide(info: LoRAInfo | null): boolean {
  if (!info || info.family !== 'anima') return false
  const ids = new Set(info.groups.flatMap(group => group.blocks.map(block => block.id)))
  // Old sparse reports need a fresh inspection before enabling the 28-layer map.
  if (info.architecture_blocks == null && !ANIMA_LAYER_GUIDE.every(layer => ids.has(layer.id))) return false
  if (info.architecture_blocks != null && info.architecture_blocks !== 28) return false
  return [...ids].some(id => !!animaLayerHint(id)) &&
    [...ids].every(id => !/^block_/.test(id) || !!animaLayerHint(id))
}
