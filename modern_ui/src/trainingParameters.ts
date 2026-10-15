/** Old project values must select the same option that the engine receives. */
export function normalizeGradientCheckpointing(value: unknown): string {
  const choice = String(value ?? 'auto').trim().toLowerCase()
  if (['开启', 'on', 'true', '1'].includes(choice)) return '开启'
  if (['关闭', 'off', 'false', '0'].includes(choice)) return '关闭'
  return ['auto', '自动', ''].includes(choice) ? 'auto' : choice
}
