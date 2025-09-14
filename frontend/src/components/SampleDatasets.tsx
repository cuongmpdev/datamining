import React from 'react'
import { Label } from '@/components/ui/Label'
import { Button } from '@/components/ui/Button'

type Sample = { label: string; path: string }

const DEFAULT_SAMPLES: Sample[] = [
  { label: 'play_tennis.csv', path: '/data/play_tennis.csv' },
  { label: 'kmeans_points.csv', path: '/data/kmeans_points.csv' },
  { label: 'naive_bayes_gaussian.csv', path: '/data/naive_bayes_gaussian.csv' },
  { label: 'naive_bayes_multinomial.csv', path: '/data/naive_bayes_multinomial.csv' },
]

export function SampleDatasets({ onPick, samples = DEFAULT_SAMPLES }: { onPick: (file: File) => void; samples?: Sample[] }) {
  const [loadingPath, setLoadingPath] = React.useState<string | null>(null)

  async function pickSample(s: Sample) {
    try {
      setLoadingPath(s.path)
      const res = await fetch(s.path)
      if (!res.ok) throw new Error('Không thể tải dữ liệu mẫu')
      const blob = await res.blob()
      const file = new File([blob], s.label, { type: 'text/csv' })
      onPick(file)
    } catch (e) {
      console.error(e)
      alert('Không thể tải dữ liệu mẫu')
    } finally {
      setLoadingPath(null)
    }
  }

  return (
    <div className="mt-2">
      <Label className="mb-2 block">Hoặc chọn dữ liệu mẫu</Label>
      <div className="flex flex-wrap gap-2">
        {samples.map((s) => (
          <Button key={s.path} variant="outline" size="sm" onClick={() => pickSample(s)} disabled={loadingPath !== null}>
            {loadingPath === s.path ? 'Đang tải...' : s.label}
          </Button>
        ))}
      </div>
    </div>
  )
}

