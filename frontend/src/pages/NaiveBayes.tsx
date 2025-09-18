import React from 'react'
import { FileUploader } from '@/components/FileUploader'
import { Button } from '@/components/ui/Button'
import { Label } from '@/components/ui/Label'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import { apiNaiveBayes, apiPreview } from '@/lib/api'
import { SampleDatasets } from '@/components/SampleDatasets'

type PreviewInfo = {
  headers: string[]
  value_samples?: Record<string, Array<string | number | null>>
}

type PosteriorInfo = {
  prior: number
  score: number
  posterior: number
  components: Array<{ feature: string; value: string; count: number; prob: number }>
}

type NaiveBayesResult = {
  target: string
  features: string[]
  evidence: Record<string, string>
  priors: Record<string, { count: number; prob: number }>
  conditionals: Record<string, { unique_count: number; values: Record<string, Record<string, { count: number; prob: number }>> }>
  posterior: Record<string, PosteriorInfo>
  prediction: string | null
  row_count: number
  classes: string[]
}

export default function NaiveBayesPage() {
  const [file, setFile] = React.useState<File | null>(null)
  const [preview, setPreview] = React.useState<PreviewInfo | null>(null)
  const [target, setTarget] = React.useState<string>('')
  const [selectedFeatures, setSelectedFeatures] = React.useState<string[]>([])
  const [evidence, setEvidence] = React.useState<Record<string, string>>({})
  const [loading, setLoading] = React.useState(false)
  const [result, setResult] = React.useState<NaiveBayesResult | null>(null)

  React.useEffect(() => {
    if (!file) {
      setPreview(null)
      setTarget('')
      setSelectedFeatures([])
      setEvidence({})
      setResult(null)
      return
    }
    apiPreview(file)
      .then((info) => {
        setPreview(info)
        if (info.headers?.length) {
          const lastHeader = info.headers[info.headers.length - 1]
          setTarget(lastHeader)
        }
        setSelectedFeatures([])
        setEvidence({})
        setResult(null)
      })
      .catch(console.error)
  }, [file])

  React.useEffect(() => {
    setSelectedFeatures((prev) => prev.filter((f) => f !== target))
    setEvidence((prev) => {
      if (!(target in prev)) return prev
      const { [target]: _removed, ...rest } = prev
      return rest
    })
  }, [target])

  const toggleFeature = (feature: string) => {
    setSelectedFeatures((prev) => {
      if (prev.includes(feature)) {
        setEvidence((ev) => {
          const { [feature]: _removed, ...rest } = ev
          return rest
        })
        return prev.filter((f) => f !== feature)
      }
      return [...prev, feature]
    })
  }

  const updateEvidence = (feature: string, value: string) => {
    setEvidence((prev) => ({ ...prev, [feature]: value }))
  }

  React.useEffect(() => {
    setResult(null)
  }, [target, selectedFeatures])

  const readyToRun = Boolean(
    file &&
      target &&
      selectedFeatures.length > 0 &&
      selectedFeatures.every((feature) => {
        const value = evidence[feature]
        return value !== undefined && value !== ''
      })
  )

  const featureOptions = React.useMemo(() => {
    if (!preview?.headers) return []
    return preview.headers.filter((h) => h !== target)
  }, [preview, target])

  const valueSamples = preview?.value_samples ?? {}

  async function onRun() {
    if (!file || !target || !readyToRun) return
    setLoading(true)
    try {
      const res = await apiNaiveBayes(file, {
        target,
        features: selectedFeatures,
        evidence: Object.fromEntries(selectedFeatures.map((f) => [f, evidence[f]])),
      })
      setResult(res)
    } catch (e) {
      console.error(e)
      alert('Lỗi khi chạy Naive Bayes')
    } finally {
      setLoading(false)
    }
  }

  const renderPriors = () => {
    if (!result) return null
    const classes = Object.keys(result.priors)
    return (
      <div className="mt-4">
        <div className="font-medium mb-2">Bước 1: Ước lượng P(C<sub>i</sub>)</div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="bg-gray-100">
                <th className="px-3 py-2 text-left">Lớp</th>
                <th className="px-3 py-2 text-left">Số lượng</th>
                <th className="px-3 py-2 text-left">P(C<sub>i</sub>)</th>
              </tr>
            </thead>
            <tbody>
              {classes.map((clazz) => (
                <tr key={clazz} className="border-t">
                  <td className="px-3 py-2">{clazz}</td>
                  <td className="px-3 py-2">{result.priors[clazz].count} / {result.row_count}</td>
                  <td className="px-3 py-2">{result.priors[clazz].prob}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  const renderConditionals = () => {
    if (!result) return null
    const classes = result.classes
    return result.features.map((feature) => {
      const info = result.conditionals[feature]
      const rows = Object.entries(info?.values ?? {})
      return (
        <div key={feature} className="mt-6">
          <div className="font-medium mb-2">
            Bước 2: P({feature} | C<sub>i</sub>)
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="bg-gray-100">
                  <th className="px-3 py-2 text-left">Giá trị</th>
                  {classes.map((clazz) => (
                    <th key={`${feature}-${clazz}`} className="px-3 py-2 text-left">
                      {clazz}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map(([value, classMap]) => (
                  <tr key={`${feature}-${value}`} className="border-t">
                    <td className="px-3 py-2">{value}</td>
                    {classes.map((clazz) => {
                      const stat = classMap[clazz] ?? { count: 0, prob: 0 }
                      const classCount = result.priors[clazz]?.count ?? 0
                      return (
                        <td key={`${feature}-${value}-${clazz}`} className="px-3 py-2">
                          {stat.count} / {classCount} → {stat.prob}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )
    })
  }

  const renderPosterior = () => {
    if (!result) return null
    const entries = Object.entries(result.posterior)
    return (
      <div className="mt-6">
        <div className="font-medium mb-2">Bước 3: Tính P(C<sub>i</sub> | X)</div>
        <div className="space-y-4 text-sm">
          {entries.map(([clazz, info]) => (
            <div key={clazz} className="border rounded-md p-3">
              <div className="font-semibold mb-2">Lớp {clazz}</div>
              <div>P({clazz}) = {info.prior}</div>
              {info.components.map((comp) => (
                <div key={`${clazz}-${comp.feature}`}>
                  P({comp.feature} = {comp.value} | {clazz}) = {comp.count} / {result.priors[clazz]?.count ?? 0} → {comp.prob}
                </div>
              ))}
              <div className="mt-2">Tích xác suất = {info.score}</div>
              <div>Kết quả chuẩn hóa: P({clazz} | X) = {info.posterior}</div>
            </div>
          ))}
        </div>
        {result.prediction && (
          <div className="mt-4 text-sm">
            <span className="font-semibold">Kết luận:</span> Chọn lớp <span className="font-semibold">{result.prediction}</span> vì có xác suất lớn nhất.
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="text-lg font-semibold">Naive Bayes</div>
          <div className="text-sm text-gray-500">Phân lớp theo công thức Bayes ngây thơ</div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
            <div className="md:col-span-2">
              <Label>Tệp dữ liệu</Label>
              <FileUploader file={file} onChange={setFile} />
              <SampleDatasets
                onPick={(f) => {
                  setFile(f)
                }}
                samples={[
                  { label: 'play_tennis.csv', path: '/data/play_tennis.csv' },
                  { label: 'naive_bayes_gaussian.csv', path: '/data/naive_bayes_gaussian.csv' },
                  { label: 'naive_bayes_multinomial.csv', path: '/data/naive_bayes_multinomial.csv' },
                ]}
              />
            </div>
            {/* <div>
              <Label className="mb-1 block">Thuộc tính mục tiêu</Label>
              <select
                className="h-9 w-full rounded-md border border-gray-300 px-3 text-sm"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
              >
                <option value="" disabled>
                  -- Chọn cột mục tiêu --
                </option>
                {preview?.headers?.map((h) => (
                  <option key={h} value={h}>
                    {h}
                  </option>
                ))}
              </select>
            </div> */}
          </div>

          {preview && (
            <div className="mt-4 space-y-4">
              <div>
                <Label className="mb-1 block">Thuộc tính sử dụng</Label>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2 text-sm">
                  {featureOptions.map((h) => (
                    <label key={h} className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={selectedFeatures.includes(h)}
                        onChange={() => toggleFeature(h)}
                      />
                      {h}
                    </label>
                  ))}
                </div>
              </div>

              {selectedFeatures.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {selectedFeatures.map((feature) => {
                    const options = valueSamples[feature] ?? []
                    return (
                      <div key={feature}>
                        <Label className="mb-1 block">Giá trị của {feature}</Label>
                        <select
                          className="h-9 w-full rounded-md border border-gray-300 px-3 text-sm"
                          value={evidence[feature] ?? ''}
                          onChange={(e) => updateEvidence(feature, e.target.value)}
                        >
                          <option value="" disabled>
                            -- Chọn giá trị --
                          </option>
                          {options.map((value, idx) => {
                            const optionValue = value == null ? '' : String(value)
                            const optionKey = `${feature}-${optionValue}-${idx}`
                            const label = optionValue === '' ? '(trống)' : optionValue
                            return (
                              <option key={optionKey} value={optionValue}>
                                {label}
                              </option>
                            )
                          })}
                        </select>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )}

          <div className="mt-4">
            <Button onClick={onRun} disabled={!readyToRun || loading}>
              {loading ? 'Đang tính...' : 'Tính toán Bayes'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <div className="font-semibold">Diễn giải kết quả</div>
            <div className="text-sm text-gray-500">
              X = ({result.features.map((f) => `${f}=${result.evidence[f] ?? '?'}`).join(', ')})
            </div>
          </CardHeader>
          <CardContent>
            {renderPriors()}
            {renderConditionals()}
            {renderPosterior()}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
