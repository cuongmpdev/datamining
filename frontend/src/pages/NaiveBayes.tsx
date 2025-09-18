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

type PriorInfo = {
  count: number
  prob: number
  numerator: number
  denominator: number
}

type ConditionalStat = {
  count: number
  prob: number
  numerator: number
  denominator: number
}

type ConditionalInfo = {
  unique_count: number
  values: Record<string, Record<string, ConditionalStat>>
}

type PosteriorComponent = {
  feature: string
  value: string | number
  count: number
  prob: number
  numerator: number
  denominator: number
  unique_count: number
}

type PosteriorInfo = {
  prior: number
  score: number
  posterior: number
  components: PosteriorComponent[]
  prior_numerator: number
  prior_denominator: number
  score_raw: number
  posterior_raw: number
}

type NaiveBayesResult = {
  target: string
  features: string[]
  evidence: Record<string, string>
  priors: Record<string, PriorInfo>
  conditionals: Record<string, ConditionalInfo>
  posterior: Record<string, PosteriorInfo>
  prediction: string | null
  row_count: number
  classes: string[]
  laplace: number
}

const DEFAULT_LAPLACE = 1

export default function NaiveBayesPage() {
  const [file, setFile] = React.useState<File | null>(null)
  const [preview, setPreview] = React.useState<PreviewInfo | null>(null)
  const [target, setTarget] = React.useState<string>('')
  const [selectedFeatures, setSelectedFeatures] = React.useState<string[]>([])
  const [evidence, setEvidence] = React.useState<Record<string, string>>({})
  const [loading, setLoading] = React.useState(false)
  const [result, setResult] = React.useState<NaiveBayesResult | null>(null)
  const [pendingLaplace, setPendingLaplace] = React.useState<number | null>(null)

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

  async function runNaiveBayes(laplaceValue: number) {
    if (!file || !target || !readyToRun) return
    setLoading(true)
    setPendingLaplace(laplaceValue)
    try {
      const res = await apiNaiveBayes(file, {
        target,
        features: selectedFeatures,
        evidence: Object.fromEntries(selectedFeatures.map((f) => [f, evidence[f]])),
        laplace: laplaceValue,
      })
      setResult(res)
    } catch (e) {
      console.error(e)
      alert('Lỗi khi chạy Naive Bayes')
    } finally {
      setLoading(false)
      setPendingLaplace(null)
    }
  }

  const renderPriors = () => {
    if (!result) return null
    const classes = Object.keys(result.priors)
    const laplace = result.laplace
    const classTotal = result.classes.length
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
              {classes.map((clazz) => {
                const info = result.priors[clazz]
                return (
                  <tr key={clazz} className="border-t">
                    <td className="px-3 py-2">{clazz}</td>
                    <td className="px-3 py-2">{info.count} / {result.row_count}</td>
                    <td className="px-3 py-2">
                      {laplace > 0 ? (
                        <>
                          ({info.count} + {laplace}) / ({result.row_count} + {laplace} * {classTotal}) = {info.numerator} / {info.denominator} = {info.prob}
                        </>
                      ) : (
                        <>
                          {info.count} / {result.row_count} = {info.prob}
                        </>
                      )}
                    </td>
                  </tr>
                )
              })}
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
      const uniqueCount = info?.unique_count ?? 0
      return (
        <div key={feature} className="mt-6">
          <div className="font-medium mb-2">
            Bước 2: P({feature} | C<sub>i</sub>)
            {result.laplace > 0 && (
              <span className="text-gray-500"> — r = {uniqueCount}</span>
            )}
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
                      const stat = classMap[clazz] ?? {
                        count: 0,
                        prob: 0,
                        numerator: result.laplace,
                        denominator: (result.priors[clazz]?.count ?? 0) + result.laplace * uniqueCount,
                      }
                      const classCount = result.priors[clazz]?.count ?? 0
                      return (
                        <td key={`${feature}-${value}-${clazz}`} className="px-3 py-2">
                          {result.laplace > 0 ? (
                            <>
                              ({stat.count} + {result.laplace}) / ({classCount} + {result.laplace} * {uniqueCount}) = {stat.numerator} / {stat.denominator} = {stat.prob}
                            </>
                          ) : (
                            <>
                              {stat.count} / {classCount} = {stat.prob}
                            </>
                          )}
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

  const renderLaplaceOverview = () => {
    if (!result || result.laplace <= 0) return null
    return (
      <div className="mt-4 text-sm space-y-1">
        <div className="font-medium">Làm trơn Laplace (α = {result.laplace})</div>
        <div>Áp dụng để tránh trường hợp P(X<sub>k</sub> | C<sub>i</sub>) = 0 khi đếm bằng 0.</div>
        <div>
          P(C<sub>i</sub>) = (|C<sub>i</sub>,D| + α) / (|D| + α * m), với m = {result.classes.length}.
        </div>
        <div>
          P(X<sub>k</sub> = v | C<sub>i</sub>) = (số mẫu v trong C<sub>i</sub> + α) / (|C<sub>i</sub>,D| + α * r<sub>k</sub>), trong đó r<sub>k</sub> là số giá trị rời rạc của thuộc tính.
        </div>
      </div>
    )
  }

  const renderPosterior = () => {
    if (!result) return null
    const entries = Object.entries(result.posterior)
    const laplace = result.laplace
    const formatValue = (value: number) => (Number.isFinite(value) ? Number(value).toFixed(6) : '0')
    return (
      <div className="mt-6">
        <div className="font-medium mb-2">Bước 3: Tính P(C<sub>i</sub> | X)</div>
        <div className="space-y-4 text-sm">
          {entries.map(([clazz, info]) => {
            const priorInfo = result.priors[clazz]
            const conditionalFractions = info.components.map((comp) => `${comp.numerator}/${comp.denominator || 1}`)
            const productFraction = [
              `${info.prior_numerator}/${info.prior_denominator || 1}`,
              ...conditionalFractions,
            ].join(' × ')
            const productProb = [info.prior, ...info.components.map((comp) => comp.prob)].join(' × ')
            return (
              <div key={clazz} className="border rounded-md p-3 space-y-1">
                <div className="font-semibold mb-2">Lớp {clazz}</div>
                <div>
                  1.{' '}
                  {laplace > 0 ? (
                    <>
                      P({clazz}) = ({priorInfo.count} + {laplace}) / ({result.row_count} + {laplace} * {result.classes.length}) = {info.prior_numerator} / {info.prior_denominator} = {info.prior}
                    </>
                  ) : (
                    <>
                      P({clazz}) = {priorInfo.count} / {result.row_count} = {info.prior}
                    </>
                  )}
                </div>
                {info.components.map((comp, index) => (
                  <div key={`${clazz}-${comp.feature}`}>
                    {index + 2}.{' '}
                    {laplace > 0 ? (
                      <>
                        P({comp.feature} = {comp.value} | {clazz}) = ({comp.count} + {laplace}) / ({priorInfo.count} + {laplace} * {comp.unique_count}) = {comp.numerator} / {comp.denominator} = {comp.prob}
                      </>
                    ) : (
                      <>
                        P({comp.feature} = {comp.value} | {clazz}) = {comp.count} / {priorInfo.count} = {comp.prob}
                      </>
                    )}
                  </div>
                ))}
                <div className="mt-2">
                  {info.components.length + 2}. Tích xác suất (không chuẩn hóa) = {productFraction} = {productProb} = {formatValue(info.score_raw)}
                </div>
                <div>
                  {info.components.length + 3}. Kết quả chuẩn hóa: P({clazz} | X) = {formatValue(info.posterior_raw)} ≈ {info.posterior}
                </div>
              </div>
            )
          })}
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

          <div className="mt-4 flex flex-wrap gap-2">
            <Button onClick={() => runNaiveBayes(0)} disabled={!readyToRun || loading}>
              {loading && pendingLaplace === 0 ? 'Đang tính...' : 'Tính toán Bayes'}
            </Button>
            <Button
              variant="outline"
              onClick={() => runNaiveBayes(DEFAULT_LAPLACE)}
              disabled={!readyToRun || loading}
            >
              {loading && pendingLaplace === DEFAULT_LAPLACE ? 'Đang làm trơn...' : 'Làm trơn Laplace'}
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
            {renderLaplaceOverview()}
            {renderPriors()}
            {renderConditionals()}
            {renderPosterior()}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
