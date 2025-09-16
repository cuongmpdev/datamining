import React from 'react'
import { FileUploader } from '@/components/FileUploader'
import { Button } from '@/components/ui/Button'
import { Label } from '@/components/ui/Label'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import { apiPreview, apiReduct } from '@/lib/api'
import { SampleDatasets } from '@/components/SampleDatasets'
import { PreviewTable } from '@/components/PreviewTable'

export default function ReductPage() {
  const [file, setFile] = React.useState<File | null>(null)
  const [preview, setPreview] = React.useState<any | null>(null)
  const [decision, setDecision] = React.useState<string>('')
  const [conditional, setConditional] = React.useState<string[]>([])
  const [loading, setLoading] = React.useState(false)
  const [result, setResult] = React.useState<any | null>(null)

  React.useEffect(() => {
    if (!file) { setPreview(null); setDecision(''); setConditional([]); return }
    apiPreview(file).then((p) => {
      setPreview(p)
      const last = p.headers[p.headers.length - 1]
      setDecision(last)
      setConditional(p.headers.filter((h: string) => h !== last))
    }).catch(console.error)
  }, [file])

  async function onRun() {
    if (!file || !decision) return
    setLoading(true)
    try {
      const res = await apiReduct(file, { decision, conditional: conditional.join(',') })
      setResult(res)
    } catch (e) {
      console.error(e)
      alert('Lỗi khi tính Reduct')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="text-lg font-semibold">Rough Set Reduct</div>
          <div className="text-sm text-gray-500">Tìm tất cả Reduct tối thiểu và thuộc tính Core</div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
            <div className="md:col-span-2">
              <Label>Tệp dữ liệu</Label>
              <FileUploader file={file} onChange={setFile} />
              <SampleDatasets
                onPick={(f) => setFile(f)}
                samples={[
                  { label: 'play_tennis.csv', path: '/data/play_tennis.csv' },
                  { label: 'decision_tree_loan.csv', path: '/data/decision_tree_loan.csv' },
                  { label: 'decision_tree_mushroom.csv', path: '/data/decision_tree_mushroom.csv' },
                ]}
              />
            </div>
            <div>
              <Label className="mb-1 block">Thuộc tính quyết định</Label>
              <select className="h-9 w-full rounded-md border border-gray-300 px-3 text-sm" value={decision} onChange={(e) => setDecision(e.target.value)}>
                <option value="" disabled>-- Chọn cột --</option>
                {preview?.headers?.map((h: string) => (
                  <option key={h} value={h}>{h}</option>
                ))}
              </select>
            </div>
          </div>

          {preview && (
            <div className="mt-3">
              <Label className="mb-2 block">Chọn thuộc tính điều kiện</Label>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                {preview.headers.filter((h: string) => h !== decision).map((h: string) => (
                  <label key={h} className="flex items-center gap-2">
                    <input type="checkbox" checked={conditional.includes(h)} onChange={(e) => setConditional((prev) => e.target.checked ? [...prev, h] : prev.filter((x) => x !== h))} />
                    {h}
                  </label>
                ))}
              </div>
            </div>
          )}

          <div className="mt-4">
            <Button onClick={onRun} disabled={!file || !decision || loading}>{loading ? 'Đang chạy...' : 'Tính Reduct'}</Button>
          </div>
        </CardContent>
      </Card>

      {preview && (
        <Card>
          <CardHeader>
            <div className="font-semibold">Dữ liệu hiện tại</div>
          </CardHeader>
          <CardContent>
            <PreviewTable headers={preview.headers} rows={preview.sample} />
          </CardContent>
        </Card>
      )}

      {result && (
        <div className="space-y-4">
          {/* Kết quả tính toán */}
          <Card>
            <CardHeader>
              <div className="font-semibold text-lg text-center">Kết quả Tính Toán</div>
            </CardHeader>
            <CardContent>
              {/* Các Reduct tìm được */}
              <div className="mb-6">
                <h4 className="font-semibold text-base mb-3">Các Reduct tìm được:</h4>
                {result.all_reducts && result.all_reducts.length > 0 ? (
                  <ul className="list-disc list-inside space-y-2">
                    {result.all_reducts.map((reduct: string[], index: number) => (
                      <li key={index} className="text-sm bg-blue-50 rounded px-3 py-2">
                        <span className="font-medium">Reduct {index + 1}:</span> [{reduct.join(', ')}]
                        <span className="text-gray-600 text-xs ml-2">(Kích thước: {reduct.length})</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-gray-500 italic">Không tìm thấy reduct.</p>
                )}
              </div>

              {/* Core Attributes */}
              {result.core && result.core.length > 0 && (
                <div className="mb-6">
                  <h4 className="font-semibold text-base mb-2">Core:</h4>
                  <div className="bg-green-50 rounded px-3 py-2">
                    <span className="font-medium text-green-800">Thuộc tính core:</span> [{result.core.join(', ')}]
                    <div className="text-xs text-green-600 mt-1">
                      (Thuộc tính xuất hiện trong tất cả reduct)
                    </div>
                  </div>
                </div>
              )}

              {/* Thống kê tổng quan */}
              <div className="mb-6">
                <h4 className="font-semibold text-base mb-3">Thống kê:</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                  <div className="bg-blue-50 rounded p-3">
                    <div className="font-medium text-blue-800">Tổng Reduct</div>
                    <div className="text-xl font-bold text-blue-900">{result.total_reducts_found || 0}</div>
                  </div>
                  <div className="bg-green-50 rounded p-3">
                    <div className="font-medium text-green-800">Core Size</div>
                    <div className="text-xl font-bold text-green-900">{result.core ? result.core.length : 0}</div>
                  </div>
                  <div className="bg-purple-50 rounded p-3">
                    <div className="font-medium text-purple-800">γ(R)</div>
                    <div className="text-xl font-bold text-purple-900">{result.gamma_R?.toFixed(3)}</div>
                  </div>
                  <div className="bg-orange-50 rounded p-3">
                    <div className="font-medium text-orange-800">γ(C)</div>
                    <div className="text-xl font-bold text-orange-900">{result.gamma_C?.toFixed(3)}</div>
                  </div>
                </div>
              </div>

              {/* Equivalence Classes */}
              {result.equivalence_classes && result.equivalence_classes.length > 0 && (
                <div className="mb-6">
                  <h4 className="font-semibold text-base mb-3">Lớp tương đương:</h4>
                  <div className="space-y-3">
                    {result.equivalence_classes.map((eqClass: any, index: number) => (
                      <div key={index} className="border rounded-lg p-3 bg-gray-50">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium">
                            Lớp {eqClass.class_id} ({eqClass.size} đối tượng)
                          </span>
                          <span className={`px-2 py-1 rounded text-xs ${eqClass.is_consistent
                            ? 'bg-green-100 text-green-800'
                            : 'bg-red-100 text-red-800'
                            }`}>
                            {eqClass.is_consistent ? 'Nhất quán' : 'Không nhất quán'}
                          </span>
                        </div>
                        <div className="text-sm text-gray-600">
                          <div className="mb-1">
                            <span className="font-medium">Quyết định:</span> {eqClass.decisions.join(', ')}
                          </div>
                          <div>
                            <span className="font-medium">Đối tượng:</span>
                            <div className="mt-1 space-y-1">
                              {eqClass.objects?.map((obj: any, objIndex: number) => (
                                <div key={objIndex} className="text-xs bg-white rounded px-2 py-1">
                                  Object {obj.index + 1}: {Object.entries(obj)
                                    .filter(([key]) => key !== 'index' && key !== 'decision')
                                    .map(([key, value]) => `${key}=${value}`)
                                    .join(', ')}
                                  → {obj.decision}
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Thông tin chi tiết */}
              {result.reduct_summary && (
                <div className="bg-gray-100 rounded-lg p-4">
                  <h4 className="font-semibold text-base mb-2">Thông tin chi tiết:</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div>
                      <p><span className="font-medium">Reduct chính:</span> [{result.reduct.join(', ')}]</p>
                      <p><span className="font-medium">Kích thước reduct:</span> {result.reduct_summary.primary_reduct_size}</p>
                      <p><span className="font-medium">Độ cải thiện phụ thuộc:</span> {result.reduct_summary.dependency_improvement?.toFixed(4)}</p>
                    </div>
                    <div>
                      <p><span className="font-medium">Reduct đầy đủ:</span> {result.reduct_summary.is_complete_reduct ? 'Có' : 'Không'}</p>
                      <p><span className="font-medium">Thuộc tính điều kiện:</span> {conditional.join(', ')}</p>
                      <p><span className="font-medium">Thuộc tính quyết định:</span> {decision}</p>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
