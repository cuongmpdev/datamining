import React from 'react'
import { FileUploader } from '@/components/FileUploader'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Label } from '@/components/ui/Label'
import { Card, CardContent, CardHeader } from '@/components/ui/Card'
import { apiKMeans, apiPreview } from '@/lib/api'
import { PreviewTable } from '@/components/PreviewTable'
import { SampleDatasets } from '@/components/SampleDatasets'

export default function KMeansPage() {
  const [file, setFile] = React.useState<File | null>(null)
  const [preview, setPreview] = React.useState<any | null>(null)
  const [k, setK] = React.useState(3)
  const [columns, setColumns] = React.useState<string[]>([])
  const [loading, setLoading] = React.useState(false)
  const [result, setResult] = React.useState<any | null>(null)
  const [originalData, setOriginalData] = React.useState<number[][] | null>(null)

  React.useEffect(() => {
    if (!file) { setPreview(null); setColumns([]); return }
    apiPreview(file).then((p) => {
      setPreview(p)
      const numeric = Object.entries(p.types as Record<string, string>)
        .filter(([_, t]) => t === 'numeric')
        .map(([h]) => h)
      setColumns(numeric)
    }).catch((e) => console.error(e))
  }, [file])

  async function onRun() {
    if (!file || !preview) return
    setLoading(true)
    try {
      // Extract original data points from preview for display purposes
      const dataPoints = preview.sample.map((row: any) =>
        columns.map((col: string) => parseFloat(row[col]) || 0)
      )
      setOriginalData(dataPoints)

      const res = await apiKMeans(file, { k, columns: columns.join(',') })
      setResult(res)
    } catch (e) {
      console.error(e)
      alert('Lỗi khi chạy KMeans')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="text-lg font-semibold">K-Means Clustering</div>
          <div className="text-sm text-gray-500">Chọn tệp CSV và tham số</div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
            <div className="md:col-span-2">
              <Label>Tệp dữ liệu</Label>
              <FileUploader file={file} onChange={setFile} />
              <SampleDatasets
                onPick={(f) => setFile(f)}
                samples={[
                  { label: 'kmeans_points.csv', path: '/data/kmeans_points.csv' },
                  { label: 'kmeans_points_b.csv', path: '/data/kmeans_points_b.csv' },
                  { label: 'kmeans_points_c.csv', path: '/data/kmeans_points_c.csv' },
                ]}
              />
            </div>
            <div>
              <Label className="mb-1 block">Số cụm (k)</Label>
              <Input type="number" min={1} value={k} onChange={(e) => setK(Number(e.target.value))} />
            </div>
          </div>

          {preview && (
            <div className="mt-4">
              <div className="font-medium mb-2">Chọn thuộc tính số dùng để gom cụm</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {preview.headers.map((h: string) => (
                  <label key={h} className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={columns.includes(h)}
                      disabled={preview.types[h] !== 'numeric'}
                      onChange={(e) => {
                        setColumns((prev) => e.target.checked ? [...prev, h] : prev.filter((x) => x !== h))
                      }}
                    />
                    <span>{h} {preview.types[h] !== 'numeric' && <em className="text-gray-400">(không phải số)</em>}</span>
                  </label>
                ))}
              </div>
            </div>
          )}

          <div className="mt-4">
            <Button onClick={onRun} disabled={!file || loading}>{loading ? 'Đang chạy...' : 'Chạy K-Means'}</Button>
          </div>
        </CardContent>
      </Card>

      {preview && (
        <Card>
          <CardHeader>
            <div className="font-semibold">Xem nhanh dữ liệu</div>
          </CardHeader>
          <CardContent>
            <PreviewTable headers={preview.headers} rows={preview.sample} />
          </CardContent>
        </Card>
      )}

      {result && originalData && (
        <Card>
          <CardHeader>
            <div className="font-semibold">Kết quả phân cụm</div>
            <div className="text-sm text-gray-600">
              Inertia: <b>{result.inertia.toFixed(4)}</b> | Số lần lặp: <b>{result.iterations}</b>
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse border border-gray-200 rounded-lg">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="border border-gray-200 px-4 py-3 text-left font-medium text-gray-700">
                      Cụm
                    </th>
                    <th className="border border-gray-200 px-4 py-3 text-left font-medium text-gray-700">
                      Các điểm trong cụm
                    </th>
                    <th className="border border-gray-200 px-4 py-3 text-left font-medium text-gray-700">
                      Trọng tâm
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result.centroids.map((centroid: number[], clusterIndex: number) => {
                    // Group points by cluster
                    const clusterPoints = originalData
                      .map((point, pointIndex) => ({ point, pointIndex }))
                      .filter(({ pointIndex }) => result.labels[pointIndex] === clusterIndex)
                      .map(({ point, pointIndex }) => ({ point, pointIndex: pointIndex + 1 }));

                    return (
                      <tr key={clusterIndex} className="hover:bg-gray-50">
                        <td className="border border-gray-200 px-4 py-3 font-medium text-center">
                          <div className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 text-blue-800 font-semibold">
                            {clusterIndex + 1}
                          </div>
                        </td>
                        <td className="border border-gray-200 px-4 py-3">
                          <div className="space-y-1">
                            {clusterPoints.length > 0 ? (
                              clusterPoints.map(({ point, pointIndex }) => (
                                <div key={pointIndex} className="text-sm bg-gray-100 rounded px-2 py-1 inline-block mr-2 mb-1">
                                  Điểm {pointIndex}: [{point.map(v => v.toFixed(2)).join(', ')}]
                                </div>
                              ))
                            ) : (
                              <span className="text-gray-500 italic">Không có điểm nào</span>
                            )}
                          </div>
                          <div className="mt-2 text-xs text-gray-600">
                            Số điểm: <b>{clusterPoints.length}</b>
                          </div>
                        </td>
                        <td className="border border-gray-200 px-4 py-3">
                          <div className="bg-green-50 rounded px-3 py-2 font-mono text-sm">
                            [{centroid.map(v => v.toFixed(4)).join(', ')}]
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Summary statistics */}
            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="bg-blue-50 rounded p-3">
                <div className="font-medium text-blue-800">Tổng số cụm</div>
                <div className="text-xl font-bold text-blue-900">{result.centroids.length}</div>
              </div>
              <div className="bg-green-50 rounded p-3">
                <div className="font-medium text-green-800">Tổng số điểm</div>
                <div className="text-xl font-bold text-green-900">{originalData.length}</div>
              </div>
              <div className="bg-purple-50 rounded p-3">
                <div className="font-medium text-purple-800">Inertia</div>
                <div className="text-xl font-bold text-purple-900">{result.inertia.toFixed(2)}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
